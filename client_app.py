import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc

import torch
import torch.quantization

from flwr.clientapp import ClientApp
from flwr.app import Context, Message, RecordDict, ArrayRecord, MetricRecord

from task import load_data, load_model, train, test, get_model_size, DEVICE


app = ClientApp()

# ── Per-client persistent state ───────────────────────────────────────────────
# We cache the model and dataloaders to avoid re-loading datasets every round.
# Unlike FedBN, ALL layers (including GN) are synchronised with the server.
_local_state: dict = {}   # node_id → {"net": net, "trainloader": ..., "testloader": ...}


def _get_or_init_local_state(node_id, num_clients, batch_size, alpha=0.0, seed=42):
    """Lazily create the persistent model and dataloaders for a node.

    Called at most once per node per process lifetime.
    The model is a standard Net() — no special treatment of GN layers.
    """
    if node_id not in _local_state:
        net = load_model()
        trainloader, testloader = load_data(
            node_id=node_id, num_clients=num_clients,
            batch_size=batch_size, alpha=alpha, seed=seed,
        )
        _local_state[node_id] = {
            "net":         net,
            "trainloader": trainloader,
            "testloader":  testloader,
        }
    return _local_state[node_id]


# ── Train ─────────────────────────────────────────────────────────────────────
@app.train()
def train_handler(msg: Message, context: Context) -> Message:
    """FedGN (FedAvg baseline) local training step.

    Unlike FedBN, ALL model weights — including GroupNorm layers — are
    synchronised with the global server model every round.  GroupNorm
    parameters are therefore shared across all clients.  This serves as
    the non-personalised baseline to measure the marginal benefit of FedBN
    and Ditto personalisation strategies.

    Workflow
    --------
    1. Receive ALL global weights from the server (including GN layers).
    2. Fully overwrite the local model.
    3. Train with standard SGD (no proximal term, no DP).
    4. Return ALL weights (including GN) to the server.
    """
    start_time = time.time()
    tracemalloc.start()

    node_id     = context.node_config.get("partition-id", 0)
    num_clients = context.node_config.get("num-partitions", 10)
    rc          = context.run_config

    batch_size = int(rc.get("batch-size",      16))
    epochs     = int(rc.get("local-epochs",    2))
    lr         = float(rc.get("lr-local",      0.01))
    momentum   = float(rc.get("momentum",      0.9))
    alpha      = float(rc.get("alpha",         0.0))
    seed       = int(rc.get("partition-seed",  42))

    # 1. Receive ALL global weights (including GN) from server
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # 2. Get (or lazily create) persistent local state for this partition
    state = _get_or_init_local_state(node_id, num_clients, batch_size, alpha, seed)

    # Fully overwrite local model with global weights (FedGN: all layers synced)
    state["net"].load_state_dict(
        {k: v.to(DEVICE) for k, v in incoming_state_dict.items()}, strict=True
    )

    # 3. Train locally with standard SGD
    train(state["net"], state["trainloader"], epochs=epochs, lr=lr, momentum=momentum)

    # 4. Return ALL weights (including GN) to the server
    full_state = {k: v.cpu() for k, v in state["net"].state_dict().items()}
    model_record = ArrayRecord(torch_state_dict=full_state)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    comm_size_mb = sum(
        p.element_size() * p.nelement() for p in full_state.values()
    ) / (1024 * 1024)
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    fit_time         = time.time() - start_time
    peak_ram_mb      = peak_ram / (1024 * 1024)
    estimated_energy = (2.0 * fit_time) + (15.0 * comm_size_mb)

    metrics = {
        "fit_time":         float(fit_time),
        "peak_ram_mb":      float(peak_ram_mb),
        "comm_size_mb":     float(comm_size_mb),
        "model_size_mb":    float(get_model_size(state["net"])),
        "dp_epsilon":       0.0,
        "estimated_energy": float(estimated_energy),
        "num-examples":     float(len(state["trainloader"].dataset)),
    }

    content = RecordDict({
        "arrays":  model_record,
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)


# ── Evaluate ──────────────────────────────────────────────────────────────────
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Three-level evaluation: global FP32 / local FP32 / local INT8 (TinyML proxy).

    For FedGN, global model == local model (no personalisation).
    local_vs_global_gap is therefore expected to be ~0 by design.
    This is the non-personalised baseline — compare it with FedBN/Ditto
    to quantify the benefit of personalisation strategies.
    """
    start_time = time.time()

    node_id     = context.node_config.get("partition-id", 0)
    num_clients = context.node_config.get("num-partitions", 10)
    rc          = context.run_config

    batch_size = int(rc.get("batch-size",     16))
    alpha      = float(rc.get("alpha",        0.0))
    seed       = int(rc.get("partition-seed", 42))

    # Receive ALL global weights (including GN) from server
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # ── EVAL 1 — Global model ─────────────────────────────────────────────────
    # For FedGN the global model has shared GN layers — inject all weights.
    model_global = load_model()
    model_global.load_state_dict(
        {k: v.to(DEVICE) for k, v in incoming_state_dict.items()}, strict=True
    )

    if node_id in _local_state:
        testloader = _local_state[node_id]["testloader"]
    else:
        _, testloader = load_data(
            node_id=node_id, num_clients=num_clients,
            batch_size=batch_size, alpha=alpha, seed=seed,
        )

    _, acc_global = test(model_global, testloader)

    # ── EVAL 2 — "Local" model ────────────────────────────────────────────────
    # For FedGN there is no separate personalised model: after training the
    # local net IS the just-trained global net.  We reuse model_global here
    # to make the gap explicit (it will be ~0 — intentional for a baseline).
    model_local   = model_global
    _, acc_local_fp32 = test(model_local, testloader)

    # ── EVAL 3 — INT8 dynamic quantisation (TinyML deployment proxy) ──────────
    net_cpu = load_model().cpu()
    net_cpu.load_state_dict({k: v.cpu() for k, v in model_local.state_dict().items()})
    net_quantized = torch.quantization.quantize_dynamic(
        net_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    loss_quantized, acc_quantized = test(
        net_quantized, testloader, device=torch.device("cpu")
    )

    eval_time           = time.time() - start_time
    local_vs_global_gap = acc_local_fp32 - acc_global   # ~0 by design for FedGN
    quantization_error  = acc_local_fp32 - acc_quantized

    metrics = {
        "accuracy":                float(acc_quantized),
        "acc_global":              float(acc_global),
        "acc_local_fp32":          float(acc_local_fp32),
        "local_vs_global_gap":     float(local_vs_global_gap),
        "quantization_error":      float(quantization_error),
        "eval_time":               float(eval_time),
        "quantized_model_size_mb": float(get_model_size(net_quantized)),
        "loss":                    float(loss_quantized),
        "num-examples":            float(len(testloader.dataset)),
    }

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
