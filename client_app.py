import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
import torch
import torch.quantization
from collections import OrderedDict

from flwr.clientapp import ClientApp
from flwr.app import Context, Message, RecordDict, ArrayRecord, MetricRecord

from task import load_data, load_model, train, test, get_model_size, DEVICE

app = ClientApp()

# ── Per-client persistent state ───────────────────────────────────────────────
# Local models with their personal GroupNorm layers must live across rounds.
_local_models: dict = {}   # node_id → { "net": net, "trainloader": trainloader, "testloader": testloader }


def _get_or_init_local_model(node_id, num_clients, batch_size):
    """Return (or lazily create) the persistent local model and dataloaders for a client."""
    if node_id not in _local_models:
        net = load_model()
        trainloader, testloader = load_data(
            node_id=node_id, num_clients=num_clients, batch_size=batch_size
        )
        _local_models[node_id] = {
            "net": net,
            "trainloader": trainloader,
            "testloader": testloader,
        }
    return _local_models[node_id]


# ── Train ─────────────────────────────────────────────────────────────────────
@app.train()
def train_handler(msg: Message, context: Context) -> Message:
    """FedBN local training step.

    Workflow
    --------
    1. Receive global weights (non-GN layers only) from the server.
    2. Inject them into the local model — GN layers remain local (FedBN).
    3. Train with standard SGD (no proximal term, no DP).
    4. Return only non-GN weights + KPI metrics.
    """
    start_time = time.time()
    tracemalloc.start()

    node_id        = context.node_config.get("partition-id", 0)
    num_clients    = context.node_config.get("num-partitions", 10)
    rc             = context.run_config

    batch_size     = int(rc.get("batch-size",       16))
    epochs         = int(rc.get("local-epochs",     2))
    lr             = float(rc.get("lr-local",       0.01))
    momentum       = float(rc.get("momentum",        0.9))

    # 1. Receive global weights — ArrayRecord → torch state dict
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # 2. Get (or lazily create) persistent local model for this partition
    state = _get_or_init_local_model(node_id, num_clients, batch_size)

    # Inject server weights into non-GN layers only (FedBN: GN stays local)
    local_state = state["net"].state_dict()
    for k in local_state:
        if "gn" not in k and k in incoming_state_dict:
            local_state[k] = incoming_state_dict[k].to(DEVICE)
    state["net"].load_state_dict(local_state, strict=True)

    # 3. Train
    train(state["net"], state["trainloader"], epochs=epochs, lr=lr, momentum=momentum)

    # 4. Return only non-GN weights (server aggregates only these)
    state_to_return = {
        k: v.cpu() for k, v in state["net"].state_dict().items()
        if "gn" not in k
    }
    model_record = ArrayRecord(torch_state_dict=state_to_return)

    # KPIs
    comm_size_mb = sum(
        p.element_size() * p.nelement() for p in state_to_return.values()
    ) / (1024 * 1024)
    _, peak_ram  = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    fit_time     = time.time() - start_time
    peak_ram_mb  = peak_ram / (1024 * 1024)
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

    acc_global     : global model accuracy (no local FedBN adaptation)
    acc_local_fp32 : local model accuracy  (with local GN layers — FedBN personalisation gain)
    quantization_error: accuracy drop from INT8 dynamic quantisation
    """
    start_time = time.time()

    node_id     = context.node_config.get("partition-id", 0)
    num_clients = context.node_config.get("num-partitions", 10)
    rc          = context.run_config
    batch_size  = int(rc.get("batch-size", 16))

    # Receive server weights (non-GN only)
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # EVAL 1 — Global model: inject server weights into a fresh model
    #   (fresh model has random GN weights → purely global performance)
    model_global = load_model()
    global_state = model_global.state_dict()
    for k in global_state:
        if "gn" not in k and k in incoming_state_dict:
            global_state[k] = incoming_state_dict[k].to(DEVICE)
    model_global.load_state_dict(global_state, strict=True)

    # EVAL 2 — Local model: inject server weights into the TRAINED model
    #   (keeps locally-trained GN weights → FedBN personalisation benefit)
    if node_id in _local_models:
        model_local = _local_models[node_id]["net"]
        local_state = model_local.state_dict()
        for k in local_state:
            if "gn" not in k and k in incoming_state_dict:
                local_state[k] = incoming_state_dict[k].to(DEVICE)
        model_local.load_state_dict(local_state, strict=True)
        testloader = _local_models[node_id]["testloader"]
    else:
        # Fallback before first training round
        model_local = model_global
        _, testloader = load_data(node_id=node_id, num_clients=num_clients,
                                  batch_size=batch_size)

    _, acc_global     = test(model_global, testloader)
    _, acc_local_fp32 = test(model_local,  testloader)

    # EVAL 3 — INT8 dynamic quantisation (TinyML deployment proxy)
    net_cpu = load_model().cpu()
    net_cpu.load_state_dict({k: v.cpu() for k, v in model_local.state_dict().items()})
    net_quantized = torch.quantization.quantize_dynamic(
        net_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    loss_quantized, acc_quantized = test(
        net_quantized, testloader, device=torch.device("cpu")
    )

    eval_time           = time.time() - start_time
    local_vs_global_gap = acc_local_fp32 - acc_global
    quantization_error  = acc_local_fp32 - acc_quantized

    metrics = {
        "accuracy":               float(acc_quantized),
        "acc_global":             float(acc_global),
        "acc_local_fp32":         float(acc_local_fp32),
        "local_vs_global_gap":    float(local_vs_global_gap),
        "quantization_error":     float(quantization_error),
        "eval_time":              float(eval_time),
        "quantized_model_size_mb": float(get_model_size(net_quantized)),
        "loss":                   float(loss_quantized),
        "num-examples":           float(len(testloader.dataset)),
    }

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
