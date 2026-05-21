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

# ── Per-client persistent state (context.state) ────────────────────────────────
# Local models with their personal GroupNorm layers must live across rounds.
# We store them in context.state as "gn_weights" (an ArrayRecord) to persist
# across different stateless Ray worker processes/actors.


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
    alpha          = float(rc.get("alpha",           0.0))
    seed           = int(rc.get("partition-seed",   42))

    # 1. Receive global weights — ArrayRecord → torch state dict
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # 2. Load the data statelessly
    trainloader, _, _ = load_data(
        node_id=node_id, num_clients=num_clients,
        batch_size=batch_size, alpha=alpha, seed=seed,
    )

    # 3. Instantiate fresh model and load local GN weights if present in context.state
    net = load_model()
    local_state = net.state_dict()

    # Load persistent GroupNorm layers from context.state
    has_persisted_gn = "gn_weights" in context.state
    if has_persisted_gn:
        gn_params = context.state["gn_weights"].to_torch_state_dict()
        for k, v in gn_params.items():
            local_state[k] = v.to(DEVICE)

    # Inject server weights into non-GN layers only (FedBN: GN stays local)
    for k in local_state:
        if "gn" not in k and k in incoming_state_dict:
            local_state[k] = incoming_state_dict[k].to(DEVICE)
    net.load_state_dict(local_state, strict=True)

    # Log state loading diagnostic
    import os
    print(f"[FIT PID {os.getpid()}] Client {node_id}: loaded_persisted_gn={has_persisted_gn}")

    # 4. Train
    train(net, trainloader, epochs=epochs, lr=lr, momentum=momentum)

    # 5. Persist the updated local GroupNorm weights back to context.state
    gn_params = {k: v.cpu() for k, v in net.state_dict().items() if "gn" in k}
    context.state["gn_weights"] = ArrayRecord(torch_state_dict=gn_params)

    # 6. Return only non-GN weights (server aggregates only these)
    state_to_return = {
        k: v.cpu() for k, v in net.state_dict().items()
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
        "model_size_mb":    float(get_model_size(net)),
        "dp_epsilon":       0.0,
        "estimated_energy": float(estimated_energy),
        "num-examples":     float(len(trainloader.dataset)),
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
    alpha       = float(rc.get("alpha", 0.0))
    seed        = int(rc.get("partition-seed", 42))

    # Receive server weights (non-GN only)
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # Load testing data statelessly
    _, testloader_global, testloader_local = load_data(
        node_id=node_id, num_clients=num_clients,
        batch_size=batch_size, alpha=alpha, seed=seed,
    )

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
    model_local = load_model()
    local_state = model_local.state_dict()

    # Load persistent GroupNorm layers from context.state
    has_persisted_gn = "gn_weights" in context.state
    if has_persisted_gn:
        gn_params = context.state["gn_weights"].to_torch_state_dict()
        for k, v in gn_params.items():
            local_state[k] = v.to(DEVICE)

    # Inject server weights into non-GN layers
    for k in local_state:
        if "gn" not in k and k in incoming_state_dict:
            local_state[k] = incoming_state_dict[k].to(DEVICE)
    model_local.load_state_dict(local_state, strict=True)

    # Log state loading diagnostic
    import os
    print(f"[EVAL PID {os.getpid()}] Client {node_id}: loaded_persisted_gn={has_persisted_gn}")

    _, acc_global            = test(model_global, testloader_global)  # global dist
    _, acc_local_fp32        = test(model_local,  testloader_global)  # global dist
    _, acc_local_on_local    = test(model_local,  testloader_local)   # local dist
    _, acc_global_on_local   = test(model_global, testloader_local)   # local dist

    # EVAL 3 — INT8 dynamic quantisation (TinyML deployment proxy)
    net_cpu = load_model().cpu()
    net_cpu.load_state_dict({k: v.cpu() for k, v in model_local.state_dict().items()})
    net_quantized = torch.quantization.quantize_dynamic(
        net_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    loss_quantized, acc_quantized = test(
        net_quantized, testloader_global, device=torch.device("cpu")
    )

    eval_time                 = time.time() - start_time
    local_vs_global_gap       = acc_local_fp32 - acc_global
    local_vs_global_local_gap = acc_local_on_local - acc_global_on_local
    quantization_error        = acc_local_fp32 - acc_quantized

    metrics = {
        "accuracy":                  float(acc_quantized),
        "acc_global":                float(acc_global),
        "acc_local_fp32":            float(acc_local_fp32),
        "acc_local_on_local":        float(acc_local_on_local),
        "acc_global_on_local":       float(acc_global_on_local),
        "local_vs_global_gap":       float(local_vs_global_gap),
        "local_vs_global_local_gap": float(local_vs_global_local_gap),
        "quantization_error":        float(quantization_error),
        "eval_time":                 float(eval_time),
        "quantized_model_size_mb":   float(get_model_size(net_quantized)),
        "loss":                      float(loss_quantized),
        "num-examples":              float(len(testloader_global.dataset)),
    }

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)

