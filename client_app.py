import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
from collections import OrderedDict

import torch
import torch.quantization
import numpy as np

# ── Flower v2 Message API imports ─────────────────────────────────────────────
from flwr.clientapp import ClientApp
from flwr.app import Context, Message, RecordDict, ArrayRecord, MetricRecord

from task import (
    load_data, load_model, train_ditto,
    test, get_model_size,
    DEVICE,
)


# ── Flower ClientApp ──────────────────────────────────────────────────────────
app = ClientApp()

# ── Per-client persistent state (context.state) ────────────────────────────────
# Local personalized models must live across rounds.
# We store them in context.state as "local_model_weights" (an ArrayRecord) to
# persist across different stateless Ray worker processes/actors.


# ── Train handler ─────────────────────────────────────────────────────────────
@app.train()
def train(msg: Message, context: Context) -> Message:
    """Ditto training step.

    Workflow
    --------
    1. Receive global weights from the server.
    2. Load client local personalization model from context.state (or init to global weights).
    3. Update global model with standard SGD, and local model with SGD + proximal penalty.
    4. Save local model weights to context.state.
    5. Return updated global weights to the server.
    """
    start_time = time.time()
    tracemalloc.start()

    # ── Node / run config ─────────────────────────────────────────────────────
    node_id     = context.node_config["partition-id"]
    num_clients = context.node_config.get("num-partitions", 10)
    rc          = context.run_config

    batch_size       = int(rc.get("batch-size",          32))
    lr_global        = float(rc.get("lr-global",          0.01))
    momentum_global  = float(rc.get("momentum",           0.9))
    alpha            = float(rc.get("alpha",              0.0))
    seed             = int(rc.get("partition-seed",       42))

    # ── Per-round config from server (msg.content["config"]) ─────────────────
    cfg           = msg.content["config"]
    local_epochs  = int(cfg.get("local_epochs",   1))
    mu            = float(cfg.get("proximal_mu",  0.01))
    lr_local      = float(cfg.get("lr_local",     0.001))
    momentum_local= float(cfg.get("momentum_local", 0.9))

    # ── 1. Receive global weights from server → inject into global_net ────────
    incoming = msg.content["arrays"].to_torch_state_dict()
    global_net = load_model()
    global_net.load_state_dict(
        {k: v.to(DEVICE) for k, v in incoming.items()}, strict=True
    )

    # ── 2. Ditto: Load local_net weights from context.state ──────────────────
    local_net = load_model()
    has_persisted_local = "local_model_weights" in context.state
    if has_persisted_local:
        local_params = context.state["local_model_weights"].to_torch_state_dict()
        local_net.load_state_dict(
            {k: v.to(DEVICE) for k, v in local_params.items()}, strict=True
        )
    else:
        # First round fallback: copy global weights
        local_net.load_state_dict(global_net.state_dict())

    # Log state loading diagnostic
    import os
    print(f"[FIT PID {os.getpid()}] Client {node_id}: loaded_persisted_local={has_persisted_local}")

    # ── 3. Load the data statelessly ──────────────────────────────────────────
    trainloader, _, _ = load_data(
        node_id=node_id, num_clients=num_clients,
        batch_size=batch_size, alpha=alpha, seed=seed,
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    global_optimizer = torch.optim.SGD(
        global_net.parameters(), lr=lr_global, momentum=momentum_global
    )
    global_net, local_net = train_ditto(
        global_net, global_optimizer, trainloader,
        local_net,
        epochs=local_epochs, mu=mu,
        lr_local=lr_local, momentum_local=momentum_local,
    )

    # ── 5. Save local model weights back to context.state ─────────────────────
    local_params = {k: v.cpu() for k, v in local_net.state_dict().items()}
    context.state["local_model_weights"] = ArrayRecord(torch_state_dict=local_params)

    # ── 6. Build reply message with updated global weights ────────────────────
    model_record = ArrayRecord(torch_state_dict={
        k: v.cpu() for k, v in global_net.state_dict().items()
    })

    params_to_return = [
        val.cpu().numpy()
        for _, val in global_net.state_dict().items()
    ]
    comm_size_mb     = sum(p.nbytes for p in params_to_return) / (1024 * 1024)
    _, peak_ram      = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    fit_time         = time.time() - start_time
    peak_ram_mb      = peak_ram / (1024 * 1024)
    estimated_energy = (2.0 * fit_time) + (15.0 * comm_size_mb)

    metrics = {
        "fit_time":         float(fit_time),
        "peak_ram_mb":      float(peak_ram_mb),
        "comm_size_mb":     float(comm_size_mb),
        "model_size_mb":    float(get_model_size(local_net)),
        "dp_epsilon":       0.0,
        "estimated_energy": float(estimated_energy),
        "num-examples":     float(len(trainloader.dataset)),
    }

    content = RecordDict({
        "arrays":  model_record,
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)


# ── Evaluate handler ──────────────────────────────────────────────────────────
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    start_time = time.time()

    node_id     = context.node_config["partition-id"]
    num_clients = context.node_config.get("num-partitions", 10)
    rc          = context.run_config
    batch_size  = int(rc.get("batch-size", 32))
    alpha       = float(rc.get("alpha", 0.0))
    seed        = int(rc.get("partition-seed", 42))

    incoming = msg.content["arrays"].to_torch_state_dict()

    # Load data statelessly
    _, testloader_global, testloader_local = load_data(
        node_id=node_id, num_clients=num_clients,
        batch_size=batch_size, alpha=alpha, seed=seed,
    )

    # ── EVAL 1 — Global model (aggregated weights) ───────────────────
    model_global = load_model()
    model_global.load_state_dict(
        {k: v.to(DEVICE) for k, v in incoming.items()}, strict=True
    )
    _, acc_global = test(model_global, testloader_global)

    # ── EVAL 2 — Local (personalised) Ditto model ───────────────────────────────────
    model_local = load_model()
    has_persisted_local = "local_model_weights" in context.state
    if has_persisted_local:
        local_params = context.state["local_model_weights"].to_torch_state_dict()
        model_local.load_state_dict(
            {k: v.to(DEVICE) for k, v in local_params.items()}, strict=True
        )
    else:
        model_local.load_state_dict(model_global.state_dict())

    # Log state loading diagnostic
    import os
    print(f"[EVAL PID {os.getpid()}] Client {node_id}: loaded_persisted_local={has_persisted_local}")

    _, acc_local_fp32        = test(model_local, testloader_global)  # global dist
    _, acc_local_on_local    = test(model_local, testloader_local)   # local dist
    _, acc_global_on_local   = test(model_global, testloader_local)  # baseline

    # ── EVAL 3 — INT8 dynamic quantisation (TinyML proxy) ────────────────────
    net_cpu = type(model_local)().cpu()
    net_cpu.load_state_dict({k: v.cpu() for k, v in model_local.state_dict().items()})
    net_quantized = torch.quantization.quantize_dynamic(
        net_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    loss_quantized, acc_quantized = test(
        net_quantized, testloader_global, device=torch.device("cpu")
    )

    eval_time           = time.time() - start_time
    local_vs_global_gap = acc_local_fp32 - acc_global
    quantization_error  = acc_local_fp32 - acc_quantized
    local_vs_global_local_gap = acc_local_on_local - acc_global_on_local

    metrics = {
        "accuracy":               float(acc_quantized),
        "acc_global":             float(acc_global),
        "acc_local_fp32":         float(acc_local_fp32),
        "acc_local_on_local":     float(acc_local_on_local),
        "acc_global_on_local":    float(acc_global_on_local),
        "local_vs_global_gap":    float(local_vs_global_gap),
        "local_vs_global_local_gap": float(local_vs_global_local_gap),
        "quantization_error":     float(quantization_error),
        "eval_time":              float(eval_time),
        "quantized_model_size_mb": float(get_model_size(net_quantized)),
        "loss":                   float(loss_quantized),
        "num-examples":           float(len(testloader_global.dataset)),
    }

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)

