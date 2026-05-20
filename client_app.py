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

# ── Persistent per-node Ditto state ──────────────────────────────────────────
_ditto_state: dict = {}

def _get_or_init_ditto_state(node_id, num_clients, batch_size,
                              lr_global, momentum_global):
    """Lazily create the persistent Ditto objects for a client node.
    Called only once per node.
    """
    if node_id not in _ditto_state:
        global_net  = load_model()
        local_net   = load_model()
        local_net.load_state_dict(global_net.state_dict())

        trainloader, testloader = load_data(
            node_id=node_id, num_clients=num_clients, batch_size=batch_size
        )

        optimizer = torch.optim.SGD(
            global_net.parameters(), lr=lr_global, momentum=momentum_global
        )

        _ditto_state[node_id] = {
            "global_net": global_net,
            "local_net":  local_net,
            "optimizer":  optimizer,
            "trainloader": trainloader,
            "testloader": testloader,
        }

    return _ditto_state[node_id]


# ── Train handler ─────────────────────────────────────────────────────────────
@app.train()
def train(msg: Message, context: Context) -> Message:
    start_time = time.time()
    tracemalloc.start()

    # ── Node / run config ─────────────────────────────────────────────────────
    node_id     = context.node_config["partition-id"]
    num_clients = context.node_config.get("num-partitions", 10)
    rc          = context.run_config

    batch_size       = int(rc.get("batch-size",          32))
    lr_global        = float(rc.get("lr-global",          0.01))
    momentum_global  = float(rc.get("momentum",           0.9))

    # ── Per-round config from server (msg.content["config"]) ─────────────────
    cfg           = msg.content["config"]
    local_epochs  = int(cfg.get("local_epochs",   1))
    mu            = float(cfg.get("proximal_mu",  0.01))
    lr_local      = float(cfg.get("lr_local",     0.001))
    momentum_local= float(cfg.get("momentum_local", 0.9))

    # ── Lazy init of persistent Ditto state ───────────────────────────────────
    state = _get_or_init_ditto_state(
        node_id, num_clients, batch_size, lr_global, momentum_global
    )

    # ── 1. Receive global weights from server → inject into global_net ────────
    incoming = msg.content["arrays"].to_torch_state_dict()
    state["global_net"].load_state_dict(
        {k: v.to(DEVICE) for k, v in incoming.items()}, strict=True
    )

    # ── 2. Re-sync local_net to current global weights (Ditto requirement) ────
    state["local_net"].load_state_dict(
        state["global_net"].state_dict()
    )

    # ── 3. Train ──────────────────────────────────────────────────────────────
    _, state["local_net"] = train_ditto(
        state["global_net"], state["optimizer"], state["trainloader"],
        state["local_net"],
        epochs=local_epochs, mu=mu,
        lr_local=lr_local, momentum_local=momentum_local,
    )

    # ── 4. Build reply message ────────────────────────────────────────────────
    model_record = ArrayRecord(torch_state_dict={
        k: v.cpu() for k, v in state["global_net"].state_dict().items()
    })

    params_to_return = [
        val.cpu().numpy()
        for _, val in state["global_net"].state_dict().items()
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
        "model_size_mb":    float(get_model_size(state["local_net"])),
        "dp_epsilon":       0.0,
        "estimated_energy": float(estimated_energy),
        "num-examples":     float(len(state["trainloader"].dataset)),
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

    incoming = msg.content["arrays"].to_torch_state_dict()

    # ── EVAL 1 — Global model (aggregated weights) ───────────────────
    model_global = load_model()
    model_global.load_state_dict(
        {k: v.to(DEVICE) for k, v in incoming.items()}, strict=True
    )

    if node_id in _ditto_state:
        testloader = _ditto_state[node_id]["testloader"]
    else:
        _, testloader = load_data(node_id=node_id, num_clients=num_clients,
                                  batch_size=batch_size)

    _, acc_global = test(model_global, testloader)

    # ── EVAL 2 — Local model ──────────────────────────────────────────────────
    if node_id in _ditto_state:
        model_local = _ditto_state[node_id]["local_net"]
        local_sd = model_local.state_dict()
        for k, v in incoming.items():
            local_sd[k] = v.to(DEVICE)
        model_local.load_state_dict(local_sd, strict=True)
    else:
        model_local = model_global

    _, acc_local_fp32 = test(model_local, testloader)

    # ── EVAL 3 — INT8 dynamic quantisation (TinyML proxy) ────────────────────
    net_cpu = type(model_local)().cpu()
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
