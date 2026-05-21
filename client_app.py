import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
from collections import OrderedDict

import torch
import torch.quantization
import numpy as np
from opacus import PrivacyEngine

# ── Flower v2 Message API imports ─────────────────────────────────────────────
from flwr.clientapp import ClientApp
from flwr.app import Context, Message, RecordDict, ArrayRecord, MetricRecord

from task import (
    load_data, load_model, train_ditto_dp,
    apply_sparsification, test, get_model_size,
    DEVICE,
)


# ── Flower ClientApp ──────────────────────────────────────────────────────────
app = ClientApp()

# ── Persistent per-node Ditto state ──────────────────────────────────────────
_ditto_state: dict = {}

def _get_or_init_ditto_state(node_id, num_clients, batch_size,
                              noise_multiplier, max_grad_norm,
                              lr_global, momentum_global,
                              alpha=0.3, seed=42):
    """Lazily create the persistent Ditto + DP objects for a client node.
    Called only once per node. Subsequent rounds reuse the same objects
    so the PrivacyEngine accumulates ε across the entire simulation.
    """
    if node_id not in _ditto_state:
        global_net  = load_model()
        local_net   = load_model()
        local_net.load_state_dict(global_net.state_dict())

        trainloader, testloader = load_data(
            node_id=node_id, num_clients=num_clients, batch_size=batch_size,
            alpha=alpha, seed=seed,
        )

        pe        = PrivacyEngine()
        optimizer = torch.optim.SGD(
            global_net.parameters(), lr=lr_global, momentum=momentum_global
        )
        global_net_dp, opt_dp, loader_dp = pe.make_private(
            module=global_net,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        _ditto_state[node_id] = {
            "global_net": global_net_dp,
            "local_net":  local_net,
            "opt_dp":     opt_dp,
            "loader_dp":  loader_dp,
            "testloader": testloader,
            "pe":         pe,
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
    noise_multiplier = float(rc.get("noise-multiplier",   1.8))
    max_grad_norm    = float(rc.get("max-grad-norm",      1.0))
    alpha            = float(rc.get("alpha",              0.3))
    seed             = int(rc.get("partition-seed",       42))

    # ── Per-round config from server (msg.content["config"]) ─────────────────
    cfg           = msg.content["config"]
    local_epochs  = int(cfg.get("local_epochs",   1))
    mu            = float(cfg.get("proximal_mu",  0.01))
    lr_local      = float(cfg.get("lr_local",     0.001))
    momentum_local= float(cfg.get("momentum_local", 0.9))
    dp_delta      = float(cfg.get("dp_delta",     1e-5))
    sparsity_ratio= float(cfg.get("sparsity_ratio", 0.5))

    # ── Lazy init of persistent Ditto state ───────────────────────────────────
    state = _get_or_init_ditto_state(
        node_id, num_clients, batch_size,
        noise_multiplier, max_grad_norm, lr_global, momentum_global,
        alpha=alpha, seed=seed,
    )

    # ── 1. Receive global weights from server → inject into global_net ────────
    incoming = msg.content["arrays"].to_torch_state_dict()
    state["global_net"]._module.load_state_dict(
        {k: v.to(DEVICE) for k, v in incoming.items()}, strict=True
    )

    # ── 2. Ditto local model: weights are kept local and persistent across rounds,
    # and only regularized towards the global weights via proximal mu.

    # ── 3. Train (persistent Opacus objects passed in — ε accumulates) ────────
    _, state["local_net"] = train_ditto_dp(
        state["global_net"], state["opt_dp"], state["loader_dp"],
        state["local_net"],
        epochs=local_epochs, mu=mu,
        lr_local=lr_local, momentum_local=momentum_local,
    )

    dp_epsilon = state["pe"].get_epsilon(delta=dp_delta)

    # ── 4. Sparsify before sending (magnitude pruning) ────────────────────────
    # Extract weights as NumPy arrays for the apply_sparsification function
    params_to_return = [
        val.cpu().numpy()
        for _, val in state["global_net"]._module.state_dict().items()
    ]
    sparse_params_to_return = apply_sparsification(
        params_to_return, sparsity_ratio=sparsity_ratio
    )
    
    # Re-package as state_dict for ArrayRecord
    state_keys = list(state["global_net"]._module.state_dict().keys())
    sparse_state_dict = OrderedDict(
        (k, torch.tensor(v)) for k, v in zip(state_keys, sparse_params_to_return)
    )

    # ── 5. Build reply message ────────────────────────────────────────────────
    model_record = ArrayRecord(torch_state_dict=sparse_state_dict)

    # Effective sparse comm size: count only non-zero elements × dtype size.
    # p.nbytes counts all bytes including pruned zeros (dense), so it always
    # equals the full model size regardless of sparsity_ratio.
    comm_size_mb     = sum(
        int(np.count_nonzero(p)) * p.itemsize
        for p in sparse_params_to_return
    ) / (1024 * 1024)
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
        "dp_epsilon":       float(dp_epsilon),
        "estimated_energy": float(estimated_energy),
        "num-examples":     float(len(state["loader_dp"].dataset)),
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
    alpha       = float(rc.get("alpha", 0.3))
    seed        = int(rc.get("partition-seed", 42))

    # Receive server weights
    incoming = msg.content["arrays"].to_torch_state_dict()

    # ── EVAL 1 — Global model (FedYogi-aggregated weights) ───────────────────
    model_global = load_model()
    model_global.load_state_dict(
        {k: v.to(DEVICE) for k, v in incoming.items()}, strict=True
    )

    if node_id in _ditto_state:
        testloader = _ditto_state[node_id]["testloader"]
    else:
        _, testloader = load_data(node_id=node_id, num_clients=num_clients,
                                  batch_size=batch_size, alpha=alpha, seed=seed)

    _, acc_global = test(model_global, testloader)

    # ── EVAL 2 — Local model (locally-trained GN → Ditto personalisation) ─────
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
