import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
import torch
import torch.quantization
from collections import OrderedDict

from opacus import PrivacyEngine

from flwr.clientapp import ClientApp
from flwr.common import Context, Message, RecordDict, ArrayRecord, MetricRecord

from task import load_data, load_model, train_fedprox_dp, test, get_model_size, DEVICE

app = ClientApp()

# ── Per-client persistent state ───────────────────────────────────────────────
# Opacus PrivacyEngine must live across rounds to accumulate ε correctly.
# We store one engine per partition-id in a module-level dict so the
# stateless @app.train decorator can access it between calls.
_privacy_engines: dict = {}   # partition_id → (privacy_engine, net_dp, opt_dp, loader_dp)


def _get_or_init_privacy_engine(node_id, num_clients, batch_size,
                                 noise_multiplier, max_grad_norm):
    """Return (or lazily create) the persistent Opacus objects for a client."""
    if node_id not in _privacy_engines:
        net              = load_model()
        trainloader, _   = load_data(node_id=node_id, num_clients=num_clients,
                                     batch_size=batch_size)
        privacy_engine   = PrivacyEngine()
        optimizer        = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        net_dp, opt_dp, loader_dp = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        _privacy_engines[node_id] = (privacy_engine, net_dp, opt_dp, loader_dp)
    return _privacy_engines[node_id]


# ── Train ─────────────────────────────────────────────────────────────────────
@app.train()
def train(msg: Message, context: Context) -> Message:
    """FedProx + FedBN + DP-SGD local training step.

    Workflow
    --------
    1. Receive global weights (non-GN layers only) from the server.
    2. Inject them into the local model — GN layers remain local (FedBN).
    3. Train with FedProx + Opacus DP-SGD.
    4. Return only non-GN weights + KPI metrics.
    """
    start_time = time.time()
    tracemalloc.start()

    node_id        = context.node_config.get("partition-id", 0)
    num_clients    = context.node_config.get("num-partitions", 10)
    batch_size     = int(context.run_config.get("batch-size",       64))
    mu             = float(context.run_config.get("proximal_mu",    0.1))
    epochs         = int(context.run_config.get("local-epochs",     1))
    noise_mult     = float(context.run_config.get("noise-multiplier", 0.5))
    max_grad_norm  = float(context.run_config.get("max-grad-norm",  1.0))
    dp_delta       = float(context.run_config.get("dp-delta",       1e-5))

    # 1. Receive global weights — ArrayRecord → torch state dict
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # 2. Get (or lazily create) persistent Opacus objects for this partition
    privacy_engine, net_dp, opt_dp, loader_dp = _get_or_init_privacy_engine(
        node_id, num_clients, batch_size, noise_mult, max_grad_norm
    )

    # Inject server weights into non-GN layers only (FedBN: GN stays local)
    local_state = net_dp._module.state_dict()
    for k in local_state:
        if "gn" not in k and k in incoming_state_dict:
            local_state[k] = incoming_state_dict[k].to(DEVICE)
    net_dp._module.load_state_dict(local_state, strict=True)

    # 3. Snapshot global (non-GN) weights for the FedProx proximal term
    global_params_on_device = {
        k: v.clone().to(DEVICE)
        for k, v in net_dp._module.state_dict().items()
        if "gn" not in k
    }

    # 4. Train FedProx + DP
    _ = train_fedprox_dp(
        net_dp, opt_dp, loader_dp,
        global_params_on_device,
        epochs=epochs, mu=mu,
    )

    # 5. Cumulative ε (correct because PrivacyEngine is persistent)
    dp_epsilon = privacy_engine.get_epsilon(delta=dp_delta)

    # 6. Return only non-GN weights (server aggregates only these)
    state_to_return = {
        k: v.cpu() for k, v in net_dp._module.state_dict().items()
        if "gn" not in k
    }
    # ✅ Correct ArrayRecord constructor — use keyword argument
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
        "model_size_mb":    float(get_model_size(net_dp._module)),
        "dp_epsilon":       float(dp_epsilon),
        "estimated_energy": float(estimated_energy),
        "num-examples":     len(loader_dp.dataset),
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
    acc_local_fp32 : local model accuracy  (with local GN layers — Ditto/FedBN gain)
    quantization_error: accuracy drop from INT8 dynamic quantisation

    The two models differ because model_global has fresh GN weights (random init)
    while model_local uses the locally-trained GN weights from the PrivacyEngine.
    """
    start_time = time.time()

    node_id     = context.node_config.get("partition-id", 0)
    num_clients = context.node_config.get("num-partitions", 10)
    batch_size  = int(context.run_config.get("batch-size", 64))

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
    if node_id in _privacy_engines:
        _, net_dp, _, _ = _privacy_engines[node_id]
        model_local = net_dp._module
        local_state = model_local.state_dict()
        for k in local_state:
            if "gn" not in k and k in incoming_state_dict:
                local_state[k] = incoming_state_dict[k].to(DEVICE)
        model_local.load_state_dict(local_state, strict=True)
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
        "num-examples":           len(testloader.dataset),
    }

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
