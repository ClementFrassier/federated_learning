import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
import torch
import torch.quantization
from opacus import PrivacyEngine

from flwr.clientapp import ClientApp
from flwr.app import Context, Message, RecordDict, ArrayRecord, MetricRecord

from task import load_data, load_model, train_fedprox_dp, test, get_model_size, DEVICE

app = ClientApp()

# ── Per-client persistent Opacus state ───────────────────────────────────────
# Dict layout:  { node_id: (PrivacyEngine, net_dp, opt_dp, loader_dp) }
_privacy_engines: dict = {}

# FIX: store locally trained weights separately for evaluate()
# Previously, evaluate() overwrote local weights with server weights,
# causing local_vs_global_gap to always be 0.0 (bug)
_local_weights: dict = {}  # { node_id: state_dict after local training }


def _get_or_init_privacy_engine(
    node_id: int,
    batch_size: int,
    lr: float,
    momentum: float,
    noise_multiplier: float,
    max_grad_norm: float,
    seed: int = 42,
):
    """
    Lazily create — and then cache — the Opacus objects for a given client.
    MLP Tiny has no BatchNorm → make_private() works natively without ExpandedWeights.
    """
    if node_id not in _privacy_engines:
        net            = load_model()
        trainloader, _ = load_data(node_id=node_id, batch_size=batch_size, seed=seed)
        privacy_engine = PrivacyEngine()
        optimizer      = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
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
    start_time = time.time()
    tracemalloc.start()

    node_id       = context.node_config.get("partition-id", 0)
    batch_size    = int(context.run_config.get("batch-size",        64))
    lr            = float(context.run_config.get("learning-rate",   0.01))
    momentum      = float(context.run_config.get("momentum",        0.9))
    mu            = float(context.run_config.get("proximal_mu",    0.05))
    epochs        = int(context.run_config.get("local-epochs",       2))
    noise_mult    = float(context.run_config.get("noise-multiplier", 0.8))  # FIX: 1.2 → 0.8
    max_grad_norm = float(context.run_config.get("max-grad-norm",    1.0))
    dp_delta      = float(context.run_config.get("dp-delta",        1e-5))
    seed          = int(context.run_config.get("partition-seed",     42))

    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    privacy_engine, net_dp, opt_dp, loader_dp = _get_or_init_privacy_engine(
        node_id, batch_size, lr, momentum, noise_mult, max_grad_norm, seed
    )

    local_state = net_dp._module.state_dict()
    for k in local_state:
        if k in incoming_state_dict:
            local_state[k] = incoming_state_dict[k].to(DEVICE)
    net_dp._module.load_state_dict(local_state, strict=True)

    global_params_on_device = {
        k: v.clone().to(DEVICE)
        for k, v in net_dp._module.state_dict().items()
    }

    train_fedprox_dp(
        net_dp, opt_dp, loader_dp,
        global_params_on_device,
        epochs=epochs,
        mu=mu,
    )

    dp_epsilon = privacy_engine.get_epsilon(delta=dp_delta)

    # FIX: save post-training local weights for evaluate()
    _local_weights[node_id] = {
        k: v.cpu().clone() for k, v in net_dp._module.state_dict().items()
    }

    state_to_return = {k: v.cpu() for k, v in net_dp._module.state_dict().items()}
    model_record    = ArrayRecord(torch_state_dict=state_to_return)

    comm_size_mb = (
        sum(p.element_size() * p.nelement() for p in state_to_return.values())
        / (1024 ** 2)
    )
    _, peak_ram     = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    fit_time         = time.time() - start_time
    peak_ram_mb      = peak_ram / (1024 ** 2)
    estimated_energy = 2.0 * fit_time + 15.0 * comm_size_mb

    metrics = {
        "fit_time":          float(fit_time),
        "peak_ram_mb":       float(peak_ram_mb),
        "comm_size_mb":      float(comm_size_mb),
        "model_size_mb":     float(get_model_size(net_dp._module)),
        "dp_epsilon":        float(dp_epsilon),
        "estimated_energy":  float(estimated_energy),
        "num-examples":      len(loader_dp.dataset),
    }

    content = RecordDict({
        "arrays":  model_record,
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)


# ── Evaluate ──────────────────────────────────────────────────────────────────
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    FIX applied here:
    BEFORE: model_local received server weights → gap was always 0.0
    AFTER: model_local uses _local_weights[node_id] (post-training weights)
           → gap accurately measures the value added by FedProx
    """
    start_time  = time.time()
    node_id     = context.node_config.get("partition-id", 0)
    batch_size  = int(context.run_config.get("batch-size",     64))
    seed        = int(context.run_config.get("partition-seed", 42))

    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # EVAL 1 — Global model (server weights, without local adaptation)
    model_global = load_model()
    g_state      = model_global.state_dict()
    for k in g_state:
        if k in incoming_state_dict:
            g_state[k] = incoming_state_dict[k].to(DEVICE)
    model_global.load_state_dict(g_state, strict=True)

    # EVAL 2 — Local model (post-training weights saved in _local_weights)
    # FIX: use _local_weights instead of overwriting with server weights
    model_local = load_model()
    if node_id in _local_weights:
        # Utilise les vrais poids locaux entraînés
        l_state = model_local.state_dict()
        for k in l_state:
            if k in _local_weights[node_id]:
                l_state[k] = _local_weights[node_id][k].to(DEVICE)
        model_local.load_state_dict(l_state, strict=True)
    else:
        # First evaluate before any training → fallback to global model
        model_local = model_global

    _, testloader = load_data(node_id=node_id, batch_size=batch_size, seed=seed)

    _, acc_global     = test(model_global, testloader)
    _, acc_local_fp32 = test(model_local,  testloader)

    # EVAL 3 — Dynamic INT8 quantization (TinyML proxy for MCU deployment)
    net_cpu = load_model().cpu()
    net_cpu.load_state_dict({k: v.cpu() for k, v in model_local.state_dict().items()})
    net_quantized = torch.quantization.quantize_dynamic(
        net_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    loss_q, acc_q = test(net_quantized, testloader, device=torch.device("cpu"))

    eval_time           = time.time() - start_time
    local_vs_global_gap = acc_local_fp32 - acc_global  # Maintenant non-nul
    quantization_error  = acc_local_fp32 - acc_q

    metrics = {
        "accuracy":                float(acc_q),
        "acc_global":              float(acc_global),
        "acc_local_fp32":          float(acc_local_fp32),
        "local_vs_global_gap":     float(local_vs_global_gap),  # FIX: real gap value
        "quantization_error":      float(quantization_error),
        "eval_time":               float(eval_time),
        "quantized_model_size_mb": float(get_model_size(net_quantized)),
        "loss":                    float(loss_q),
        "num-examples":            len(testloader.dataset),
    }

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
