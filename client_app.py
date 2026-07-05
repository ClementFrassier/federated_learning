import warnings
warnings.filterwarnings("ignore")

import os
import random
import time
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.quantization
from opacus import PrivacyEngine

from flwr.clientapp import ClientApp
from flwr.app import Context, Message, RecordDict, ArrayRecord, MetricRecord

from task import (
    load_data, load_model, train_fedprox_dp, test, get_model_size, DEVICE,
    get_global_class_weights, NetFedRepScaffold, get_total_train_examples, get_client_ids,
)

app = ClientApp()

# ── Per-client persistent Opacus state ───────────────────────────────────────
# Dict layout:  { (node_id, model_type): (PrivacyEngine, net_dp, opt_dp, loader_dp) }
_privacy_engines: dict = {}

# FIX: store locally trained weights separately for evaluate()
# Previously, evaluate() overwrote local weights with server weights,
# causing local_vs_global_gap to always be 0.0 (bug)
_local_weights: dict = {}  # { node_id: state_dict after local training }

# ── FedRep+SCAFFOLD state (personalization-mode=fedrep_scaffold) ─────────────
# Dict layout: { node_id: {privacy_engine, model, extractor_dp, opt_extractor_dp,
#                          opt_head, loader_dp, c_i} }
_fedrep_scaffold_state: dict = {}
# { node_id: full local state dict (extractor.*+head.* keys) after training },
# persisted for evaluate() the same way _local_weights is for FedProx/FedPer.
_fedrep_scaffold_local_full: dict = {}


def _underlying_module(m):
    """Return the state-dict-bearing module, whether `m` is an Opacus
    GradSampleModule (exposes `._module`) or a plain nn.Module (doesn't)."""
    return getattr(m, "_module", m)


def _get_or_init_privacy_engine(
    node_id: int,
    batch_size: int,
    lr: float,
    momentum: float,
    noise_multiplier: float,
    max_grad_norm: float,
    seed: int = 42,
    model_type: str = "tiny",
    dp_enabled: bool = True,
):
    """
    Lazily create — and then cache — the training objects for a given client.
    Index by (node_id, model_type) to avoid conflict if Ray workers are recycled.

    When dp_enabled is False, skips Opacus entirely (plain SGD, no clipping, no
    noise) — used by the FL-without-DP ablation to isolate the cost of DP from
    the cost of federation itself. The cache stores `None` in place of the
    PrivacyEngine to signal this.
    """
    cache_key = (node_id, model_type)
    if cache_key not in _privacy_engines:
        # Seed once per client, at first use only — gives a reproducible model
        # init and data order, while letting the RNG evolve naturally across
        # rounds afterward (fresh DP noise / batch shuffling each round).
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        net            = load_model(model_type)
        trainloader, _ = load_data(node_id=node_id, batch_size=batch_size, seed=seed)
        optimizer      = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        if dp_enabled:
            privacy_engine = PrivacyEngine()
            net_dp, opt_dp, loader_dp = privacy_engine.make_private(
                module=net,
                optimizer=optimizer,
                data_loader=trainloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
            )
            _privacy_engines[cache_key] = (privacy_engine, net_dp, opt_dp, loader_dp)
        else:
            _privacy_engines[cache_key] = (None, net, optimizer, trainloader)
    return _privacy_engines[cache_key]


# ── FedRep+SCAFFOLD: tête multi-couches + DP restreint à l'extracteur ────────
# Architecture "meilleure possible hors contrainte TinyML, contraintes de
# sécurité seules" (voir rapport). Chemin d'exécution entièrement séparé du
# FedProx/FedPer ci-dessus, pour ne rien risquer sur les résultats déjà validés.
CV_PREFIX = "cv__"


def _get_or_init_fedrep_scaffold(node_id, batch_size, lr, momentum, noise_multiplier, max_grad_norm, seed):
    if node_id not in _fedrep_scaffold_state:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        model          = NetFedRepScaffold().to(DEVICE)
        trainloader, _ = load_data(node_id=node_id, batch_size=batch_size, seed=seed)
        opt_extractor  = torch.optim.SGD(model.extractor.parameters(), lr=lr, momentum=momentum)
        opt_head       = torch.optim.SGD(model.head.parameters(), lr=lr, momentum=momentum)

        privacy_engine = PrivacyEngine()
        extractor_dp, opt_extractor_dp, loader_dp = privacy_engine.make_private(
            module=model.extractor,
            optimizer=opt_extractor,
            data_loader=trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        # SCAFFOLD client control variate, one tensor per extractor parameter,
        # initialised to zero (standard SCAFFOLD init).
        c_i = {
            name: torch.zeros_like(p, device=DEVICE)
            for name, p in _underlying_module(extractor_dp).named_parameters()
        }

        _fedrep_scaffold_state[node_id] = dict(
            privacy_engine=privacy_engine,
            model=model,
            extractor_dp=extractor_dp,
            opt_extractor_dp=opt_extractor_dp,
            opt_head=opt_head,
            loader_dp=loader_dp,
            c_i=c_i,
        )
    return _fedrep_scaffold_state[node_id]


def _train_fedrep_scaffold(msg: Message, context: Context, node_id: int, start_time: float) -> Message:
    tracemalloc.start()
    batch_size       = int(context.run_config.get("batch-size",        64))
    lr               = float(context.run_config.get("learning-rate",   0.01))
    momentum         = float(context.run_config.get("momentum",        0.9))
    epochs           = int(context.run_config.get("local-epochs",       2))
    noise_mult       = float(context.run_config.get("noise-multiplier", 0.8))
    max_grad_norm    = float(context.run_config.get("max-grad-norm",    1.0))
    dp_delta         = float(context.run_config.get("dp-delta",        1e-5))
    seed             = int(context.run_config.get("partition-seed",     42))
    excluded_node_id = int(context.run_config.get("excluded-node-id",   -1))

    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()
    incoming_extractor   = {k: v for k, v in incoming_state_dict.items() if not k.startswith(CV_PREFIX)}
    incoming_cglobal_raw = {k[len(CV_PREFIX):]: v for k, v in incoming_state_dict.items() if k.startswith(CV_PREFIX)}

    if node_id == excluded_node_id:
        # Same principle as FedPer's fix: echo back a key set consistent with
        # trained clients (extractor + cv__ keys, never head.*) — otherwise
        # this round's replies would have inconsistent ArrayRecord keys.
        tracemalloc.stop()
        metrics = {
            "fit_time": float(time.time() - start_time), "peak_ram_mb": 0.0,
            "peak_ram_train_mb": 0.0, "peak_ram_epsilon_mb": 0.0, "comm_size_mb": 0.0,
            "model_size_mb": 0.0, "dp_epsilon": 0.0, "estimated_energy": 0.0,
            "num-examples": 0,
        }
        content = RecordDict({"arrays": msg.content["arrays"], "metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    state = _get_or_init_fedrep_scaffold(node_id, batch_size, lr, momentum, noise_mult, max_grad_norm, seed)
    privacy_engine, model, extractor_dp, opt_extractor_dp, opt_head, loader_dp, c_i = (
        state["privacy_engine"], state["model"], state["extractor_dp"], state["opt_extractor_dp"],
        state["opt_head"], state["loader_dp"], state["c_i"],
    )

    local_state = _underlying_module(extractor_dp).state_dict()
    for k in local_state:
        if k in incoming_extractor:
            local_state[k] = incoming_extractor[k].to(DEVICE)
    _underlying_module(extractor_dp).load_state_dict(local_state, strict=True)

    c_global = (
        {k: v.to(DEVICE) for k, v in incoming_cglobal_raw.items()}
        if incoming_cglobal_raw else {k: torch.zeros_like(v) for k, v in c_i.items()}
    )

    extractor_before = {k: v.clone() for k, v in _underlying_module(extractor_dp).state_dict().items()}

    criterion = nn.CrossEntropyLoss()
    extractor_dp.train()
    model.head.train()
    n_local_steps = 0
    any_pre_step_false = False
    for _ in range(epochs):
        for X_batch, y_batch in loader_dp:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            opt_extractor_dp.zero_grad()
            opt_head.zero_grad()

            logits = model.head(extractor_dp(X_batch))
            loss = criterion(logits, y_batch)
            loss.backward()

            # Opacus 1.6.0: DPOptimizer.step() == pre_step() [clip+noise+scale,
            # populates .grad] then original_optimizer.step(). Calling pre_step()
            # ourselves lets us splice the SCAFFOLD correction AFTER DP noise,
            # BEFORE the actual parameter update (verified against installed
            # Opacus source, see plan).
            if opt_extractor_dp.pre_step():
                # FIX (found during the grid run, not the 3-round smoke test):
                # the raw gradient is bounded by DP-SGD clipping (max_grad_norm),
                # but the SCAFFOLD correction (c_global - c_i) is not — c_i
                # accumulates round over round and the unbounded correction
                # quickly dominates the DP-protected (deliberately tiny) signal,
                # a positive-feedback loop that diverges to NaN within a few
                # rounds. Clip the correction's global L2 norm to max_grad_norm
                # too, same bound as Opacus applies to the real gradient, before
                # adding it — keeps both terms on a comparable scale.
                correction = {
                    name: (c_global[name] - c_i[name])
                    for name, _ in _underlying_module(extractor_dp).named_parameters()
                    if name in c_global and name in c_i
                }
                correction_norm = torch.sqrt(sum((v ** 2).sum() for v in correction.values()))
                if correction_norm > max_grad_norm:
                    clip_factor = max_grad_norm / (correction_norm + 1e-6)
                    correction = {k: v * clip_factor for k, v in correction.items()}
                for name, param in _underlying_module(extractor_dp).named_parameters():
                    if name in correction:
                        param.grad += correction[name]
                opt_extractor_dp.original_optimizer.step()
            else:
                any_pre_step_false = True
            opt_head.step()
            n_local_steps += 1

    if any_pre_step_false:
        print(f"[SCAFFOLD WARNING] node_id={node_id}: pre_step() returned False at least "
              f"once this round — no virtual-batch accumulation is expected in this "
              f"pipeline, investigate before trusting this round's correction.")

    dp_epsilon = privacy_engine.get_epsilon(delta=dp_delta)

    # SCAFFOLD "option 2" client control-variate update.
    K = n_local_steps  # nombre reel de pas d'optimiseur locaux (batches x epochs)
    extractor_after = _underlying_module(extractor_dp).state_dict()
    new_c_i = {}
    for name in c_i:
        delta_w = (extractor_before[name] - extractor_after[name]) / (K * lr)
        new_c_i[name] = c_i[name] - c_global.get(name, torch.zeros_like(c_i[name])) + delta_w
    state["c_i"] = new_c_i

    c_i_norm = float(torch.sqrt(sum((v ** 2).sum() for v in new_c_i.values())))
    grad_norm = float(torch.sqrt(sum(
        (p.grad ** 2).sum() for p in _underlying_module(extractor_dp).parameters() if p.grad is not None
    )))
    print(f"[SCAFFOLD DEBUG] node_id={node_id}: ||c_i||={c_i_norm:.6f}  "
          f"||grad_extracteur(dernier batch)||={grad_norm:.6f}  K={K}")

    # Persist full local state (extractor.*+head.* keys, matching
    # NetFedRepScaffold's own state_dict() naming) to disk and in memory for
    # evaluate() — mirrors FedPer's _local_weights/disk-save pattern.
    full_local_state = {f"extractor.{k}": v.cpu().clone() for k, v in extractor_after.items()}
    full_local_state.update({f"head.{k}": v.cpu().clone() for k, v in model.head.state_dict().items()})
    _fedrep_scaffold_local_full[node_id] = full_local_state

    base_dir = "/home/clement/projet/flwr/pytorch_flwr/resultsfeat"
    out_dir = os.path.join(base_dir, "ablation_fedrep_scaffold_heads")
    os.makedirs(out_dir, exist_ok=True)
    # BUG FIX: sigma must be part of the filename — the main grid reuses the
    # same 5 seeds across 5 sigma values, and without sigma in the name each
    # new sigma silently overwrote the previous one's head files on disk
    # (found only at the end of the first full grid run, after 4/5 sigma
    # values' heads had already been overwritten).
    torch.save(full_local_state, os.path.join(out_dir, f"seed{seed}_sigma{noise_mult:.1f}_client{node_id}.pt"))

    # SCAFFOLD c_global rescaling: aggregate_arrayrecords weighs by num-examples
    # (correct for model weights, FedAvg-standard) but SCAFFOLD's canonical
    # c_global update is a SIMPLE (unweighted) average across clients. Rescale
    # c_i before transmission so the native weighted average reconstructs the
    # simple average: v_i = c_i_new * n_total / (N * n_i).
    n_i = len(loader_dp.dataset)
    n_total = get_total_train_examples()
    n_clients = len(get_client_ids())
    scale = n_total / (n_clients * n_i)

    state_to_return = {k: v.cpu() for k, v in extractor_after.items()}
    for name, v in new_c_i.items():
        state_to_return[f"{CV_PREFIX}{name}"] = (v.cpu() * scale)
    model_record = ArrayRecord(torch_state_dict=state_to_return)

    comm_size_mb = sum(p.element_size() * p.nelement() for p in state_to_return.values()) / (1024 ** 2)

    tracemalloc.stop()
    fit_time = time.time() - start_time
    metrics = {
        "fit_time":            float(fit_time),
        "peak_ram_mb":         0.0,
        "peak_ram_train_mb":   0.0,
        "peak_ram_epsilon_mb": 0.0,
        "comm_size_mb":        float(comm_size_mb),
        "model_size_mb":       float(get_model_size(model)),
        "dp_epsilon":          float(dp_epsilon),
        "estimated_energy":    float(2.0 * fit_time + 15.0 * comm_size_mb),
        "num-examples":        n_i,
    }
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


def _evaluate_fedrep_scaffold(msg: Message, context: Context, node_id: int, start_time: float) -> Message:
    batch_size = int(context.run_config.get("batch-size",     64))
    seed       = int(context.run_config.get("partition-seed", 42))

    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()
    incoming_extractor = {k: v for k, v in incoming_state_dict.items() if not k.startswith(CV_PREFIX)}

    # "Global": shared extractor from server + a FRESH (never transmitted) head —
    # same limitation already documented for FedPer's acc_global: not a
    # meaningful deployable metric on its own, kept only for continuity with the
    # other modes' CSV schema.
    model_global = NetFedRepScaffold().to(DEVICE)
    g_state = model_global.extractor.state_dict()
    for k in g_state:
        if k in incoming_extractor:
            g_state[k] = incoming_extractor[k].to(DEVICE)
    model_global.extractor.load_state_dict(g_state, strict=True)

    model_local = NetFedRepScaffold().to(DEVICE)
    if node_id in _fedrep_scaffold_local_full:
        l_state = model_local.state_dict()
        for k in l_state:
            if k in _fedrep_scaffold_local_full[node_id]:
                l_state[k] = _fedrep_scaffold_local_full[node_id][k].to(DEVICE)
        model_local.load_state_dict(l_state, strict=True)
    else:
        model_local = model_global  # first eval before any training

    _, testloader = load_data(node_id=node_id, batch_size=batch_size, seed=seed)
    _, acc_global     = test(model_global, testloader)
    _, acc_local_fp32 = test(model_local,  testloader)

    net_cpu = NetFedRepScaffold().cpu()
    net_cpu.load_state_dict({k: v.cpu() for k, v in model_local.state_dict().items()})
    net_quantized = torch.quantization.quantize_dynamic(net_cpu, {torch.nn.Linear}, dtype=torch.qint8)
    loss_q, acc_q = test(net_quantized, testloader, device=torch.device("cpu"))

    eval_time = time.time() - start_time
    metrics = {
        "accuracy":                float(acc_q),
        "acc_global":              float(acc_global),
        "acc_local_fp32":          float(acc_local_fp32),
        "local_vs_global_gap":     float(acc_local_fp32 - acc_global),
        "quantization_error":      float(acc_local_fp32 - acc_q),
        "eval_time":               float(eval_time),
        "quantized_model_size_mb": float(get_model_size(net_quantized)),
        "loss":                    float(loss_q),
        "num-examples":            len(testloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


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
    model_type    = context.run_config.get("model-type", "tiny")
    dp_enabled    = str(context.run_config.get("dp-enabled", True)).lower() != "false"
    excluded_node_id = int(context.run_config.get("excluded-node-id", -1))
    loss_type     = context.run_config.get("loss-type", "ce")
    personalization_mode = context.run_config.get("personalization-mode", "none")

    if personalization_mode == "fedrep_scaffold":
        # Chemin d'exécution entièrement séparé — le code FedProx/FedPer
        # ci-dessous n'est jamais atteint pour ce mode. _train_fedrep_scaffold
        # gère son propre tracemalloc.start()/stop().
        return _train_fedrep_scaffold(msg, context, node_id, start_time)

    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    if node_id == excluded_node_id:
        # Late-joiner ablation: this client contributes nothing to training —
        # it never trains locally, so it never influences the aggregated global
        # model. Returning num-examples=0 gives it zero weight in FedProx's
        # weighted-average aggregation. Its weights are passed through unchanged.
        tracemalloc.stop()
        metrics = {
            "fit_time":            float(time.time() - start_time),
            "peak_ram_mb":         0.0,
            "peak_ram_train_mb":   0.0,
            "peak_ram_epsilon_mb": 0.0,
            "comm_size_mb":        0.0,
            "model_size_mb":       0.0,
            "dp_epsilon":          0.0,
            "estimated_energy":    0.0,
            "num-examples":        0,
        }
        echoed_arrays = msg.content["arrays"]
        if personalization_mode == "fedper":
            # FedPer: must echo back the same key set as trained clients (fc1/fc2
            # only) — otherwise this round's replies have inconsistent ArrayRecord
            # keys across clients (this client would still carry fc3 from the
            # round-1 broadcast) and aggregation raises InconsistentMessageReplies.
            echoed_sd = {k: v for k, v in echoed_arrays.to_torch_state_dict().items() if "fc3" not in k}
            echoed_arrays = ArrayRecord(torch_state_dict=echoed_sd)
        content = RecordDict({
            "arrays":  echoed_arrays,
            "metrics": MetricRecord(metrics),
        })
        return Message(content=content, reply_to=msg)

    privacy_engine, net_dp, opt_dp, loader_dp = _get_or_init_privacy_engine(
        node_id, batch_size, lr, momentum, noise_mult, max_grad_norm, seed, model_type,
        dp_enabled=dp_enabled,
    )

    local_state = _underlying_module(net_dp).state_dict()
    for k in local_state:
        if personalization_mode == "fedper" and "fc3" in k:
            # FedPer: the classification head is 100% local, never touched by the
            # server — including round 1, where the initial broadcast still
            # contains a fresh (server-side, uncontrolled) fc3 from server_app.py's
            # Net() init. Without this explicit skip, every client would adopt
            # that same random head at round 1 instead of keeping its own.
            continue
        if k in incoming_state_dict:
            local_state[k] = incoming_state_dict[k].to(DEVICE)
    _underlying_module(net_dp).load_state_dict(local_state, strict=True)

    global_params_on_device = {
        k: v.clone().to(DEVICE)
        for k, v in _underlying_module(net_dp).state_dict().items()
    }

    train_fedprox_dp(
        net_dp, opt_dp, loader_dp,
        global_params_on_device,
        epochs=epochs,
        mu=mu,
        class_weights=get_global_class_weights() if loss_type == "weighted_ce" else None,
    )

    # ── Tâche 3 (TRUE FIX): separate RAM before / after get_epsilon() ─────────
    # We call get_traced_memory() immediately after training, BEFORE get_epsilon(),
    # to capture pure local training RAM peak.
    _, peak_ram_train = tracemalloc.get_traced_memory()
    peak_ram_train_mb = peak_ram_train / (1024 ** 2)

    # Now calculate epsilon (triggering Opacus PRV accountant convolutions).
    # Skipped entirely when DP is disabled — there is no privacy budget to track.
    dp_epsilon = privacy_engine.get_epsilon(delta=dp_delta) if privacy_engine is not None else float("inf")

    # Capture the new absolute peak including get_epsilon
    _, peak_ram_after_eps = tracemalloc.get_traced_memory()
    peak_ram_epsilon_mb = peak_ram_after_eps / (1024 ** 2)

    # FIX: save post-training local weights for evaluate()
    # NOTE (FedPer): always the FULL state dict (fc1+fc2+fc3), independent of
    # what gets filtered out of state_to_return below — this dict comprehension
    # re-iterates net_dp.state_dict() on its own, it does not share references
    # with state_to_return's comprehension. Verified in smoke test 2a/6.
    _local_weights[node_id] = {
        k: v.cpu().clone() for k, v in _underlying_module(net_dp).state_dict().items()
    }

    if personalization_mode == "fedper":
        # Persist the final local head to disk — the only way to recover it
        # after the simulation process exits, since fc3 is never sent to the
        # server and _local_weights lives only in memory. Overwritten every
        # round, so the file left after the last round is the final head.
        base_dir = "/home/clement/projet/flwr/pytorch_flwr/resultsfeat"
        out_dir = os.path.join(base_dir, "ablation_fedper_heads")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            _local_weights[node_id],
            os.path.join(out_dir, f"seed{seed}_sigma{noise_mult:.1f}_client{node_id}.pt"),
        )

    state_to_return = {
        k: v.cpu() for k, v in _underlying_module(net_dp).state_dict().items()
        if not (personalization_mode == "fedper" and "fc3" in k)
    }
    model_record    = ArrayRecord(torch_state_dict=state_to_return)

    comm_size_mb = (
        sum(p.element_size() * p.nelement() for p in state_to_return.values())
        / (1024 ** 2)
    )

    tracemalloc.stop()
    fit_time         = time.time() - start_time
    estimated_energy = 2.0 * fit_time + 15.0 * comm_size_mb

    metrics = {
        "fit_time":            float(fit_time),
        "peak_ram_mb":         float(peak_ram_train_mb),    # backwards compat: training-only RAM
        "peak_ram_train_mb":   float(peak_ram_train_mb),    # stable ~20 Mo, σ-independent
        "peak_ram_epsilon_mb": float(peak_ram_epsilon_mb),  # PRV spike, σ-dependent (26–96 Mo)
        "comm_size_mb":        float(comm_size_mb),
        "model_size_mb":       float(get_model_size(_underlying_module(net_dp))),
        "dp_epsilon":          float(dp_epsilon),
        "estimated_energy":    float(estimated_energy),
        "num-examples":        len(loader_dp.dataset),
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
    model_type  = context.run_config.get("model-type", "tiny")
    personalization_mode = context.run_config.get("personalization-mode", "none")

    if personalization_mode == "fedrep_scaffold":
        return _evaluate_fedrep_scaffold(msg, context, node_id, start_time)

    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()

    # EVAL 1 — Global model (server weights, without local adaptation)
    model_global = load_model(model_type)
    g_state      = model_global.state_dict()
    for k in g_state:
        if k in incoming_state_dict:
            g_state[k] = incoming_state_dict[k].to(DEVICE)
    model_global.load_state_dict(g_state, strict=True)

    # EVAL 2 — Local model (post-training weights saved in _local_weights)
    # FIX: use _local_weights instead of overwriting with server weights
    model_local = load_model(model_type)
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
    net_cpu = load_model(model_type).cpu()
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
