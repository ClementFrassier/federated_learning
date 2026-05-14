import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
from collections import OrderedDict

import torch
import torch.quantization
import numpy as np
from opacus import PrivacyEngine

# NOTE: As of Flower 1.21, these imports are deprecated.
# The new paths are: flwr.clientapp, flwr.serverapp, flwr.app
# Migration to the Message API (flwr.app.Context, flwr.clientapp.ClientApp)
# is tracked separately as it requires a full architecture change.
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from task import (
    load_data, load_model, train_ditto_dp,
    apply_sparsification, test, get_model_size,
    DEVICE,
)


# ── Flower Client ─────────────────────────────────────────────────────────────
class FlowerClient(NumPyClient):
    """Ditto + FedYogi + Stepwise-DP + SecAgg (simulated) + Sparse client.

    Architecture summary
    --------------------
    - **Ditto personalisation**: two models — global (sent to server) and
      local (kept on device, regularised toward global weights each round).
    - **Stepwise-DP**: a single Opacus PrivacyEngine is created once and
      kept alive across rounds so that the cumulative epsilon (ε) grows
      correctly over the full experiment.
    - **Sparsification**: 50 % magnitude pruning applied to global weights
      before the uplink, cutting effective communication size.
    - **INT8 quantisation**: local model is dynamically quantised for the
      TinyML evaluation (deployment proxy only — inference metric).
    - **SecAgg** (simulated): in a real deployment the aggregation step
      would use secure aggregation; here we log the flag for traceability.
    """

    def __init__(self, node_id: int, num_clients: int, batch_size: int,
                 lr_global: float, momentum_global: float,
                 noise_multiplier: float, max_grad_norm: float):

        # Ditto: two models — global (DP-SGD) and local (personalised)
        self.global_net = load_model()
        self.local_net  = load_model()
        self.local_net.load_state_dict(self.global_net.state_dict())

        # Load this client's IID partition
        self.trainloader, self.testloader = load_data(
            node_id=node_id, num_clients=num_clients, batch_size=batch_size
        )

        # Persistent PrivacyEngine — epsilon accumulates across all rounds
        self.privacy_engine  = PrivacyEngine()
        global_optimizer     = torch.optim.SGD(
            self.global_net.parameters(), lr=lr_global, momentum=momentum_global
        )
        (self.global_net_dp,
         self.global_optimizer_dp,
         self.trainloader_dp) = self.privacy_engine.make_private(
            module=self.global_net,
            optimizer=global_optimizer,
            data_loader=self.trainloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        self.total_epsilon = 0.0

    # ── Parameter exchange ────────────────────────────────────────────────────
    def get_parameters(self, config):
        """Return global model parameters (from unwrapped DP module)."""
        return [
            val.cpu().numpy()
            for _, val in self.global_net_dp._module.state_dict().items()
        ]

    def set_parameters(self, parameters):
        """Inject server weights into the unwrapped global model."""
        state_dict = self.global_net_dp._module.state_dict()
        param_dict = zip(state_dict.keys(), parameters)
        update_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        state_dict.update(update_dict)
        self.global_net_dp._module.load_state_dict(state_dict, strict=True)

    # ── Training round ────────────────────────────────────────────────────────
    def fit(self, parameters, config):
        """Local Ditto + DP training step.

        Workflow
        --------
        1. Receive aggregated global weights w_t from the server.
        2. Sync local_net to w_t (Ditto requirement each round).
        3. Run train_ditto_dp (global DP-SGD + local proximal SGD).
        4. Sparsify global weights (magnitude pruning) before uplink.
        5. Return sparse weights + KPI metrics to the server.
        """
        start_time = time.time()
        tracemalloc.start()

        # 1. Pull global weights from server
        self.set_parameters(parameters)

        # 2. Re-sync local_net to the received global weights
        self.local_net.load_state_dict(self.global_net_dp._module.state_dict())

        # 3. Train — all hyperparams injected by server from pyproject.toml
        mu              = config.get("proximal_mu",     0.01)
        local_epochs    = config.get("local_epochs",    1)
        lr_local        = config.get("lr_local",        0.001)
        momentum_local  = config.get("momentum_local",  0.9)

        # train_ditto_dp returns (global_net_unwrapped, local_net).
        # We do NOT reassign self.global_net here — the Opacus wrapper
        # self.global_net_dp must remain the single source of truth.
        # The returned unwrapped net is the same object as self.global_net_dp._module.
        _, self.local_net = train_ditto_dp(
            self.global_net_dp, self.global_optimizer_dp, self.trainloader_dp,
            self.local_net,
            epochs=local_epochs, mu=mu,
            lr_local=lr_local, momentum_local=momentum_local,
        )

        # Cumulative epsilon (ε) over all rounds
        dp_delta           = config.get("dp_delta", 1e-5)
        self.total_epsilon = self.privacy_engine.get_epsilon(delta=dp_delta)

        # 4. Sparsify before uplink
        sparsity_ratio          = config.get("sparsity_ratio", 0.5)
        params_to_return        = self.get_parameters(config)
        sparse_params_to_return = apply_sparsification(
            params_to_return, sparsity_ratio=sparsity_ratio
        )

        # KPIs
        comm_size_mb = sum(p.nbytes for p in sparse_params_to_return) / (1024 * 1024)
        _, peak_ram  = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fit_time    = time.time() - start_time
        peak_ram_mb = peak_ram / (1024 * 1024)
        # Energy proxy:  α·CPU_time + β·comm_size  (W·s ≈ J)
        estimated_energy = (2.0 * fit_time) + (15.0 * comm_size_mb)

        metrics = {
            "fit_time":         float(fit_time),
            "peak_ram_mb":      float(peak_ram_mb),
            "comm_size_mb":     float(comm_size_mb),
            "model_size_mb":    float(get_model_size(self.local_net)),
            "dp_epsilon":       float(self.total_epsilon),
            "estimated_energy": float(estimated_energy),
        }

        return sparse_params_to_return, len(self.trainloader.dataset), metrics

    # ── Evaluation round ──────────────────────────────────────────────────────
    def evaluate(self, parameters, config):
        """Three-level evaluation: global FP32 / local FP32 / local INT8.

        Metrics
        -------
        accuracy              : INT8 quantised model (production target)
        acc_global            : global model (server aggregation baseline)
        acc_local_fp32        : personalised FP32 local model (Ditto gain)
        local_vs_global_gap   : personalisation gain (Ditto benefit)
        quantization_error    : accuracy loss from INT8 quantisation
        accuracy_stddev       : cross-client fairness indicator
        quantized_model_size_mb: INT8 model size (TinyML footprint)
        eval_time             : wall-clock evaluation time
        loss                  : INT8 model cross-entropy loss
        num-examples          : number of test samples (for weighted average)
        """
        start_time = time.time()

        # Update global model with latest aggregated server weights
        self.set_parameters(parameters)

        # EVAL 1 — Global model (FedYogi-aggregated weights, no personalisation)
        _, acc_global = test(self.global_net_dp._module, self.testloader)

        # EVAL 2 — Personalised local model (Ditto proximal trained)
        _, acc_local_fp32 = test(self.local_net, self.testloader)

        # EVAL 3 — INT8 dynamic quantisation (TinyML deployment proxy)
        # Build a clean CPU copy to avoid moving self.local_net off GPU
        net_cpu = type(self.local_net)()
        net_cpu.load_state_dict({k: v.cpu() for k, v in self.local_net.state_dict().items()})
        net_quantized = torch.quantization.quantize_dynamic(
            net_cpu, {torch.nn.Linear}, dtype=torch.qint8
        )
        loss_quantized, acc_quantized = test(
            net_quantized, self.testloader, device=torch.device("cpu")
        )

        eval_time = time.time() - start_time

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
            "num-examples":           len(self.testloader.dataset),
        }

        return float(loss_quantized), len(self.testloader.dataset), metrics


# ── Flower ClientApp entry point ──────────────────────────────────────────────
def client_fn(context: Context):
    # Partition info (set automatically by Flower from num-supernodes)
    node_id     = context.node_config["partition-id"]
    num_clients = context.node_config.get("num-partitions", 10)

    # All hyperparams come from pyproject.toml via context.run_config
    rc = context.run_config
    return FlowerClient(
        node_id=node_id,
        num_clients=num_clients,
        batch_size=int(rc.get("batch-size",         32)),
        lr_global=float(rc.get("lr-global",         0.01)),
        momentum_global=float(rc.get("momentum",    0.9)),
        noise_multiplier=float(rc.get("noise-multiplier", 1.8)),
        max_grad_norm=float(rc.get("max-grad-norm", 1.0)),
    ).to_client()


app = ClientApp(client_fn=client_fn)
