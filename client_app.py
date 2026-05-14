import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
from collections import OrderedDict

import numpy as np
import torch
import torch.quantization

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from task import load_data, load_model, train_ditto_dp, apply_sparsification, test, get_model_size


# ── Flower Client ─────────────────────────────────────────────────────────────
class FlowerClient(NumPyClient):
    """Ditto + Quantisation + Local-DP + Sparse client."""

    def __init__(self, node_id: int, num_clients: int, batch_size: int):
        # Ditto: two models — global (server-synced) and local (personalised)
        self.global_net = load_model()
        self.local_net  = load_model()
        self.local_net.load_state_dict(self.global_net.state_dict())
        # Data partitioned by node_id; batch_size comes from pyproject.toml
        self.trainloader, self.testloader = load_data(
            node_id=node_id, num_clients=num_clients, batch_size=batch_size
        )

    # ── Parameter exchange ────────────────────────────────────────────────────
    def get_parameters(self, config):
        """Return all global model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.global_net.state_dict().items()]

    def set_parameters(self, parameters):
        """Overwrite global model weights from a list of NumPy arrays."""
        state_dict = self.global_net.state_dict()
        param_dict = zip(state_dict.keys(), parameters)
        update_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        state_dict.update(update_dict)
        self.global_net.load_state_dict(state_dict, strict=True)

    # ── Training round ────────────────────────────────────────────────────────
    def fit(self, parameters, config):
        """Local Ditto training step.

        Workflow:
        1. Receive global weights w_t from the server.
        2. Sync local_net to w_t (Ditto requirement).
        3. Run train_ditto_dp (global DP-SGD + local proximal SGD).
        4. Sparsify the global weights before sending (50 % magnitude pruning).
        5. Return sparse weights + KPI metrics.
        """
        start_time = time.time()
        tracemalloc.start()

        # 1. Pull global weights
        self.set_parameters(parameters)

        # 2. Sync local model to global at the start of each round (Ditto)
        self.local_net.load_state_dict(self.global_net.state_dict())

        # 3. Train: global (DP-SGD) + local (SGD + proximal)
        #    All hyperparams are injected by the server from pyproject.toml
        mu               = config.get("proximal_mu",      0.05)
        local_epochs     = config.get("local_epochs",     1)
        lr               = config.get("lr",               0.001)
        momentum         = config.get("momentum",         0.9)
        noise_multiplier = config.get("noise_multiplier", 1.5)
        max_grad_norm    = config.get("max_grad_norm",    1.0)
        dp_delta         = config.get("dp_delta",         1e-5)
        sparsity_ratio   = config.get("sparsity_ratio",   0.5)
        self.global_net, self.local_net, dp_epsilon = train_ditto_dp(
            self.global_net, self.local_net, self.trainloader,
            epochs=local_epochs, mu=mu, lr=lr, momentum=momentum,
            noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm,
            dp_delta=dp_delta,
        )

        # 4. Sparsify global weights before uplink
        params_to_return        = self.get_parameters(config)
        sparse_params_to_return = apply_sparsification(
            params_to_return, sparsity_ratio=sparsity_ratio
        )

        # KPI computation
        comm_size_mb = sum(p.nbytes for p in sparse_params_to_return) / (1024 * 1024)
        _, peak_ram  = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        fit_time     = time.time() - start_time
        peak_ram_mb  = peak_ram / (1024 * 1024)
        # Energy proxy: alpha_cpu = 2.0 W, beta_radio = 15.0 W/MB
        estimated_energy = (2.0 * fit_time) + (15.0 * comm_size_mb)

        metrics = {
            "fit_time":        float(fit_time),
            "peak_ram_mb":     float(peak_ram_mb),
            "comm_size_mb":    float(comm_size_mb),
            "model_size_mb":   float(get_model_size(self.local_net)),
            "dp_epsilon":      float(dp_epsilon),
            "estimated_energy": float(estimated_energy),
        }

        return sparse_params_to_return, len(self.trainloader.dataset), metrics

    # ── Evaluation round ──────────────────────────────────────────────────────
    def evaluate(self, parameters, config):
        """Three-level evaluation: global FP32, local FP32, local INT8 (TinyML).

        Metrics returned:
        - accuracy            : quantised model accuracy (production metric)
        - acc_global          : global model accuracy
        - acc_local_fp32      : personalised FP32 model accuracy
        - local_vs_global_gap : personalisation gain from Ditto
        - quantization_error  : accuracy drop from INT8 quantisation
        - eval_time           : wall-clock time
        - quantized_model_size_mb : INT8 model size in MB
        - loss                : quantised model cross-entropy loss
        """
        start_time = time.time()

        # Update global model with latest server weights
        self.set_parameters(parameters)

        # EVAL 1 — Global model (what the server aggregated)
        _, acc_global = test(self.global_net, self.testloader)

        # EVAL 2 — Personalised local model (trained with Ditto proximal term)
        _, acc_local_fp32 = test(self.local_net, self.testloader)

        # EVAL 3 — INT8 quantised local model (TinyML deployment proxy)
        # Build a CPU copy to avoid moving self.local_net off the GPU
        net_cpu = type(self.local_net)()
        net_cpu.load_state_dict({k: v.cpu() for k, v in self.local_net.state_dict().items()})
        net_quantized = torch.quantization.quantize_dynamic(
            net_cpu, {torch.nn.Linear}, dtype=torch.qint8
        )
        loss_quantized, acc_quantized = test(
            net_quantized, self.testloader, device=torch.device("cpu")
        )

        eval_time = time.time() - start_time

        # Derived metrics
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
    node_id     = context.node_config["partition-id"]
    num_clients = context.node_config.get("num-partitions", 10)
    # batch-size is read here so the DataLoader is built once at init
    batch_size  = context.run_config.get("batch-size", 32)
    return FlowerClient(node_id=node_id, num_clients=num_clients, batch_size=batch_size).to_client()

app = ClientApp(client_fn=client_fn)
