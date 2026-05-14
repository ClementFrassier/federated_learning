import csv
import os

import numpy as np

# NOTE: As of Flower 1.21, these imports are deprecated.
# The new paths are: flwr.serverapp, flwr.serverapp.strategy, flwr.app
# Migration to the Message API is tracked separately.
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedYogi

from task import load_model

# ── CSV logger ────────────────────────────────────────────────────────────────
CSV_DIR  = "resultsfeat"
CSV_FILE = os.path.join(CSV_DIR, "results_ditto-FedYogi-sdp-secAgg-sparse.csv")


class MetricLogger:
    """Stateful round counter — ensures CSV rows carry the correct round number.

    The CSV is reset at the start of every run to avoid duplicate rows.
    """
    def __init__(self):
        self.fit_round  = 1
        self.eval_round = 1
        os.makedirs(CSV_DIR, exist_ok=True)
        if os.path.isfile(CSV_FILE):
            os.remove(CSV_FILE)


logger = MetricLogger()


def write_to_csv(round_num: int, phase: str, metrics: dict) -> None:
    """Append one row per metric to the KPI CSV file."""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Round", "Phase", "Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([round_num, phase, k, v])


# ── Metric aggregation ────────────────────────────────────────────────────────
def fit_metrics_aggregation_fn(metrics):
    """Average all FIT metrics across clients and log to CSV."""
    aggregated: dict = {}
    if not metrics:
        return aggregated

    for metric_name in metrics[0][1].keys():
        aggregated[metric_name] = (
            sum(m[metric_name] for _, m in metrics) / len(metrics)
        )

    print(f"\n--- FIT KPIs (Round {logger.fit_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")

    write_to_csv(logger.fit_round, "FIT", aggregated)
    logger.fit_round += 1
    return aggregated


def evaluate_metrics_aggregation_fn(metrics):
    """Aggregate EVAL metrics with weighted accuracy + stddev, log to CSV."""
    aggregated: dict = {}
    if not metrics:
        return aggregated

    # Weighted accuracy
    total_examples     = sum(n for n, _ in metrics)
    aggregated["accuracy"] = (
        sum(n * m["accuracy"] for n, m in metrics) / total_examples
    )

    # Cross-client accuracy std-dev (fairness indicator)
    acc_list = [m["accuracy"] for _, m in metrics]
    mean_acc = sum(acc_list) / len(acc_list)
    variance = sum((a - mean_acc) ** 2 for a in acc_list) / len(acc_list)
    aggregated["accuracy_stddev"] = variance ** 0.5

    # Simple mean for every other metric
    for metric_name in metrics[0][1].keys():
        if metric_name != "accuracy":
            aggregated[metric_name] = (
                sum(m[metric_name] for _, m in metrics) / len(metrics)
            )

    print(f"\n--- EVAL KPIs (Round {logger.eval_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")

    write_to_csv(logger.eval_round, "EVAL", aggregated)
    logger.eval_round += 1
    return aggregated


# ── ServerApp entry point ─────────────────────────────────────────────────────
def server_fn(context: Context) -> ServerAppComponents:
    """Build the FedYogi strategy and server config from pyproject.toml values."""
    rc = context.run_config

    num_rounds     = int(rc.get("num-server-rounds",  30))
    proximal_mu    = float(rc.get("proximal-mu",       0.01))
    local_epochs   = int(rc.get("local-epochs",        1))
    lr_local       = float(rc.get("lr-local",          0.001))
    momentum_local = float(rc.get("momentum",          0.9))
    dp_delta       = float(rc.get("dp-delta",          1e-5))
    sparsity_ratio = float(rc.get("sparsity-ratio",    0.5))
    # FedYogi-specific adaptive learning rate parameters
    eta            = float(rc.get("fedyogi-eta",       0.01))
    eta_l          = float(rc.get("fedyogi-eta-l",     0.01))
    beta_1         = float(rc.get("fedyogi-beta-1",    0.9))
    beta_2         = float(rc.get("fedyogi-beta-2",    0.99))
    tau            = float(rc.get("fedyogi-tau",       1e-3))

    config = ServerConfig(num_rounds=num_rounds)

    # Per-round config dict injected into every client's fit() call
    def fit_config(server_round: int) -> dict:
        return {
            "proximal_mu":    proximal_mu,
            "local_epochs":   local_epochs,
            "lr_local":       lr_local,
            "momentum_local": momentum_local,
            "dp_delta":       dp_delta,
            "sparsity_ratio": sparsity_ratio,
        }

    # Initialise global model parameters (required by FedYogi)
    initial_model  = load_model()
    initial_ndarrays = [
        val.cpu().numpy() for _, val in initial_model.state_dict().items()
    ]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    # FedYogi server-side adaptive optimiser
    # SecAgg is simulated: in production, strategy would be wrapped with
    # SecAggPlusWorkflow or equivalent before being passed to ServerAppComponents.
    strategy = FedYogi(
        initial_parameters=initial_parameters,
        eta=eta,
        eta_l=eta_l,
        beta_1=beta_1,
        beta_2=beta_2,
        tau=tau,
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
