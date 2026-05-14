import csv
import os

from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

# ── CSV logger ────────────────────────────────────────────────────────────────
CSV_FILE = "results_ditto-quant-ldp-sparse.csv"


class MetricLogger:
    """Stateful round counter so CSV rows carry the correct round number."""
    def __init__(self):
        self.fit_round  = 1
        self.eval_round = 1
        # Reset the CSV at the start of every run to avoid duplicate rows
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


# ── Metric aggregation functions ──────────────────────────────────────────────
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
    """Aggregate EVAL metrics, compute weighted accuracy and stddev, log to CSV."""
    aggregated: dict = {}
    if not metrics:
        return aggregated

    # Weighted average for accuracy
    total_examples = sum(n for n, _ in metrics)
    aggregated["accuracy"] = (
        sum(n * m["accuracy"] for n, m in metrics) / total_examples
    )

    # Accuracy standard deviation across clients (fairness indicator)
    acc_list  = [m["accuracy"] for _, m in metrics]
    mean_acc  = sum(acc_list) / len(acc_list)
    variance  = sum((a - mean_acc) ** 2 for a in acc_list) / len(acc_list)
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
    # Read all hyperparameters from pyproject.toml [tool.flwr.app.config]
    num_rounds       = int(context.run_config.get("num-server-rounds",  30))
    local_epochs     = int(context.run_config.get("local-epochs",        1))
    proximal_mu      = float(context.run_config.get("proximal-mu",       0.05))
    lr_local         = float(context.run_config.get("lr-local",          0.001))
    momentum_local   = float(context.run_config.get("momentum",          0.9))
    noise_multiplier = float(context.run_config.get("noise-multiplier",  1.5))
    max_grad_norm    = float(context.run_config.get("max-grad-norm",     1.0))
    dp_delta         = float(context.run_config.get("dp-delta",          1e-5))
    sparsity_ratio   = float(context.run_config.get("sparsity-ratio",    0.5))
    batch_size       = int(context.run_config.get("batch-size",          16))

    config = ServerConfig(num_rounds=num_rounds)

    # Bundle all client-side hyperparams into the per-round config dict.
    # Key names here must match exactly what client_app.py reads via config.get()
    def fit_config(server_round: int) -> dict:
        return {
            "local_epochs":    local_epochs,
            "proximal_mu":     proximal_mu,
            "lr_local":        lr_local,       # matches config.get("lr_local") in client
            "momentum_local":  momentum_local, # matches config.get("momentum_local")
            "dp_delta":        dp_delta,
            "sparsity_ratio":  sparsity_ratio,
        }

    # FedAvg aggregates the global model weights (Ditto server-side)
    strategy = FedAvg(
        on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
