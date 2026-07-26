import csv
import os

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from task import Net

# ── CSV logger ────────────────────────────────────────────────────────────────
CSV_DIR  = "resultsfeat"
CSV_FILE = os.path.join(CSV_DIR, "results_ditto-quant-ldp-sparse.csv")


class MetricLogger:
    """Stateful round counter so CSV rows carry the correct round number."""
    def __init__(self):
        self.fit_round  = 1
        self.eval_round = 1
        # Create the directory and reset the CSV at startup
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


# ── Metric aggregation functions ──────────────────────────────────────────────
def fit_metrics_aggregation_fn(record_dicts: list[RecordDict], weighted_by: str) -> MetricRecord:
    """Average all FIT metrics across clients and log to CSV."""
    aggregated: dict = {}
    if not record_dicts:
        return MetricRecord(aggregated)

    metrics_list = [rd.metric_records["metrics"] for rd in record_dicts]
    for metric_name in metrics_list[0].keys():
        aggregated[metric_name] = (
            sum(m[metric_name] for m in metrics_list) / len(metrics_list)
        )

    print(f"\n--- FIT KPIs (Round {logger.fit_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")

    write_to_csv(logger.fit_round, "FIT", aggregated)
    logger.fit_round += 1
    return MetricRecord(aggregated)


def evaluate_metrics_aggregation_fn(record_dicts: list[RecordDict], weighted_by: str) -> MetricRecord:
    """Aggregate EVAL metrics, compute weighted accuracy and stddev, log to CSV."""
    aggregated: dict = {}
    if not record_dicts:
        return MetricRecord(aggregated)

    metrics_list = [rd.metric_records["metrics"] for rd in record_dicts]

    # Weighted average for accuracy
    total_examples = sum(m["num-examples"] for m in metrics_list)
    aggregated["accuracy"] = (
        sum(m["num-examples"] * m["accuracy"] for m in metrics_list) / total_examples
    )

    # Accuracy standard deviation across clients (fairness indicator)
    acc_list  = [m["accuracy"] for m in metrics_list]
    mean_acc  = sum(acc_list) / len(acc_list)
    variance  = sum((a - mean_acc) ** 2 for a in acc_list) / len(acc_list)
    aggregated["accuracy_stddev"] = variance ** 0.5

    # Simple mean for every other metric
    for metric_name in metrics_list[0].keys():
        if metric_name not in ("accuracy", "num-examples"):
            aggregated[metric_name] = (
                sum(m[metric_name] for m in metrics_list) / len(metrics_list)
            )

    print(f"\n--- EVAL KPIs (Round {logger.eval_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")

    write_to_csv(logger.eval_round, "EVAL", aggregated)
    logger.eval_round += 1
    return MetricRecord(aggregated)


# ── ServerApp ─────────────────────────────────────────────────────────────────
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read all hyperparameters from pyproject.toml [tool.flwr.app.config]
    num_rounds       = int(context.run_config.get("num-server-rounds",  30))
    local_epochs     = int(context.run_config.get("local-epochs",        1))
    proximal_mu      = float(context.run_config.get("proximal-mu",       0.05))
    lr_local         = float(context.run_config.get("lr-local",          0.001))
    momentum_local   = float(context.run_config.get("momentum",          0.9))
    dp_delta         = float(context.run_config.get("dp-delta",          1e-5))
    sparsity_ratio   = float(context.run_config.get("sparsity-ratio",    0.5))

    # Initialise global model
    global_model = Net()
    arrays = ArrayRecord(torch_state_dict=global_model.state_dict())

    # Build the training config to send to clients
    def fit_config(server_round: int) -> ConfigRecord:
        return ConfigRecord({
            "local_epochs":    local_epochs,
            "proximal_mu":     proximal_mu,
            "lr_local":        lr_local,
            "momentum_local":  momentum_local,
            "dp_delta":        dp_delta,
            "sparsity_ratio":  sparsity_ratio,
        })

    # FedAvg strategy (Ditto uses FedAvg on the server side)
    strategy = FedAvg(
        train_metrics_aggr_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggr_fn=evaluate_metrics_aggregation_fn,
    )

    # Start the simulation loop
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=fit_config(1), # Passed directly; Flower will reuse it or you can pass a dict of configs per round if needed
        num_rounds=num_rounds,
    )
    
    # Save the final global model
    print("\nSaving final model to disk...")
    import torch
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
