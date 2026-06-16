import csv
import os
import torch
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedProx
from flwr.app import Context, ConfigRecord, ArrayRecord, MetricRecord

from task import Net

CSV_DIR  = "resultsfeat"
CSV_FILE = os.path.join(CSV_DIR, "results_nurse_mlp_tiny.csv")


# ── Metric logger (round counter) ─────────────────────────────────────────────
class MetricLogger:
    def __init__(self):
        self.fit_round  = 1
        self.eval_round = 1

logger = MetricLogger()


def write_to_csv(round_num: int, phase: str, metrics: dict) -> None:
    """Append one row per metric to the KPI CSV file."""
    os.makedirs(CSV_DIR, exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Round", "Phase", "Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([round_num, phase, k, v])


# ── Fit aggregation ───────────────────────────────────────────────────────────
def train_metrics_aggr_fn(record_dicts, weighted_by):
    """Average all FIT metrics across clients and log to CSV."""
    if not record_dicts:
        return MetricRecord({})

    metrics_list = [rd.metric_records["metrics"] for rd in record_dicts]
    aggregated   = {
        k: sum(m[k] for m in metrics_list) / len(metrics_list)
        for k in metrics_list[0]
        if k != "num-examples"
    }

    print(f"\n--- FIT KPIs (Round {logger.fit_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")

    write_to_csv(logger.fit_round, "FIT", aggregated)
    logger.fit_round += 1
    return MetricRecord(aggregated)


# ── Evaluate aggregation ──────────────────────────────────────────────────────
def evaluate_metrics_aggregation_fn(record_dicts, weighted_by):
    """
    Aggregate EVAL metrics:
    - accuracy       : weighted by num-examples (fair global accuracy)
    - accuracy_stddev: spread of INT8 accuracy across clients (fairness KPI)
    - all others     : simple mean
    """
    if not record_dicts:
        return MetricRecord({})

    metrics_list  = [rd.metric_records["metrics"] for rd in record_dicts]

    # Weighted accuracy (accounts for imbalanced client dataset sizes)
    total_examples = sum(m["num-examples"] for m in metrics_list)
    aggregated = {
        "accuracy": sum(
            m["num-examples"] * m["accuracy"] for m in metrics_list
        ) / total_examples
    }

    # Fairness: std-dev of INT8 accuracy across the 15 nurses
    acc_list = [m["accuracy"] for m in metrics_list]
    mean_acc = sum(acc_list) / len(acc_list)
    aggregated["accuracy_stddev"] = (
        sum((a - mean_acc) ** 2 for a in acc_list) / len(acc_list)
    ) ** 0.5

    # Simple mean for all other metrics
    for k in metrics_list[0]:
        if k not in ("accuracy", "num-examples"):
            aggregated[k] = sum(m[k] for m in metrics_list) / len(metrics_list)

    print(f"\n--- EVAL KPIs (Round {logger.eval_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")

    write_to_csv(logger.eval_round, "EVAL", aggregated)
    logger.eval_round += 1
    return MetricRecord(aggregated)


# ── Server app ────────────────────────────────────────────────────────────────
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Federated training orchestration.

    Strategy: FedProx
      - Handles non-IID data across the 14 nurses with a proximal term
      - All weights are aggregated (no FedBN exclusion needed for MLP)
    """
    num_rounds  = int(context.run_config.get("num-server-rounds", 5))
    proximal_mu = float(context.run_config.get("proximal_mu",     0.05))

    # Initialise global model — all parameters sent to clients
    global_model  = Net()
    global_params = {k: v.cpu() for k, v in global_model.state_dict().items()}
    arrays        = ArrayRecord(torch_state_dict=global_params)

    strategy = FedProx(
        proximal_mu=proximal_mu,
        train_metrics_aggr_fn=train_metrics_aggr_fn,
        evaluate_metrics_aggr_fn=evaluate_metrics_aggregation_fn,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({}),
        num_rounds=num_rounds,
    )

    # Save the final aggregated model
    print("\nSaving final model to 'final_model_nurse.pt'...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model_nurse.pt")
    print(f"Done! FL results saved to '{CSV_FILE}'")
