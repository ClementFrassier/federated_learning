import csv
import os
import torch
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedProx
from flwr.app import Context, ConfigRecord, ArrayRecord, MetricRecord

from task import load_model, NetFedRepScaffold

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
    csv_file = getattr(logger, "csv_file", CSV_FILE)
    csv_dir = os.path.dirname(csv_file)
    os.makedirs(csv_dir, exist_ok=True)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
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
      - Handles non-IID data across the 15 nurses with a proximal term
      - All weights are aggregated (no FedBN exclusion needed for MLP)
    """
    num_rounds  = int(context.run_config.get("num-server-rounds", 5))
    proximal_mu = float(context.run_config.get("proximal_mu",     0.05))
    seed        = int(context.run_config.get("partition-seed",   42))
    sigma       = float(context.run_config.get("noise-multiplier", 0.8))
    model_type  = context.run_config.get("model-type", "tiny")
    dp_enabled  = str(context.run_config.get("dp-enabled", True)).lower() != "false"
    excluded_node_id = int(context.run_config.get("excluded-node-id", -1))
    loss_type   = context.run_config.get("loss-type", "ce")
    personalization_mode = context.run_config.get("personalization-mode", "none")
    test_dir    = os.environ.get("FLWR_TEST_DIR", None)

    # Pick the output CSV path for this run's condition, most specific first
    # (combined ablations before single-axis ones) so no run is misfiled into
    # a more generic bucket than it belongs in.
    base_results_dir = CSV_DIR
    if model_type == "medium":
        csv_file = os.path.join(base_results_dir, "ablation_model_size", f"seed{seed}_sigma{sigma:.1f}.csv")
    elif test_dir or seed == 999:
        csv_file = os.path.join(base_results_dir, "test", f"seed{seed}_sigma{sigma:.1f}.csv")
    elif not dp_enabled:
        csv_file = os.path.join(base_results_dir, "ablation_no_dp", f"seed{seed}.csv")
    elif personalization_mode == "fedper" and excluded_node_id >= 0:
        csv_file = os.path.join(base_results_dir, f"ablation_fedper_leave_out_client{excluded_node_id}", f"seed{seed}.csv")
    elif personalization_mode == "fedrep_scaffold" and excluded_node_id >= 0:
        csv_file = os.path.join(base_results_dir, f"ablation_fedrep_scaffold_leave_out_client{excluded_node_id}", f"seed{seed}.csv")
    elif excluded_node_id >= 0:
        csv_file = os.path.join(base_results_dir, f"ablation_leave_out_client{excluded_node_id}", f"seed{seed}.csv")
    elif personalization_mode == "fedper":
        csv_file = os.path.join(base_results_dir, "ablation_fedper", f"seed{seed}_sigma{sigma:.1f}.csv")
    elif personalization_mode == "fedrep_scaffold":
        csv_file = os.path.join(base_results_dir, "ablation_fedrep_scaffold", f"seed{seed}_sigma{sigma:.1f}.csv")
    elif loss_type == "weighted_ce":
        csv_file = os.path.join(base_results_dir, "ablation_weighted_loss", f"seed{seed}.csv")
    else:
        csv_file = os.path.join(base_results_dir, "grid_fl", f"seed{seed}_sigma{sigma:.1f}.csv")

    csv_dir = os.path.dirname(csv_file)
    os.makedirs(csv_dir, exist_ok=True)

    if os.path.exists(csv_file):
        print(f"[WARNING] {os.path.basename(csv_file)} already exists and will be overwritten")
        os.remove(csv_file)

    logger.csv_file = csv_file

    # Initialise global model — all parameters sent to clients
    if personalization_mode == "fedrep_scaffold":
        # Distinct architecture (extractor/head split) — see task.py. Only the
        # extractor is meaningfully aggregated; the head keys are absent from
        # client replies so they naturally disappear from the aggregate after
        # round 1 regardless of what's in this initial broadcast.
        global_model  = NetFedRepScaffold()
        global_params = {k: v.cpu() for k, v in global_model.extractor.state_dict().items()}
    else:
        global_model  = load_model(model_type)
        global_params = {k: v.cpu() for k, v in global_model.state_dict().items()}
    arrays = ArrayRecord(torch_state_dict=global_params)

    # SCAFFOLD replaces FedProx's proximal term as the drift-correction
    # mechanism for the shared extractor — combining both would confound two
    # competing corrections (see report for the reasoning). mu=0 here makes
    # this explicit at the server level too, even though train_fedprox_dp
    # (which reads proximal_mu) is never called by this mode's dedicated
    # client-side training path either way.
    effective_mu = 0.0 if personalization_mode == "fedrep_scaffold" else proximal_mu

    strategy = FedProx(
        proximal_mu=effective_mu,
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
    
    unique_model_file = csv_file.replace(".csv", ".pt")
    print(f"Saving run-specific model checkpoint to '{unique_model_file}'...")
    torch.save(result.arrays.to_torch_state_dict(), unique_model_file)
    print(f"Done! FL results saved to '{logger.csv_file}'")
