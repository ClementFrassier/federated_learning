import csv
import os
import torch
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import Context, ConfigRecord, ArrayRecord, MetricRecord

from task import Net

CSV_DIR  = "resultsfeat"
CSV_FILE = os.path.join(CSV_DIR, "results_fedbn-only.csv")


class MetricLogger:
    """Stateful round counter — resets CSV at run start to avoid duplicate rows."""
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
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Round', 'Phase', 'Metric', 'Value'])
        for k, v in metrics.items():
            writer.writerow([round_num, phase, k, v])

def train_metrics_aggr_fn(record_dicts, weighted_by):
    aggregated = {}
    if not record_dicts:
        return MetricRecord(aggregated)
    metrics_list = [rd.metric_records["metrics"] for rd in record_dicts]
    for metric_name in metrics_list[0].keys():
        if metric_name == "num-examples":
            continue
        aggregated[metric_name] = sum([m[metric_name] for m in metrics_list]) / len(metrics_list)
    
    print(f"\n--- FIT KPIs (Round {logger.fit_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")
        
    write_to_csv(logger.fit_round, 'FIT', aggregated)
    logger.fit_round += 1
    return MetricRecord(aggregated)

def evaluate_metrics_aggregation_fn(record_dicts, weighted_by):
    aggregated = {}
    if not record_dicts:
        return MetricRecord(aggregated)
        
    metrics_list = [rd.metric_records["metrics"] for rd in record_dicts]
        
    accuracies = [m["num-examples"] * m["accuracy"] for m in metrics_list]
    total_num_examples = sum([m["num-examples"] for m in metrics_list])
    aggregated["accuracy"] = sum(accuracies) / total_num_examples
    
    acc_list = [m["accuracy"] for m in metrics_list]
    mean_acc = sum(acc_list) / len(acc_list)
    variance = sum([(a - mean_acc) ** 2 for a in acc_list]) / len(acc_list)
    aggregated["accuracy_stddev"] = variance ** 0.5
    
    for metric_name in metrics_list[0].keys():
        if metric_name not in ["accuracy", "num-examples"]:
            aggregated[metric_name] = sum([m[metric_name] for m in metrics_list]) / len(metrics_list)
            
    print(f"\n--- EVAL KPIs (Round {logger.eval_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")
        
    write_to_csv(logger.eval_round, 'EVAL', aggregated)
    logger.eval_round += 1
    return MetricRecord(aggregated)

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds  = int(context.run_config.get("num-server-rounds", 10))

    # Initialise global model -- server only tracks non-BN layers (FedBN)
    global_model  = Net()
    global_params = {k: v.cpu() for k, v in global_model.state_dict().items() if "bn" not in k}
    arrays = ArrayRecord(torch_state_dict=global_params)

    strategy = FedAvg(
        train_metrics_aggr_fn=train_metrics_aggr_fn,
        evaluate_metrics_aggr_fn=evaluate_metrics_aggregation_fn,
    )
    
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({}),
        num_rounds=num_rounds
    )
    
    # Save final model
    print("\nSaving final model to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
