import csv
import os
import torch
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedProx
from flwr.app import Context, ConfigRecord, ArrayRecord, MetricRecord

from task import Net

CSV_DIR  = "resultsfeat"
CSV_FILE = os.path.join(CSV_DIR, "results_fedProx+fedBN+standDP+SecAgg.csv")


class MetricLogger:
    def __init__(self):
        self.fit_round = 1
        self.eval_round = 1

logger = MetricLogger()

def write_to_csv(round_num: int, phase: str, metrics: dict) -> None:
    """Append one row per metric to the KPI CSV file."""
    os.makedirs(CSV_DIR, exist_ok=True)
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
    num_rounds  = int(context.run_config.get("num-server-rounds", 3))
    # Read proximal_mu from run_config so it stays consistent with client_app.py.
    # The value passed to FedProx() here is used by the server-side strategy;
    # the client reads the same key from context.run_config independently.
    proximal_mu = float(context.run_config.get("proximal_mu", 0.1))

    # Initialise global model — server only tracks non-GN layers (FedBN)
    global_model  = Net()
    global_params = {k: v.cpu() for k, v in global_model.state_dict().items() if "gn" not in k}
    # ✅ Correct ArrayRecord constructor — must use the torch_state_dict kwarg
    arrays = ArrayRecord(torch_state_dict=global_params)

    strategy = FedProx(
        proximal_mu=proximal_mu,
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
