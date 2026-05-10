from typing import List, Tuple
from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import csv
import os

class MetricLogger:
    def __init__(self):
        self.fit_round = 1
        self.eval_round = 1

logger = MetricLogger()

def write_to_csv(round_num, phase, metrics):
    file_exists = os.path.isfile('results_ditto-quant-ldp-sparse.csv')
    with open('results_ditto-quant-ldp-sparse.csv', mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Round', 'Phase', 'Metric', 'Value'])
        for k, v in metrics.items():
            writer.writerow([round_num, phase, k, v])

def fit_metrics_aggregation_fn(metrics):
    aggregated = {}
    if not metrics:
        return aggregated
    for metric_name in metrics[0][1].keys():
        aggregated[metric_name] = sum([m[metric_name] for _, m in metrics]) / len(metrics)
    
    print(f"\n--- FIT KPIs (Round {logger.fit_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")
        
    write_to_csv(logger.fit_round, 'FIT', aggregated)
    logger.fit_round += 1
    return aggregated

def evaluate_metrics_aggregation_fn(metrics):
    aggregated = {}
    if not metrics:
        return aggregated
        
    accuracies = [num_example * m["accuracy"] for num_example, m in metrics]
    total_num_examples = sum([num_example for num_example, m in metrics])
    aggregated["accuracy"] = sum(accuracies) / total_num_examples
    
    acc_list = [m["accuracy"] for _, m in metrics]
    mean_acc = sum(acc_list) / len(acc_list)
    variance = sum([(a - mean_acc) ** 2 for a in acc_list]) / len(acc_list)
    aggregated["accuracy_stddev"] = variance ** 0.5
    
    for metric_name in metrics[0][1].keys():
        if metric_name != "accuracy":
            aggregated[metric_name] = sum([m[metric_name] for _, m in metrics]) / len(metrics)
            
    print(f"\n--- EVAL KPIs (Round {logger.eval_round}) ---")
    for k, v in aggregated.items():
        print(f"  > {k}: {v:.4f}")
        
    write_to_csv(logger.eval_round, 'EVAL', aggregated)
    logger.eval_round += 1
    return aggregated

def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config.get("num-server-rounds", 3)
    config = ServerConfig(num_rounds=num_rounds)
    
    # Strategy FedAvg (for Ditto architecture)
    strategy = FedAvg(
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )
    
    return ServerAppComponents(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)