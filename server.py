import flwr as fl

def weighted_average(metrics):
    accuracies = [num_example * m["accuracy"] for num_example, m in metrics]
    total_num_examples = sum([num_example for num_example, m in metrics])
    return {"accuracy":sum(accuracies) / total_num_examples}

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),    
    #autre arg de strategy : fraction_fit : Pourcentage de clients à utiliser pour l'entraînement,min_available_clients : Nombre minimum de clients connectés avant de lancer le round
    strategy=fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average),
)   