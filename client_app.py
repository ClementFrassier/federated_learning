import warnings
warnings.filterwarnings("ignore")

import time
import tracemalloc
import torch
import torch.quantization
from collections import OrderedDict
import numpy as np

from flwr.clientapp import ClientApp
from flwr.common import Context, Message, RecordDict, ArrayRecord, MetricRecord

from task import load_data, load_model, train_fedprox_dp, test, get_model_size

app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:
    start_time = time.time()
    tracemalloc.start()
    
    # 1. Init model
    model = load_model()
    
    # 2. Extract received weights (excluding gn from global)
    # The server sends the global weights
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()
    
    # We update the local model ONLY with the non-GN weights we received
    state_dict = model.state_dict()
    keys_to_update = [k for k in state_dict.keys() if "gn" not in k]
    update_dict = OrderedDict({k: incoming_state_dict[k] for k in keys_to_update if k in incoming_state_dict})
    state_dict.update(update_dict)
    model.load_state_dict(state_dict, strict=True)
    
    # 3. Load Data
    node_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 10)
    batch_size = context.run_config.get("batch-size", 64)
    trainloader, _ = load_data(node_id=node_id, num_clients=num_partitions, batch_size=batch_size)
    
    # Global params for FedProx
    global_params_dict = {k: v for k, v in model.state_dict().items() if "gn" not in k}
    
    # 4. Train FedProx + DP
    mu = context.run_config.get("proximal_mu", 0.1)
    epochs = context.run_config.get("local-epochs", 1)
    model, dp_epsilon = train_fedprox_dp(model, global_params_dict, trainloader, epochs=epochs, mu=mu)
    
    # 5. Prepare weights to return (exclude GN)
    state_to_return = {k: v.cpu() for k, v in model.state_dict().items() if "gn" not in k}
    model_record = ArrayRecord(state_to_return)
    
    # Metrics
    comm_size_mb = sum([p.element_size() * p.nelement() for p in state_to_return.values()]) / (1024 * 1024)
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    fit_time = time.time() - start_time
    peak_ram_mb = peak_ram / (1024 * 1024)
    estimated_energy = (2.0 * fit_time) + (15.0 * comm_size_mb)
    
    metrics = {
        "fit_time": float(fit_time),
        "peak_ram_mb": float(peak_ram_mb),
        "comm_size_mb": float(comm_size_mb),
        "model_size_mb": float(get_model_size(model)),
        "dp_epsilon": float(dp_epsilon),
        "estimated_energy": float(estimated_energy),
        "num-examples": len(trainloader.dataset)
    }
    
    content = RecordDict({
        "arrays": model_record,
        "metrics": MetricRecord(metrics)
    })
    
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    start_time = time.time()
    
    # 1. Init models
    model = load_model()
    model_before_fit = load_model()
    
    # We update the local model ONLY with the non-GN weights we received
    incoming_state_dict = msg.content["arrays"].to_torch_state_dict()
    state_dict = model.state_dict()
    keys_to_update = [k for k in state_dict.keys() if "gn" not in k]
    update_dict = OrderedDict({k: incoming_state_dict[k] for k in keys_to_update if k in incoming_state_dict})
    state_dict.update(update_dict)
    
    model.load_state_dict(state_dict, strict=True)
    model_before_fit.load_state_dict(state_dict, strict=True)
    
    # 2. Load Data
    node_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 10)
    batch_size = context.run_config.get("batch-size", 64)
    _, testloader = load_data(node_id=node_id, num_clients=num_partitions, batch_size=batch_size)
    
    # EVALUATION 1 : Global
    _, acc_global = test(model_before_fit, testloader)
    
    # EVALUATION 2 : Local fp32
    _, acc_local_fp32 = test(model, testloader)
    
    # EVALUATION 3 : Local Quantized
    net_cpu = load_model().cpu()
    net_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    net_quantized = torch.quantization.quantize_dynamic(
        net_cpu, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    loss_quantized, acc_quantized = test(net_quantized, testloader, device=torch.device('cpu'))
    
    eval_time = time.time() - start_time
    local_vs_global_gap = acc_local_fp32 - acc_global
    quantization_error = acc_local_fp32 - acc_quantized
    
    metrics = {
        "accuracy": float(acc_quantized),
        "acc_global": float(acc_global),
        "acc_local_fp32": float(acc_local_fp32),
        "local_vs_global_gap": float(local_vs_global_gap),
        "quantization_error": float(quantization_error),
        "eval_time": float(eval_time),
        "quantized_model_size_mb": float(get_model_size(net_quantized)),
        "loss": float(loss_quantized),
        "num-examples": len(testloader.dataset)
    }
    
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
