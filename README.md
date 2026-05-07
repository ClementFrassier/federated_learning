# Federated Learning with Flower, PyTorch, and TensorFlow

This repository demonstrates how to set up a basic Federated Learning environment using the [Flower](https://flower.ai/) framework. It includes examples of centralized training (both PyTorch and TensorFlow) and a federated training setup (client/server) using PyTorch.

## Requirements

Ensure you have the following installed:
- Python 3.8+
- PyTorch & Torchvision
- TensorFlow
- Flower (`flwr`)

You can install the necessary dependencies with:
```bash
pip install torch torchvision tensorflow flwr
```

## Running the Code

### 1. Centralized Training (Baseline)

You can run the centralized training scripts to train the model on a single machine without federated learning. This is useful as a baseline or for debugging your model architecture.

**For PyTorch:**
```bash
python centralized.py
```

**For TensorFlow:**
```bash
python centralized_tf.py
```

### 2. Federated Learning (Client / Server)

To run the federated learning simulation, you need to start the server and then start one or multiple clients.

**Step 1: Start the server**
Open a terminal and run the server script:
```bash
python server.py
```
*The server will start on `0.0.0.0:8080` and wait for clients to connect.*

**Step 2: Start the clients**
Open additional terminal windows (one for each client you want to simulate) and run the client script:
```bash
python client.py
```
*By default, the server configuration (`num_rounds=3`) and strategy (`FedAvg`) will coordinate the training across the connected clients. You may need to run at least two clients depending on your Flower version and FedAvg defaults.*

## Switching between PyTorch and TensorFlow

The PyTorch and TensorFlow centralized files (`centralized.py` and `centralized_tf.py`) share the exact same function signatures (`load_data`, `load_model`, `train`, `test`).

If you wish to switch your federated client to use TensorFlow instead of PyTorch, you can simply change the import statement in `client.py` from:
```python
from centralized import load_data, load_model, train, test
```
to:
```python
from centralized_tf import load_data, load_model, train, test
```

*(Note: You will also need to update `get_parameters` and `set_parameters` in `client.py` to handle TensorFlow weights instead of PyTorch tensors, as `flwr` provides different weight extraction methods for Keras models. In Keras, you can generally use `model.get_weights()` and `model.set_weights()`).*
