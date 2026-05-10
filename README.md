# Federated Learning with Flower, PyTorch, and TensorFlow

This repository demonstrates how to set up a basic Federated Learning environment using the [Flower](https://flower.ai/) framework. It includes examples of centralized training (both PyTorch and TensorFlow) and a federated training setup using the modern Flower App architecture (`ServerApp` / `ClientApp`).

## Requirements

Ensure you have the following installed:
- Python 3.8+
- PyTorch & Torchvision
- TensorFlow
- Flower (with simulation extras)

You can install the necessary dependencies with:
```bash
pip install "flwr[simulation]>=1.9.0" torch torchvision tensorflow opacus
```

## Running the Code

### 1. Centralized Training (Baseline)

You can run the centralized training scripts to train the model on a single machine without federated learning. This is useful as a baseline.

**For PyTorch:**
```bash
python centralized.py
```

**For TensorFlow:**
```bash
python centralized_tf.py
```

### 2. Federated Learning (Simulation)

This project uses the modern **Flower Next** architecture. You no longer need to run the server and clients manually in separate terminals! Flower's simulation engine will automatically orchestrate the server and the clients for you.

Simply open your terminal, go to the project folder, and run:
```bash
flwr run . (lance en arrière-plan)
flwr run . --stream (affiche les logs en direct)
flwr run . --stream --federation-config "num-supernodes=2" (lance la simulation avec 2 clients)
```

By default, this will launch a server strategy and simulate 2 clients locally.

*Note : The number of rounds is defined in `pyproject.toml` under the `[tool.flwr.app.config]` section. You can easily change it there.*

## Switching between PyTorch and TensorFlow

The PyTorch and TensorFlow centralized files (`centralized.py` and `centralized_tf.py`) share the exact same function signatures.

If you wish to switch your federated client to use TensorFlow instead of PyTorch, you can simply change the import statement in `client.py` from:
```python
from centralized import load_data, load_model, train, test
```
to:
```python
from centralized_tf import load_data, load_model, train, test
```
