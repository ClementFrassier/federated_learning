# Flower v2 - FedBN Only

This repository contains a **Federated Learning** simulation utilizing the **Flower v2** architecture (`ClientApp` / `ServerApp`). 
This version implements **FedBN** only, using standard **FedAvg** strategy and keeping the GroupNorm layers local to each client.

## Architecture

This version isolates FedBN behavior:
- **`task.py`**: Defines the PyTorch CNN `Net` with `GroupNorm` (for FedBN). Implements the standard local training loop (SGD) and evaluation.
- **`client_app.py`**: A `ClientApp` using `@app.train` and `@app.evaluate`. It exchanges `Message` payloads, filtering out GroupNorm weights from global synchronization. Computes CPU time, RAM (`tracemalloc`), quantization error, and communication size, returning them to the server via `MetricRecord`.
- **`server_app.py`**: A `ServerApp` using `@app.main`. Employs the `FedAvg` aggregation strategy to aggregate non-GN client weights, and aggregates all custom KPIs into the `results_fedbn-only.csv` logger.
- **`pyproject.toml`**: Stores dependencies (including `flwr`) and hyperparameter configurations.

## Requirements

Python **3.10+** required. Install all dependencies via:

```bash
pip install -e .
```

*Required libraries: `flwr[simulation]`, `torch`, `torchvision`, `numpy`.*

## Running the Simulation

To start the simulation, execute:

```bash
flwr run .
```

### Configuration

You can alter hyperparameters directly within `pyproject.toml` under `[tool.flwr.app.config]`:
- `num-server-rounds`: Default 10.
- `batch-size`: Default 16.
- `local-epochs`: Default 2.
- `lr-local`: SGD learning rate (default 0.01).
- `momentum`: SGD momentum (default 0.9).

After execution, all KPIs are appended to the `resultsfeat/results_fedbn-only.csv` file, and the finalized model is saved as `final_model.pt`.
