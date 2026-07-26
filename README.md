# Flower v2 - FedBN Only

This repository contains a **Federated Learning** simulation utilizing the **Flower v2** architecture (`ClientApp` / `ServerApp`). 
This version implements **FedBN** only, using standard **FedAvg** strategy and keeping the real BatchNorm layers (weight, bias, running_mean, running_var) local to each client. There is no differential privacy on this branch, which is what makes a genuine BatchNorm layer usable here in the first place: Opacus refuses to train a model containing BatchNorm, since its cross-sample statistics break the per-example gradient isolation DP-SGD requires.

## Architecture

This version isolates FedBN behavior:
- **`task.py`**: Defines the PyTorch CNN `Net` with `BatchNorm2d` (for FedBN). Implements the standard local training loop (SGD) and evaluation.
- **`client_app.py`**: A `ClientApp` using `@app.train` and `@app.evaluate`. It exchanges `Message` payloads, filtering out BatchNorm weights (and running statistics) from global synchronization. Computes CPU time, RAM (`tracemalloc`), quantization error, and communication size, returning them to the server via `MetricRecord`.
- **`server_app.py`**: A `ServerApp` using `@app.main`. Employs the `FedAvg` aggregation strategy to aggregate non-BN client weights, and aggregates all custom KPIs into the `results_fedbn-only.csv` logger.
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
