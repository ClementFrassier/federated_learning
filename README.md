# Flower v2 - FedProx + FedBN + DP + SecAgg KPIs

This repository contains an advanced **Federated Learning** simulation utilizing the **Flower v2** architecture (`ClientApp` / `ServerApp`). 
The implementation tracks complex KPIs while integrating a realistic non-IID setup on the **FashionMNIST** dataset.

## Architecture

This project strictly preserves your original complex logic (FedProx, Opacus Local DP, GroupNorm/FedBN) but adapts it to the Native Flower v2 Engine:

- **`task.py`**: Defines the robust PyTorch CNN `Net` with `GroupNorm` (for FedBN). Contains the `PrivacyEngine` from `opacus` for Local Differential Privacy during the `FedProx` training loop. Implements the original manual IID/non-IID subset partitioning of FashionMNIST.
- **`client_app.py`**: A `ClientApp` using `@app.train` and `@app.evaluate`. It securely exchanges `Message` payloads, rigorously filtering out GroupNorm weights from global synchronization. Computes CPU time, RAM (`tracemalloc`), quantization error, DP epsilon, and communication size limits, returning them to the server via `MetricRecord`.
- **`server_app.py`**: A `ServerApp` using `@app.main`. Employs the `FedProx` aggregation strategy to penalize diverging clients, and aggregates all custom KPIs into the `results_fedProx+fedBN+standDP+SecAgg.csv` logger.
- **`pyproject.toml`**: Stores dependencies (including `opacus` and `flwr`) and hyperparameter configurations.

## Requirements

Ensure you have Python 3.8+ installed.

```bash
pip install -e .
```

*Required libraries: `flwr[simulation]`, `torch`, `torchvision`, `opacus`, `numpy`.*

## Running the Simulation

You do not need to open multiple terminals to launch the server and clients separately. Flower's internal simulation engine will orchestrate the federated components dynamically.

To start the simulation, navigate to this directory and execute:

```bash
flwr run .
```

### Configuration

You can alter hyperparameters directly within `pyproject.toml` under `[tool.flwr.app.config]` without touching the Python code:
- `num-server-rounds`: Default 3.
- `batch-size`: Default 64.
- `local-epochs`: Default 1.
- `proximal_mu`: FedProx penalty parameter (default 0.1).

After execution, all KPIs are appended to the `results_fedProx+fedBN+standDP+SecAgg.csv` file, and the finalized model is saved as `final_model.pt`.
