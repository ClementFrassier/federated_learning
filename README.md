# Flower v2 - FedProx + FedGN + DP KPIs

This repository contains an advanced **Federated Learning** simulation utilizing the **Flower v2** architecture (`ClientApp` / `ServerApp`). 
The implementation tracks complex KPIs while integrating a realistic non-IID setup on the **FashionMNIST** dataset.

**Note on Secure Aggregation:** this branch was previously named
`feat/fedProx+fedBN+standDP+SecAgg`, but no SecAgg code was ever
present, and its normalization layer was GroupNorm, not BatchNorm (the
former is what Opacus DP-SGD actually supports; BatchNorm's cross-sample
statistics break the per-example gradient isolation DP-SGD requires,
see `feat/fedbn-only` for the DP-free branch with real BatchNorm).
Flower's native SecAgg+ protocol (`SecAggPlusWorkflow` / `secaggplus_mod`)
only exists for the legacy `NumPyClient` / `Workflow` / `Driver` API
(verified directly against the installed `flwr==1.29.0`:
`flwr.serverapp.strategy` and `flwr.clientapp.mod` expose no SecAgg
equivalent), not for the message-passing `ServerApp` / `ClientApp` API
used here, so the branch was renamed on both counts rather than claim
features that cannot actually run in this codebase.

## Architecture

This project combines FedProx, Opacus DP-SGD, and GroupNorm-based FedGN on the Flower v2 message-passing engine:

- **`task.py`**: Defines the PyTorch CNN `Net` with `GroupNorm` (for FedGN). Contains the `PrivacyEngine` from `opacus` for differential privacy during the `FedProx` training loop. Implements manual IID/non-IID subset partitioning of FashionMNIST.
- **`client_app.py`**: A `ClientApp` using `@app.train` and `@app.evaluate`. It exchanges `Message` payloads, filtering out GroupNorm weights from global synchronization. Computes CPU time, RAM (`tracemalloc`), quantization error, DP epsilon, and communication size, returning them to the server via `MetricRecord`.
- **`server_app.py`**: A `ServerApp` using `@app.main`. Employs the `FedProx` aggregation strategy to penalize diverging clients, and aggregates all custom KPIs into the `results_fedProx+fedGN+standDP.csv` logger.
- **`pyproject.toml`**: Stores dependencies (including `opacus` and `flwr`) and hyperparameter configurations.

## Requirements

Python **3.10+** required (Flower 1.26+ dropped support for 3.8/3.9). Install all dependencies via:

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

After execution, all KPIs are appended to the `results_fedProx+fedGN+standDP.csv` file, and the finalized model is saved as `final_model.pt`.
