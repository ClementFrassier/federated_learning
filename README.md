# Flower v2 - FedGN Only (Baseline)

This repository contains a **Federated Learning** simulation utilizing the **Flower v2** architecture (`ClientApp` / `ServerApp`).

This version implements **FedGN** — the **non-personalised baseline** where **all** model layers, including the GroupNorm (GN) layers, are synchronised globally via standard `FedAvg`.

> **Comparison role:** Use this branch as the baseline to quantify the gain from:
> - `feat/fedbn-only` — where GN layers stay **local** (FedBN personalisation)
> - `feat/ditto-only` — where a proximal term drives local personalisation (Ditto)

## Architecture

- **`task.py`**: CNN `Net` with `GroupNorm` layers. Standard SGD training loop. Dirichlet non-IID data partitioning (`alpha` parameter).
- **`client_app.py`**: Receives **all** global weights (including GN), fully overwrites the local model, trains with SGD, and returns **all** weights. No GN filtering. No persistent personalised model.
- **`server_app.py`**: `FedAvg` strategy that aggregates **all** weights including GN. Logs KPIs to `resultsfeat/results_fedgn-only.csv`.
- **`pyproject.toml`**: Hyperparameter configuration, including `alpha` for Dirichlet non-IID partitioning.

### Key difference vs other branches

| Branch | GN layers | Strategy |
|--------|-----------|----------|
| `feat/fedgn-only` | **Global** (shared) | FedAvg baseline |
| `feat/fedbn-only` | **Local** (not shared) | FedBN personalisation |
| `feat/ditto-only` | Global + local proximal | Ditto personalisation |

## Requirements

Python **3.10+** required. Install all dependencies via:

```bash
pip install -e .
```

*Required libraries: `flwr[simulation]`, `torch`, `torchvision`, `numpy`.*

## Running the Simulation

```bash
flwr run .
```

### Configuration

Edit `pyproject.toml` under `[tool.flwr.app.config]`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-server-rounds` | 10 | Total FL rounds |
| `batch-size` | 16 | Mini-batch size (keep low to avoid OOM) |
| `local-epochs` | 2 | Local epochs per round |
| `lr-local` | 0.01 | SGD learning rate |
| `momentum` | 0.9 | SGD momentum |
| `alpha` | 0.3 | Dirichlet concentration (0=IID, 0.1=very non-IID) |
| `partition-seed` | 42 | Reproducibility seed |

After execution, all KPIs are appended to `resultsfeat/results_fedgn-only.csv`.

### Expected behaviour

Because FedGN has no personalisation mechanism, `local_vs_global_gap ≈ 0` by design on every round — this is intentional and proves that the gap observed in FedBN and Ditto branches comes from their respective personalisation strategies.
