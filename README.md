# Federated Learning — Ditto + Quantisation + Local-DP + Sparse

A production-ready Federated Learning simulation built on **Flower v2** (`ServerApp` / `ClientApp` architecture), trained on **FashionMNIST**.

## Architecture

This project implements a research-grade combination of four complementary techniques:

| Component | Role |
|---|---|
| **Ditto** | Two-model personalisation (global DP model + local proximal model) |
| **Local Differential Privacy** | Opacus `PrivacyEngine` wraps the global model with DP-SGD |
| **Sparse Upload** | 50 % magnitude pruning applied to global weights before uplink |
| **INT8 Quantisation** | Dynamic `qint8` quantisation of the local model for TinyML evaluation |

### Files

| File | Description |
|---|---|
| `task.py` | `Net` model (GroupNorm CNN), `load_data`, `train_ditto_dp`, `test`, `apply_sparsification`, `get_model_size` |
| `client_app.py` | `ClientApp` via `NumPyClient` — full Ditto training loop + KPI metrics |
| `server_app.py` | `ServerApp` — FedAvg aggregation + per-round config + CSV KPI logger |
| `pyproject.toml` | Dependencies and Flower run configuration |

## Requirements

Python 3.10+ required (Flower 1.26+ dropped support for 3.8/3.9). Install all dependencies via:

```bash
pip install -e .
```

Core dependencies: `flwr[simulation]>=1.13.0`, `torch`, `torchvision`, `opacus`, `numpy`.

## Running the Simulation

A single command launches the full simulation — no need for separate terminal windows:

```bash
flwr run .
```

### Configuration

Edit `[tool.flwr.app.config]` in `pyproject.toml` to change hyperparameters without touching Python code:

| Key | Default | Description |
|---|---|---|
| `num-server-rounds` | `30` | Number of federated rounds |

Per-round client hyperparameters (`proximal_mu`, `local_epochs`) are set in `server_app.py → fit_config()`.

## KPI Output

All metrics are appended round-by-round to `results_ditto-quant-ldp-sparse.csv`:

**FIT metrics** (per round, averaged across clients):
- `fit_time`, `peak_ram_mb`, `comm_size_mb`, `model_size_mb`
- `dp_epsilon` — cumulative DP privacy budget consumed
- `estimated_energy` — energy proxy (α·time + β·comm_size)

**EVAL metrics** (per round, aggregated):
- `accuracy` — INT8 quantised model (production target)
- `acc_global` — global model (server aggregation baseline)
- `acc_local_fp32` — personalised FP32 model
- `local_vs_global_gap` — personalisation gain from Ditto
- `quantization_error` — accuracy loss from INT8 quantisation
- `accuracy_stddev` — cross-client fairness indicator
- `quantized_model_size_mb`, `eval_time`, `loss`
