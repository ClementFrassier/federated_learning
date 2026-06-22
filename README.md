# Federated Learning — Nurse Stress Detection (MLP Tiny + FedProx + DP-SGD)

This repository implements a **privacy-preserving Federated Learning** pipeline for stress detection in nurses using physiological signals from the [Nurse Stress Dataset (Empatica E4)](https://physionet.org/content/nstdb/1.0.0/). The system simulates a real **TinyML + Wristband** deployment scenario using the [Flower v2](https://flower.ai/) framework.

## Architecture

| Component | Description |
|---|---|
| **Model** | MLP Tiny: `FC(24→32)→ReLU→FC(32→16)→ReLU→FC(16→2)` — ~1,362 parameters, ~5.4 KB FP32 |
| **Strategy** | FedProx (proximal term µ=0.05 to handle non-IID data across nurses) |
| **Privacy** | DP-SGD via Opacus (`noise_multiplier=0.8`, `max_grad_norm=1.0`, `δ=1e-5`) |
| **Clients** | 15 nurses, each owning their physiological data (natural partition by nurse ID) |
| **Signals** | EDA, HR, TEMP, X, Y, Z (accelerometer) — windowed at 60 samples / 30 stride |
| **Features** | `[mean, std, min, max] × 6 signals = 24 features` per window |
| **Deployment** | INT8 dynamic quantization for MCU deployment (STM32 / ESP32 / Cortex-M) |

## Project Structure

```
pytorch_flwr/
├── task.py                    # Model (Net), data loading, feature extraction, FedProx+DP training
├── client_app.py              # Flower ClientApp: @train / @evaluate decorators, Opacus DP engine
├── server_app.py              # Flower ServerApp: FedProx strategy, KPI aggregation, CSV logging
├── baseline_centralized.py   # Centralized baseline (same model, no FL, no DP) for comparison
├── pyproject.toml             # Project config & Flower hyperparameters
├── KPI.md                     # Description of all tracked metrics
├── REPORT_STRESS_FL.md        # Full technical report
└── data_nurse/
    └── merged_data.csv        # Nurse stress dataset (EDA, HR, TEMP, X, Y, Z, id, label)
```

## Requirements

Python 3.10+ is required (Opacus + Flower v2 constraint).

Install all dependencies via the project definition:

```bash
pip install -e .
```

> Core dependencies: `flwr[simulation]`, `opacus`, `torch`, `torchvision`, `pandas`, `numpy`

## Running the Simulation

Flower v2 uses a **simulation engine** — no need to launch server and clients in separate terminals.

### 1. Federated Learning (main experiment)

```bash
flwr run .
```

Results are automatically saved to:
- `resultsfeat/results_nurse_mlp_tiny.csv` — per-round FL KPIs (accuracy, dp_epsilon, comm_size_mb, …)
- `final_model_nurse.pt` — final aggregated model weights

### 2. Centralized Baseline (for comparison)

```bash
python baseline_centralized.py
```

Results are saved to:
- `resultsfeat/results_baseline_centralized.csv`

## Configuration

All hyperparameters are defined in `pyproject.toml` under `[tool.flwr.app.config]`. No Python code changes needed to tune the experiment:

```toml
[tool.flwr.app.config]
num-server-rounds = 30       # Number of FL rounds
local-epochs      = 2        # Local training epochs per client per round
batch-size        = 64
learning-rate     = 0.01
proximal_mu       = 0.05     # FedProx proximal penalty (non-IID robustness)
noise-multiplier  = 0.8      # Opacus DP-SGD noise (ε ≈ 1.5–2.0 after 30 rounds)
max-grad-norm     = 1.0      # Gradient clipping bound
dp-delta          = 1e-5     # DP delta (privacy guarantee target)
partition-seed    = 42
```

## Key Metrics (KPIs)

| Metric | Description |
|---|---|
| `accuracy` (INT8) | Primary TinyML metric — quantized model accuracy |
| `acc_global` (FP32) | Accuracy of the raw server model (no local adaptation) |
| `acc_local_fp32` | Accuracy after local FedProx fine-tuning |
| `local_vs_global_gap` | Value added by local training over the global model |
| `dp_epsilon` (ε) | Accumulated privacy budget (target: ε ≤ 2.0 for PoC) |
| `accuracy_stddev` | Std-dev of accuracy across all 15 nurses (fairness KPI) |
| `quantization_error` | FP32 → INT8 accuracy drop (target: < 2%) |
| `comm_size_mb` | Size of model updates sent to server per round (~5.2 KB) |
| `estimated_energy` | Energy heuristic: `2.0 × fit_time + 15.0 × comm_size_mb` |

See [`KPI.md`](KPI.md) for full definitions and [`REPORT_STRESS_FL.md`](REPORT_STRESS_FL.md) for the complete technical report.

## Experiment Branches

This repository contains several experimental branches, each exploring a different FL algorithm:

| Branch | Algorithm | Notes |
|---|---|---|
| `feat/nurse-stress-fl` | **FedProx + DP-SGD** | ✅ Main branch — recommended setup |
| `feat/fedgn-only` | FedGN (all-global baseline) | No personalisation |
| `feat/fedbn-only` | FedBN (local BN layers) | BatchNorm personalisation |
| `feat/ditto-only` | Ditto | Dual-model personalisation |
| `feat/ditto-FedYogi-sdp-secAgg-sparse` | Ditto + FedYogi + SDP + SecAgg | Advanced privacy stack |
| `feat/ditto-quant-ldp-sparse` | Ditto + LDP + quantisation | Local differential privacy |
| `feat/fedProx+fedBN+standDP+SecAgg` | FedProx + FedBN + standard DP | Combined approach |
