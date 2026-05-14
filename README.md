# Federated Learning — Ditto + FedYogi + Stepwise-DP + SecAgg + Sparse

Production-ready Federated Learning simulation built on **Flower v2** (`ServerApp` / `ClientApp`), trained on **FashionMNIST**.
Branch: `feat/ditto-FedYogi-sdp-secAgg-sparse`

## Architecture

This project implements five complementary techniques:

| Component | Role |
|---|---|
| **Ditto** | Two-model personalisation: global model (DP-SGD, sent to server) + local model (proximal SGD, kept on device) |
| **FedYogi** | Server-side adaptive optimiser (momentum + sign-based variance correction) replacing plain FedAvg |
| **Stepwise-DP** | Persistent Opacus `PrivacyEngine` accumulates ε across all rounds — realistic cumulative privacy accounting |
| **SecAgg** | Simulated — logged for traceability; in production, wrap strategy with `SecAggPlusWorkflow` |
| **Sparse uplink** | 50 % magnitude pruning applied to global weights before uplink — cuts effective communication by ~50 % |
| **INT8 quantisation** | Dynamic `qint8` quantisation of the local model — TinyML deployment proxy (evaluation only) |

### File structure

| File | Role |
|---|---|
| `task.py` | `Net` (GroupNorm CNN), `load_data`, `train_ditto_dp`, `test`, `apply_sparsification`, `get_model_size` |
| `client_app.py` | `ClientApp` — persistent PrivacyEngine, Ditto training loop, sparsification, INT8 eval, KPI metrics |
| `server_app.py` | `ServerApp` — FedYogi strategy, per-round config injection, CSV KPI logger |
| `pyproject.toml` | All hyperparameters + Flower run configuration |

## Requirements

Python 3.8+ required. Install all dependencies:

```bash
pip install -e .
```

Core dependencies: `flwr[simulation]>=1.13.0`, `torch`, `torchvision`, `opacus`, `numpy`.

## Running the Simulation

```bash
flwr run .
```

No separate terminal windows needed — Flower's simulation engine orchestrates everything.

## Configuration

All hyperparameters are in `pyproject.toml` under `[tool.flwr.app.config]`:

| Key | Default | Description |
|---|---|---|
| `num-server-rounds` | `3` | Total FL rounds |
| `local-epochs` | `1` | Local epochs per client per round |
| `batch-size` | `16` | Mini-batch size |
| `lr-global` | `0.01` | SGD lr — global (DP) model |
| `lr-local` | `0.001` | SGD lr — local (personalised) model |
| `momentum` | `0.9` | SGD momentum |
| `proximal-mu` | `0.01` | Ditto proximal penalty µ |
| `noise-multiplier` | `1.8` | Opacus DP-SGD noise σ |
| `max-grad-norm` | `1.0` | Gradient clipping norm C |
| `dp-delta` | `1e-5` | DP δ for ε reporting |
| `sparsity-ratio` | `0.5` | Fraction of weights zeroed before uplink |
| `fedyogi-eta` | `0.01` | FedYogi server learning rate η |
| `fedyogi-beta-1` | `0.9` | FedYogi first moment β₁ |
| `fedyogi-beta-2` | `0.99` | FedYogi second moment β₂ |
| `fedyogi-tau` | `1e-3` | FedYogi adaptivity τ |

Simulation resources (under `[tool.flwr.federations.local-simulation]`):

| Key | Default | Description |
|---|---|---|
| `options.num-supernodes` | `10` | Number of simulated clients |
| `options.backend.client-resources.num-cpus` | `4` | CPUs per actor (↑ = fewer concurrent actors = less RAM) |

## KPI Output

Results are written to `resultsfeat/results_ditto-FedYogi-sdp-secAgg-sparse.csv`.

**FIT metrics** (averaged across clients per round):
`fit_time`, `peak_ram_mb`, `comm_size_mb`, `model_size_mb`, `dp_epsilon`, `estimated_energy`

**EVAL metrics** (aggregated per round):
`accuracy` (INT8), `acc_global`, `acc_local_fp32`, `local_vs_global_gap`, `quantization_error`, `accuracy_stddev`, `quantized_model_size_mb`, `eval_time`, `loss`
