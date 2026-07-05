import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Compute device (simulation) ───────────────────────────────────────────────
# In a real TinyML scenario, the model runs directly on the smart wristband
# (e.g. Empatica E4, STM32, ESP32). Here we simulate on CPU/GPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset configuration ─────────────────────────────────────────────────────
DATA_PATH   = "data_nurse/merged_data.csv"
FEATURES    = ["EDA", "HR", "TEMP", "X", "Y", "Z"]
WINDOW_SIZE = 60   # samples per sliding window
STEP_SIZE   = 30   # stride (50% overlap)
N_STAT_FEAT = len(FEATURES) * 4  # [mean, std, min, max] × 6 signals = 24 features


# ── Model ─────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """
    MLP Tiny — stress detection from statistical features of physiological signals.

    Real TinyML scenario:
      This model is designed to run DIRECTLY on the smart wristband.
      Raw data (EDA, HR, TEMP) never leaves the device.
      Only the ~5 KB of weights are sent to the FL server for aggregation.

    Architecture:  FC(24 → 32) → ReLU → FC(32 → 16) → ReLU → FC(16 → 2)
    Parameters:    ~1,362
    FP32 Size:     ~5.4 KB   (FL transmission)
    INT8 Size:     ~1.4 KB   (MCU deployment after quantization)

    MCU constraints respected (STM32, ESP32, Cortex-M):
    - No BatchNorm → Native compatibility with Opacus DP-SGD
    - No recurrence → Deterministic, maximum auditability (medical)
    - No convolutions → Runs on any MCU without FPU
    - Input = window-aggregated features → fixed, minimal tensor size
    """

    def __init__(self, input_dim: int = N_STAT_FEAT) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NetMedium(nn.Module):
    """
    MLP Medium — 10x larger model (~13,000 parameters) to isolate model size effects.
    Architecture: FC(24 → 128) → ReLU → FC(128 → 64) → ReLU → FC(64 → 32) → ReLU → FC(32 → 2)
    """

    def __init__(self, input_dim: int = N_STAT_FEAT) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class NetFedRepScaffold(nn.Module):
    """
    Dedicated architecture for the "best architecture ignoring TinyML, security
    constraints only" experiment: same-size shared extractor as Net (fc1/fc2,
    aggregated globally, DP-SGD applied here only), multi-layer local head
    (never transmitted, no DP noise) instead of FedPer's single linear layer.

    Deliberately NOT reusing Net's layer names (fc1/fc2/fc3) — this is a
    self-contained architecture with its own extractor/head submodules, so DP
    and SCAFFOLD can target the extractor specifically. Not comparable to Net
    for TinyML deployment purposes (see report caveat).
    """

    def __init__(self, input_dim: int = N_STAT_FEAT, head_hidden: int = 16) -> None:
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(16, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extractor(x))


# ── Caching system to prevent Ray OOM (Out Of Memory) ─────────────────────────
PROCESSED_DIR = "data_nurse/processed"

def preprocess_if_needed() -> list:
    """
    Pre-extract sliding window features for all 15 clients once.
    Saves results to data_nurse/processed/client_{node_id}.pt.
    This saves ~95% RAM during simulation.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    client_ids_path = os.path.join(PROCESSED_DIR, "client_ids.pt")

    # If client_ids.pt exists and all client files exist, load and return
    if os.path.exists(client_ids_path):
        try:
            client_ids = torch.load(client_ids_path, weights_only=False)
            all_exist = True
            for i in range(len(client_ids)):
                if not os.path.exists(os.path.join(PROCESSED_DIR, f"client_{i}.pt")):
                    all_exist = False
                    break
            if all_exist:
                return client_ids
        except Exception:
            pass  # re-run preprocessing if loading fails

    # Perform one-time preprocessing
    print(f"[PREPROCESS] Preprocessing '{DATA_PATH}' to extract features...")
    df = pd.read_csv(
        DATA_PATH,
        usecols=["EDA", "HR", "TEMP", "X", "Y", "Z", "id", "label"],
    )
    df["id"] = df["id"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1, 2])].reset_index(drop=True)
    # Binary recombination: raw label 0 = no-stress (18.8%), 1 = low-stress (7.0%),
    # 2 = high-stress (74.2%) — see Abu-Samah et al. 2025. Low + high stress are
    # merged into a single "stress" class (1) so the task uses the full dataset
    # instead of discarding the high-stress majority class.
    df["label"] = (df["label"] > 0).astype(int)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)

    client_ids = sorted(df["id"].unique().tolist())
    torch.save(client_ids, client_ids_path)

    for node_id, nurse_id in enumerate(client_ids):
        print(f"[PREPROCESS] Extracting windows for client {node_id} (nurse_id={nurse_id})...")
        df_client = df[df["id"] == nurse_id].copy()
        X, y = _extract_windows(df_client)
        torch.save((X, y), os.path.join(PROCESSED_DIR, f"client_{node_id}.pt"))

    print("[PREPROCESS] Preprocessing completed! Cache saved.")
    return client_ids


def _ensure_global_data_loaded() -> tuple[pd.DataFrame, list]:
    """Fallback / backward compatibility for centralized baseline."""
    global _global_df, _client_ids
    if _global_df is None:
        print(f"[DATA] Loading nurse dataset from '{DATA_PATH}' (one-time, ~30 s)...")
        df = pd.read_csv(
            DATA_PATH,
            usecols=["EDA", "HR", "TEMP", "X", "Y", "Z", "id", "label"],
        )
        df["id"] = df["id"].astype(str)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df[df["label"].isin([0, 1, 2])].reset_index(drop=True)
        # Binary recombination: see preprocess_if_needed() for details.
        df["label"] = (df["label"] > 0).astype(int)
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
        _global_df = df
        _client_ids = sorted(df["id"].unique().tolist())
    return _global_df, _client_ids


def get_client_ids() -> list:
    """Return sorted list of unique nurse IDs present in the dataset."""
    return preprocess_if_needed()


def _extract_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window feature extraction over a single nurse's time series.
    """
    data   = df[FEATURES].values.astype(np.float32)
    labels = df["label"].values.astype(np.int64)

    X_list, y_list = [], []
    for start in range(0, len(data) - WINDOW_SIZE + 1, STEP_SIZE):
        window = data[start : start + WINDOW_SIZE]
        feats  = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
        ]).astype(np.float32)
        label = int(np.bincount(labels[start : start + WINDOW_SIZE]).argmax())
        X_list.append(feats)
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(
    node_id: int,
    batch_size: int = 64,
    test_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Loads data for an FL device (simulated smart wristband).
    Loads from preprocessed file to save memory and avoid OOM.

    SPLIT STRATEGY — Temporal (chronological), NOT random permutation.
    ─────────────────────────────────────────────────────────────────
    Sliding windows with 50% overlap (WINDOW_SIZE=60, STEP_SIZE=30) share
    30 raw samples between consecutive windows. A random permutation split
    places window i in train and window i+1 in test, leaking 50% of raw
    test samples into training → inflated accuracy (~94% contamination rate
    measured empirically on this dataset, see check_leakage.py).

    The temporal split eliminates this:
      - Train : first (1 − test_split) fraction of windows in chronological order
      - Gap   : 2 windows excluded (= WINDOW_SIZE raw samples, no overlap possible)
      - Test  : remaining last windows in chronological order

    The `seed` argument is kept for API compatibility but no longer controls
    the split — results are fully deterministic across seeds.
    """
    client_ids = preprocess_if_needed()
    if node_id >= len(client_ids):
        raise ValueError(
            f"node_id={node_id} out of range — dataset only has {len(client_ids)} clients."
        )

    # Load from cached preprocessed file
    X, y = torch.load(os.path.join(PROCESSED_DIR, f"client_{node_id}.pt"), weights_only=False)
    n_total = len(X)

    # ── Temporal split with gap (zero leakage guaranteed) ─────────────────────
    gap_windows = max(2, WINDOW_SIZE // STEP_SIZE)   # = 2 windows = 60 raw samples
    n_train     = max(1, int(n_total * (1.0 - test_split)))
    n_test      = max(1, n_total - n_train - gap_windows)

    train_idx = np.arange(n_train)
    test_idx  = np.arange(n_train + gap_windows, n_train + gap_windows + n_test)

    # Local standardization on the device (pure NumPy) — fit on TRAIN only,
    # then applied to both train and test, to avoid target leakage into the
    # standardization statistics (mirrors baseline_centralized.py).
    mean = X[train_idx].mean(axis=0)
    std  = X[train_idx].std(axis=0)
    std[std == 0] = 1.0
    X = ((X - mean) / std).astype(np.float32)

    def _make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        X_t = torch.from_numpy(X[indices])
        y_t = torch.from_numpy(y[indices])
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

    return _make_loader(train_idx, shuffle=True), _make_loader(test_idx, shuffle=False)


def load_model(model_type: str = "tiny") -> nn.Module:
    """Instantiate a fresh model (Net or NetMedium) and move it to the available device."""
    if model_type == "medium":
        return NetMedium().to(DEVICE)
    return Net().to(DEVICE)


def get_model_size(model: nn.Module) -> float:
    """Return the model size in megabytes (parameters + buffers)."""
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


# ── Global class weights (for the weighted-loss ablation) ─────────────────────
_global_class_weights = None


def get_global_class_weights() -> torch.Tensor:
    """
    Inverse-frequency class weights (0=no-stress, 1=stress), computed once from
    the pooled TRAIN partitions of all clients' cached windows and cached for
    reuse. Used only by the weighted-loss ablation (loss-type=weighted_ce).
    """
    global _global_class_weights
    if _global_class_weights is None:
        client_ids = preprocess_if_needed()
        counts = np.zeros(2, dtype=np.int64)
        for node_id in range(len(client_ids)):
            X, y = torch.load(os.path.join(PROCESSED_DIR, f"client_{node_id}.pt"), weights_only=False)
            n_train = max(1, int(len(X) * 0.8))
            y_train = y[:n_train]
            for c in (0, 1):
                counts[c] += int((y_train == c).sum())
        weights = counts.sum() / (2 * counts)
        _global_class_weights = torch.tensor(weights, dtype=torch.float32)
    return _global_class_weights


# ── Total train-set size across all clients (for SCAFFOLD's c_global rescaling) ─
_total_train_examples = None


def get_total_train_examples() -> int:
    """
    Sum of local TRAIN-partition sizes across all clients, computed once and
    cached. Same pattern and same transparency note as get_global_class_weights():
    an aggregate, non-sensitive statistic (dataset size, not content), used only
    by the FedRep+SCAFFOLD experiment (personalization-mode=fedrep_scaffold) to
    rescale each client's control-variate update before transmission, so that
    Flower's native num-examples-weighted aggregation reconstructs SCAFFOLD's
    canonical unweighted average of c_i across clients instead of a
    size-weighted one (see report for the reasoning).
    """
    global _total_train_examples
    if _total_train_examples is None:
        client_ids = preprocess_if_needed()
        total = 0
        for node_id in range(len(client_ids)):
            X, _ = torch.load(os.path.join(PROCESSED_DIR, f"client_{node_id}.pt"), weights_only=False)
            total += max(1, int(len(X) * 0.8))
        _total_train_examples = total
    return _total_train_examples


# ── FedProx + DP-SGD training ─────────────────────────────────────────────────
def train_fedprox_dp(
    net_dp,
    optimizer_dp,
    trainloader_dp,
    global_params_on_device: dict,
    epochs: int,
    mu: float = 0.05,
    class_weights: torch.Tensor | None = None,
) -> None:
    """
    Local training on the device — one FL round (FedProx + DP-SGD).

    In a real TinyML scenario, this function executes directly
    on the smart wristband. Raw data remains on the device.
    Only net_dp._module (weights, ~5 KB) will be returned to the server.

    IMPORTANT — The PrivacyEngine is NOT created here.
    It is created once per client in client_app.py and kept alive across ALL
    rounds so that epsilon accumulates correctly over the full federation.

    Args:
        net_dp:                  Opacus-wrapped model (persistent across rounds).
        optimizer_dp:            Opacus-wrapped SGD optimiser (persistent).
        trainloader_dp:          Opacus-wrapped DataLoader (persistent).
        global_params_on_device: Server weight snapshot used for FedProx penalty.
        epochs:                  Number of local training epochs per round.
        mu:                      FedProx proximal penalty weight µ.
        class_weights:           Optional per-class loss weights (weighted-loss ablation).
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if class_weights is not None else None
    )
    net_dp.train()

    for _ in range(epochs):
        for X_batch, y_batch in trainloader_dp:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer_dp.zero_grad()

            loss = criterion(net_dp(X_batch), y_batch)

            # FedProx proximal term: µ/2 · ||w − w_global||²
            # MLP has no GN layers → ALL parameters are included
            proximal_term = sum(
                torch.sum(
                    (param - global_params_on_device[name.replace("_module.", "")]) ** 2
                )
                for name, param in net_dp.named_parameters()
                if name.replace("_module.", "") in global_params_on_device
            )
            (loss + (mu / 2.0) * proximal_term).backward()
            optimizer_dp.step()


# ── Evaluation ────────────────────────────────────────────────────────────────
def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Evaluate model on the test set → (avg_loss, accuracy)."""
    if device is None:
        device = DEVICE
    criterion = nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0

    net.eval()
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs     = net(X_batch)
            total_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
            correct    += (outputs.argmax(1) == y_batch).sum().item()

    n = len(testloader.dataset)
    return total_loss / n, correct / n
