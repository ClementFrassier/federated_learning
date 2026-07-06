import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Compute device (simulation) ───────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset configuration ─────────────────────────────────────────────────────
# WESAD wrist-only (Empatica E4) signals — the realistic wearable-deployment
# subset, unlike the chest RespiBAN device which is a research-grade sensor,
# not something a TinyML wristband would carry. All 4 wrist channels are
# downsampled/upsampled to a common 4 Hz timeline (EDA/TEMP's native rate)
# so a single sliding window can be taken over all of them together, exactly
# mirroring the nurse pipeline's structure (6 signals x 4 stats = 24 features,
# same Net architecture, no code change needed there).
DATA_ROOT   = "../data_wesad/WESAD"
SUBJECTS    = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
               "S11", "S13", "S14", "S15", "S16", "S17"]
FEATURES    = ["ACC_x", "ACC_y", "ACC_z", "BVP", "EDA", "TEMP"]
FS_TARGET   = 4.0     # Hz, common rate after resampling (EDA/TEMP native rate)
FS_ACC_RAW  = 32.0
FS_BVP_RAW  = 64.0
FS_LABEL_RAW = 700.0  # chest RespiBAN label rate, synchronised at t=0 with wrist
WINDOW_SIZE = 60      # samples per sliding window (60 @ 4Hz = 15s)
STEP_SIZE   = 30       # stride, 50% overlap (30 @ 4Hz = 7.5s)
N_STAT_FEAT = len(FEATURES) * 4  # [mean, std, min, max] x 6 signals = 24 features

# WESAD protocol labels: 0=transient/undefined, 1=baseline, 2=stress,
# 3=amusement, 4=meditation, 5/6/7=should be ignored (see wesad_readme.pdf).
USABLE_LABELS = {1, 2, 3, 4}
STRESS_LABEL  = 2


# ── Model ─────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    """
    MLP Tiny — stress detection from statistical features of wrist-worn
    physiological signals (Empatica E4: ACC, BVP, EDA, TEMP).

    Identical architecture to the nurse-dataset pipeline (same 24-feature
    input, same layer sizes) — chosen for direct comparability across the
    two case studies, not re-tuned for WESAD specifically.

    Architecture:  FC(24 -> 32) -> ReLU -> FC(32 -> 16) -> ReLU -> FC(16 -> 2)
    Parameters:    ~1,362
    FP32 Size:     ~5.4 KB
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
    """MLP Medium — 10x larger model (~13,000 parameters), kept for API parity
    with the nurse pipeline's model-size ablation (not run by default here)."""

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
    """Kept for API parity with the nurse pipeline's client_app.py import list.
    Not exercised by the WESAD grid (FedProx+DP and FedPer only, per scope)."""

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
PROCESSED_DIR = "data_wesad_processed"


def _block_mean(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Downsample by averaging non-overlapping blocks of `block_size` samples."""
    n = (len(arr) // block_size) * block_size
    arr = arr[:n]
    return arr.reshape(-1, block_size, *arr.shape[1:]).mean(axis=1)


def _block_mode(arr: np.ndarray, block_size: int) -> np.ndarray:
    """Downsample a label array by majority vote over non-overlapping blocks."""
    n = (len(arr) // block_size) * block_size
    arr = arr[:n]
    blocks = arr.reshape(-1, block_size)
    out = np.empty(blocks.shape[0], dtype=arr.dtype)
    for i, b in enumerate(blocks):
        vals, counts = np.unique(b, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def _load_subject_signals(subject: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one WESAD subject's wrist signals, resample everything to a common
    4 Hz timeline, and return (signals, binary_label) restricted to the
    usable protocol labels (1=baseline, 2=stress, 3=amusement, 4=meditation).
    """
    pkl_path = os.path.join(DATA_ROOT, subject, f"{subject}.pkl")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    wrist = d["signal"]["wrist"]
    acc   = np.asarray(wrist["ACC"],  dtype=np.float64)          # (N,3) @ 32Hz
    bvp   = np.asarray(wrist["BVP"],  dtype=np.float64).reshape(-1, 1)  # @64Hz
    eda   = np.asarray(wrist["EDA"],  dtype=np.float64).reshape(-1)     # @4Hz
    temp  = np.asarray(wrist["TEMP"], dtype=np.float64).reshape(-1)     # @4Hz
    label = np.asarray(d["label"]).reshape(-1)                          # @700Hz

    acc_ds   = _block_mean(acc, int(FS_ACC_RAW / FS_TARGET))            # 32/4=8
    bvp_ds   = _block_mean(bvp, int(FS_BVP_RAW / FS_TARGET)).reshape(-1)  # 64/4=16
    label_ds = _block_mode(label, int(FS_LABEL_RAW / FS_TARGET))       # 700/4=175

    n = min(len(acc_ds), len(bvp_ds), len(eda), len(temp), len(label_ds))
    signals = np.column_stack([
        acc_ds[:n], bvp_ds[:n], eda[:n], temp[:n],
    ]).astype(np.float32)
    labels_4hz = label_ds[:n]

    # Restrict to usable protocol labels BEFORE windowing — mirrors the nurse
    # pipeline's df[df.label.isin(...)] filter, accepting that windows may
    # span originally non-contiguous time after dropping transient/ignore
    # samples (same simplification, same reasoning as task.py).
    mask = np.isin(labels_4hz, list(USABLE_LABELS))
    signals = signals[mask]
    labels_bin = (labels_4hz[mask] == STRESS_LABEL).astype(np.int64)

    return signals, labels_bin


def _extract_windows(signals: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window feature extraction, identical scheme to the nurse pipeline."""
    X_list, y_list = [], []
    for start in range(0, len(signals) - WINDOW_SIZE + 1, STEP_SIZE):
        window = signals[start : start + WINDOW_SIZE]
        feats = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            window.min(axis=0),
            window.max(axis=0),
        ]).astype(np.float32)
        lab_window = labels[start : start + WINDOW_SIZE]
        label = int(np.bincount(lab_window).argmax())
        X_list.append(feats)
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def preprocess_if_needed() -> list:
    """
    Pre-extract sliding window features for all 15 WESAD subjects once.
    Saves results to data_wesad_processed/client_{node_id}.pt.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    client_ids_path = os.path.join(PROCESSED_DIR, "client_ids.pt")

    if os.path.exists(client_ids_path):
        try:
            client_ids = torch.load(client_ids_path, weights_only=False)
            all_exist = all(
                os.path.exists(os.path.join(PROCESSED_DIR, f"client_{i}.pt"))
                for i in range(len(client_ids))
            )
            if all_exist:
                return client_ids
        except Exception:
            pass

    print(f"[PREPROCESS] Preprocessing WESAD wrist signals from '{DATA_ROOT}'...")
    client_ids = list(SUBJECTS)
    torch.save(client_ids, client_ids_path)

    for node_id, subject in enumerate(client_ids):
        print(f"[PREPROCESS] Extracting windows for client {node_id} (subject={subject})...")
        signals, labels = _load_subject_signals(subject)
        X, y = _extract_windows(signals, labels)
        torch.save((X, y), os.path.join(PROCESSED_DIR, f"client_{node_id}.pt"))

    print("[PREPROCESS] Preprocessing completed! Cache saved.")
    return client_ids


def get_client_ids() -> list:
    """Return the list of WESAD subject IDs (in client-index order)."""
    return preprocess_if_needed()


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(
    node_id: int,
    batch_size: int = 64,
    test_split: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Loads data for an FL device (simulated smart wristband). Same temporal
    split + anti-leakage gap + local standardization strategy as the nurse
    pipeline's load_data() — see task.py there for the full rationale.
    """
    client_ids = preprocess_if_needed()
    if node_id >= len(client_ids):
        raise ValueError(
            f"node_id={node_id} out of range — dataset only has {len(client_ids)} clients."
        )

    X, y = torch.load(os.path.join(PROCESSED_DIR, f"client_{node_id}.pt"), weights_only=False)
    n_total = len(X)

    gap_windows = max(2, WINDOW_SIZE // STEP_SIZE)
    n_train     = max(1, int(n_total * (1.0 - test_split)))
    n_test      = max(1, n_total - n_train - gap_windows)

    train_idx = np.arange(n_train)
    test_idx  = np.arange(n_train + gap_windows, n_train + gap_windows + n_test)

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
    if model_type == "medium":
        return NetMedium().to(DEVICE)
    return Net().to(DEVICE)


def get_model_size(model: nn.Module) -> float:
    param_size  = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2


# ── Global class weights (kept for API parity — not used by the FedProx+DP /
# FedPer grid, only by the nurse pipeline's weighted-loss ablation) ───────────
_global_class_weights = None


def get_global_class_weights() -> torch.Tensor:
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


_total_train_examples = None


def get_total_train_examples() -> int:
    """Kept for API parity with client_app.py's FedRep+SCAFFOLD import (unused
    by the FedProx+DP/FedPer grid actually run for WESAD)."""
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
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if class_weights is not None else None
    )
    net_dp.train()

    for _ in range(epochs):
        for X_batch, y_batch in trainloader_dp:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer_dp.zero_grad()

            loss = criterion(net_dp(X_batch), y_batch)

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
