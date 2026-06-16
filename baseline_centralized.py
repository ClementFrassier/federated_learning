#!/usr/bin/env python3
"""
Centralized Baseline — Nurse Stress Detection (MLP Tiny)
========================================================
Trains the SAME MLP Tiny model on ALL aggregated data
(without Federated Learning) to serve as a comparative baseline.

Centralized baseline characteristics:
  → Global standardization (sees all clients' data)
  → No privacy constraints (no DP-SGD)

Comparing FL results (resultsfeat/results_nurse_mlp_tiny.csv)
with this baseline helps quantify the cost of privacy and data distribution.

Usage:
    python baseline_centralized.py
"""
import warnings
warnings.filterwarnings("ignore")

import os
import time
import csv
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader, TensorDataset

from task import (
    Net,
    get_model_size,
    test,
    DEVICE,
    FEATURES,
    preprocess_if_needed,
    PROCESSED_DIR,
)

def custom_classification_report(y_true, y_pred, target_names, digits=4):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    
    lines = []
    headers = ["precision", "recall", "f1-score", "support"]
    max_len = max(len(name) for name in target_names)
    
    fmt_header = f"{{:<{max_len + 4}}} " + " ".join([f"{{:>{digits + 6}}}" for _ in headers])
    lines.append(fmt_header.format("", *headers))
    lines.append("")
    
    support_total = len(y_true)
    class_metrics = {}
    
    for c, name in zip(classes, target_names):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support = np.sum(y_true == c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
        
        fmt_row = f"{{:<{max_len + 4}}} " + " ".join([
            f"{{:>{digits + 6}.{digits}f}}",
            f"{{:>{digits + 6}.{digits}f}}",
            f"{{:>{digits + 6}.{digits}f}}",
            f"{{:>{digits + 6}d}}"
        ])
        lines.append(fmt_row.format(name, precision, recall, f1, int(support)))
        
    lines.append("")
    
    accuracy = np.sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0
    fmt_acc = f"{{:<{max_len + 4}}} " + " ".join([
        f"{{:>{digits + 6}}}",
        f"{{:>{digits + 6}}}",
        f"{{:>{digits + 6}.{digits}f}}",
        f"{{:>{digits + 6}d}}"
    ])
    lines.append(fmt_acc.format("accuracy", "", "", accuracy, int(support_total)))
    
    macro_prec = np.mean([class_metrics[c]["precision"] for c in classes])
    macro_rec = np.mean([class_metrics[c]["recall"] for c in classes])
    macro_f1 = np.mean([class_metrics[c]["f1-score"] for c in classes])
    lines.append(fmt_acc.format("macro avg", f"{macro_prec:.{digits}f}", f"{macro_rec:.{digits}f}", macro_f1, int(support_total)))
    
    weighted_prec = np.sum([class_metrics[c]["precision"] * class_metrics[c]["support"] for c in classes]) / support_total
    weighted_rec = np.sum([class_metrics[c]["recall"] * class_metrics[c]["support"] for c in classes]) / support_total
    weighted_f1 = np.sum([class_metrics[c]["f1-score"] * class_metrics[c]["support"] for c in classes]) / support_total
    lines.append(fmt_acc.format("weighted avg", f"{weighted_prec:.{digits}f}", f"{weighted_rec:.{digits}f}", weighted_f1, int(support_total)))
    
    return "\n".join(lines)


# ── Config ────────────────────────────────────────────────────────────────────
CSV_DIR    = "resultsfeat"
CSV_FILE   = os.path.join(CSV_DIR, "results_baseline_centralized.csv")
EPOCHS     = 30
LR         = 0.01
BATCH_SIZE = 64
SEED       = 42
TEST_SPLIT = 0.2


def run_centralized() -> None:
    print("=" * 60)
    print("  CENTRALIZED BASELINE — Nurse Stress Detection")
    print("  Model: MLP Tiny  |  No FL, No DP")
    print("=" * 60)

    # ── Load all data (cached preprocessed windows) ───────────────────────────
    client_ids = preprocess_if_needed()
    print(f"\n[DATA] {len(client_ids)} nurses")

    X_list, y_list = [], []
    for node_id in range(len(client_ids)):
        X, y = torch.load(os.path.join(PROCESSED_DIR, f"client_{node_id}.pt"), weights_only=False)
        X_list.append(X)
        y_list.append(y)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    n_stress     = int(y_all.sum())
    print(
        f"[DATA] {len(X_all):,} windows | "
        f"stress={n_stress} ({100 * n_stress / len(y_all):.1f}%) | "
        f"no-stress={len(y_all) - n_stress} ({100 * (1 - n_stress / len(y_all)):.1f}%)"
    )

    # Global standardization (pure NumPy)
    mean = X_all.mean(axis=0)
    std  = X_all.std(axis=0)
    std[std == 0] = 1.0
    X_all = ((X_all - mean) / std).astype(np.float32)

    # Reproducible train / test split
    rng    = np.random.default_rng(SEED)
    idx    = rng.permutation(len(X_all))
    n_test = int(len(X_all) * TEST_SPLIT)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    def make_loader(indices, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(
                torch.from_numpy(X_all[indices]),
                torch.from_numpy(y_all[indices]),
            ),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
        )

    trainloader = make_loader(train_idx, shuffle=True)
    testloader  = make_loader(test_idx,  shuffle=False)

    # ── Train ─────────────────────────────────────────────────────────────────
    model     = Net().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model_size_kb = get_model_size(model) * 1024
    print(f"\n[TRAIN] MLP Tiny | {model_size_kb:.1f} KB | {EPOCHS} epochs")
    print(f"        SGD lr={LR} | batch={BATCH_SIZE} | seed={SEED}")
    print("-" * 60)

    tracemalloc.start()
    t_start = time.time()
    history  = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(trainloader.dataset)
        _, acc      = test(model, testloader)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "test_acc": acc})
        print(f"  Epoch {epoch:2d}/{EPOCHS}  |  train_loss={epoch_loss:.4f}  |  test_acc={acc:.4f}")

    _, peak_ram  = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_time = time.time() - t_start

    # ── Evaluate FP32 ─────────────────────────────────────────────────────────
    loss_fp32, acc_fp32 = test(model, testloader)

    # ── Evaluate INT8 (TinyML proxy) ──────────────────────────────────────────
    model_cpu = Net().cpu()
    model_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    net_q = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )
    loss_q, acc_q = test(net_q, testloader, device=torch.device("cpu"))

    # ── Classification report (FP32) ──────────────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in testloader:
            preds = model(X_batch.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    print("\n[REPORT] Classification Report (FP32, centralized):")
    print(custom_classification_report(
        all_labels, all_preds,
        target_names=["No Stress", "Stress"],
        digits=4,
    ))

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "mode":               "centralized",
        "epochs":             EPOCHS,
        "lr":                 LR,
        "batch_size":         BATCH_SIZE,
        "n_windows_train":    len(train_idx),
        "n_windows_test":     len(test_idx),
        "train_time_s":       round(train_time, 3),
        "peak_ram_mb":        round(peak_ram / 1024 ** 2, 4),
        "model_size_mb":      round(get_model_size(model), 6),
        "model_size_kb":      round(model_size_kb, 3),
        "acc_fp32":           round(acc_fp32, 4),
        "loss_fp32":          round(loss_fp32, 4),
        "acc_int8":           round(acc_q, 4),
        "loss_int8":          round(loss_q, 4),
        "quantization_error": round(acc_fp32 - acc_q, 4),
    }

    os.makedirs(CSV_DIR, exist_ok=True)
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print("=" * 60)
    print("  CENTRALIZED RESULTS")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:25s}: {v}")
    print(f"\nResults saved → '{CSV_FILE}'")
    print("=" * 60)
    print("\nTo compare with FL results, see:")
    print("  → resultsfeat/results_nurse_mlp_tiny.csv  (FL)")
    print("  → resultsfeat/results_baseline_centralized.csv  (this file)")


if __name__ == "__main__":
    run_centralized()
