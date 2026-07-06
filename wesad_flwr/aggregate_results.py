import os
import csv
import re
import pandas as pd
import numpy as np

def aggregate(fl_dir=None, cent_dir=None, output_file=None):
    if fl_dir is None:
        fl_dir = os.path.join("resultsfeat", "grid_fl")
    if cent_dir is None:
        cent_dir = os.path.join("resultsfeat", "grid_centralized")
    if output_file is None:
        output_file = os.path.join("resultsfeat", "grid_summary.csv")

    rows = []

    # 1. Parse FL files
    if os.path.exists(fl_dir):
        for filename in sorted(os.listdir(fl_dir)):
            if filename.endswith(".csv") and filename.startswith("seed"):
                match = re.match(r"seed(\d+)_sigma([\d.]+)\.csv", filename)
                if not match:
                    continue
                seed = int(match.group(1))
                sigma = float(match.group(2))

                filepath = os.path.join(fl_dir, filename)
                
                try:
                    df = pd.read_csv(filepath)
                    if df.empty:
                        continue
                    
                    last_round = df["Round"].max()
                    df_round = df[df["Round"] == last_round]
                    
                    metrics = {}
                    for _, row in df_round.iterrows():
                        phase = row["Phase"]
                        metric = row["Metric"]
                        val = row["Value"]
                        metrics[f"{phase}_{metric}"] = val
                    
                    rows.append({
                        "mode": "federated",
                        "seed": seed,
                        "sigma": sigma,
                        "round": last_round,
                        "accuracy_int8": metrics.get("EVAL_accuracy", np.nan),
                        "accuracy_fp32_local": metrics.get("EVAL_acc_local_fp32", np.nan),
                        "accuracy_global": metrics.get("EVAL_acc_global", np.nan),
                        "local_vs_global_gap": metrics.get("EVAL_local_vs_global_gap", np.nan),
                        "dp_epsilon": metrics.get("FIT_dp_epsilon", np.nan),
                        "quantization_error": metrics.get("EVAL_quantization_error", np.nan),
                        "fit_time_s": metrics.get("FIT_fit_time", np.nan),
                        "peak_ram_mb": metrics.get("FIT_peak_ram_mb", np.nan),
                        "comm_size_mb": metrics.get("FIT_comm_size_mb", np.nan),
                        "accuracy_fp32": np.nan,
                        "train_time_s": np.nan,
                    })
                except Exception as e:
                    print(f"Error parsing FL file {filename}: {e}")

    # 2. Parse Centralized files
    if os.path.exists(cent_dir):
        for filename in sorted(os.listdir(cent_dir)):
            if filename.endswith(".csv") and filename.startswith("seed"):
                match = re.match(r"seed(\d+)\.csv", filename)
                if not match:
                    continue
                seed = int(match.group(1))

                filepath = os.path.join(cent_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    if df.empty:
                        continue
                    
                    row_data = df.iloc[0]
                    
                    rows.append({
                        "mode": "centralized",
                        "seed": seed,
                        "sigma": np.nan,
                        "round": np.nan,
                        "accuracy_int8": row_data.get("acc_int8", np.nan),
                        "accuracy_fp32_local": np.nan,
                        "accuracy_global": np.nan,
                        "local_vs_global_gap": np.nan,
                        "dp_epsilon": np.nan,
                        "quantization_error": row_data.get("quantization_error", np.nan),
                        "fit_time_s": np.nan,
                        "peak_ram_mb": row_data.get("peak_ram_mb", np.nan),
                        "comm_size_mb": np.nan,
                        "accuracy_fp32": row_data.get("acc_fp32", np.nan),
                        "train_time_s": row_data.get("train_time_s", np.nan),
                    })
                except Exception as e:
                    print(f"Error parsing centralized file {filename}: {e}")

    if rows:
        columns = [
            "mode", "seed", "sigma", "round", 
            "accuracy_int8", "accuracy_fp32_local", "accuracy_global", "local_vs_global_gap", "dp_epsilon", "quantization_error", "fit_time_s", "peak_ram_mb", "comm_size_mb",
            "accuracy_fp32", "train_time_s"
        ]
        df_out = pd.DataFrame(rows, columns=columns)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_out.to_csv(output_file, index=False)
        print(f"Aggregated summary written to '{output_file}' with {len(rows)} rows.")
    else:
        print("No result files found to aggregate.")

if __name__ == "__main__":
    aggregate()
