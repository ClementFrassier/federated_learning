import os
import subprocess
import time
from datetime import datetime
import argparse
import shutil
from aggregate_results import aggregate


def log_to_file(log_path, message):
    with open(log_path, "a") as f:
        f.write(message + "\n")


def run_one(cmd_list, log_file, run_name, extra_env=None):
    """
    Run a command given as an argv list (no shell=True, no string quoting
    at all) — avoids the nested-shell escaping issues a string command would
    hit when the outer orchestrator (this script) is itself invoked through
    an extra `wsl -e bash -lc "..."` layer.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    start_time_str = datetime.now().isoformat()
    start_t = time.time()
    log_to_file(log_file, f"--- Run start: {start_time_str} ---\nCommand: {cmd_list}\nEnv: {extra_env}")
    try:
        subprocess.run(
            cmd_list, check=True, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        duration = time.time() - start_t
        print(f"[GRID] Finished {run_name} — duration: {duration:.1f}s")
        log_to_file(log_file, f"Status: SUCCESS\nDuration: {duration:.2f}s\nEnd: {datetime.now().isoformat()}\n")
        return True, None
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_t
        print(f"[GRID] ERROR in {run_name} — duration: {duration:.1f}s")
        print(e.stderr)
        log_to_file(
            log_file,
            f"Status: FAILED\nDuration: {duration:.2f}s\nEnd: {datetime.now().isoformat()}\n"
            f"Error:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}\n",
        )
        return False, run_name


def already_done_fl(out_dir, seed, sigma, expected_rounds):
    """Resume-safety: skip a run whose CSV already reached the expected final
    round (protects against re-running work after an interrupted/crashed grid,
    exactly the scenario that motivated adding this check)."""
    path = os.path.join(out_dir, f"seed{seed}_sigma{sigma:.1f}.csv")
    if not os.path.exists(path):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return not df.empty and df["Round"].max() >= expected_rounds
    except Exception:
        return False


def already_done_cent(out_dir, seed):
    path = os.path.join(out_dir, f"seed{seed}.csv")
    if not os.path.exists(path):
        return False
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return not df.empty
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a single quick pass to verify the pipeline")
    args = parser.parse_args()

    results_dir = "resultsfeat"
    log_file = os.path.join(results_dir, "run_log.txt")
    os.makedirs(results_dir, exist_ok=True)

    if args.test:
        print("[GRID] Running in TEST mode...")
        test_dir = os.path.join(results_dir, "test")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)

        seeds = [999]
        sigmas = [0.8]
        fl_rounds = 2
        fedprox_out_dir = test_dir
        fedper_out_dir = test_dir
        cent_out_dir = test_dir
        summary_out = os.path.join(test_dir, "grid_summary.csv")
    else:
        print("[GRID] Running in PRODUCTION mode...")
        test_dir = None
        seeds = [42, 123, 7, 2024, 31415]
        sigmas = [0.6, 0.8, 1.0, 1.4, 1.8]
        fl_rounds = 30
        fedprox_out_dir = os.path.join(results_dir, "grid_fl")
        fedper_out_dir = os.path.join(results_dir, "ablation_fedper")  # matches server_app.py's actual csv_file routing for personalization-mode=fedper
        cent_out_dir = os.path.join(results_dir, "grid_centralized")
        summary_out = os.path.join(results_dir, "grid_summary.csv")

    os.makedirs(fedprox_out_dir, exist_ok=True)
    os.makedirs(fedper_out_dir, exist_ok=True)
    os.makedirs(cent_out_dir, exist_ok=True)

    fl_runs = [(seed, sigma) for seed in seeds for sigma in sigmas]
    total_fedprox = len(fl_runs)
    total_fedper = len(fl_runs)
    total_cent = len(seeds)
    total_runs = total_fedprox + total_fedper + total_cent

    success_count, fail_count, failures = 0, 0, []
    start_time_grid = time.time()

    # ── 1. FedProx + DP runs ──────────────────────────────────────────────────
    for idx, (seed, sigma) in enumerate(fl_runs, 1):
        run_name = f"FedProx+DP run {idx}/{total_fedprox} — seed={seed}, sigma={sigma}"
        if not test_dir and already_done_fl(fedprox_out_dir, seed, sigma, fl_rounds):
            print(f"[GRID] Skipping {run_name} — already completed (round {fl_rounds} present)")
            success_count += 1
            continue
        print(f"\n[GRID] Starting {run_name}")
        cmd_list = [
            "flwr", "run", ".", "--stream",
            "--run-config", f"noise-multiplier={sigma}",
            "--run-config", f"partition-seed={seed}",
            "--run-config", f"num-server-rounds={fl_rounds}",
            "--run-config", 'personalization-mode="none"',
        ]
        extra_env = {"FLWR_TEST_DIR": test_dir} if test_dir else None
        ok, fail = run_one(cmd_list, log_file, run_name, extra_env)
        success_count += ok
        fail_count += not ok
        if fail:
            failures.append(fail)

    # ── 2. FedPer + DP runs (same 5x5 grid, personalization-mode=fedper) ──────
    for idx, (seed, sigma) in enumerate(fl_runs, 1):
        run_name = f"FedPer+DP run {idx}/{total_fedper} — seed={seed}, sigma={sigma}"
        if not test_dir and already_done_fl(fedper_out_dir, seed, sigma, fl_rounds):
            print(f"[GRID] Skipping {run_name} — already completed (round {fl_rounds} present)")
            success_count += 1
            continue
        print(f"\n[GRID] Starting {run_name}")
        cmd_list = [
            "flwr", "run", ".", "--stream",
            "--run-config", f"noise-multiplier={sigma}",
            "--run-config", f"partition-seed={seed}",
            "--run-config", f"num-server-rounds={fl_rounds}",
            "--run-config", 'personalization-mode="fedper"',
        ]
        extra_env = {"FLWR_TEST_DIR": test_dir} if test_dir else None
        ok, fail = run_one(cmd_list, log_file, run_name, extra_env)
        success_count += ok
        fail_count += not ok
        if fail:
            failures.append(fail)

    # ── 3. Centralized runs ────────────────────────────────────────────────────
    for idx, seed in enumerate(seeds, 1):
        run_name = f"Centralized run {idx}/{total_cent} — seed={seed}"
        if not test_dir and already_done_cent(cent_out_dir, seed):
            print(f"[GRID] Skipping {run_name} — already completed")
            success_count += 1
            continue
        print(f"\n[GRID] Starting {run_name}")
        cmd_list = ["python3", "baseline_centralized.py", "--seed", str(seed)]
        if test_dir:
            cmd_list += ["--test-dir", test_dir]
        ok, fail = run_one(cmd_list, log_file, run_name)
        success_count += ok
        fail_count += not ok
        if fail:
            failures.append(fail)

    # ── 4. Post-processing aggregation ────────────────────────────────────────
    print("\n[GRID] Aggregating FedProx+DP results...")
    aggregate(fl_dir=fedprox_out_dir, cent_dir=cent_out_dir, output_file=summary_out)
    if not args.test:
        print("[GRID] Aggregating FedPer results...")
        aggregate(
            fl_dir=fedper_out_dir, cent_dir=cent_out_dir,
            output_file=os.path.join(results_dir, "grid_summary_fedper.csv"),
        )

    total_duration = time.time() - start_time_grid
    summary_report = (
        f"\n==================================================\n"
        f"WESAD GRID EXPERIMENT COMPLETE\n"
        f"Total runs: {total_runs}\n"
        f"Succeeded: {success_count}\n"
        f"Failed: {fail_count}\n"
        f"Total Wall-Clock Time: {total_duration/60:.2f} minutes ({total_duration:.1f}s)\n"
    )
    if failures:
        summary_report += "Failed runs:\n" + "\n".join(f" - {f}" for f in failures) + "\n"
    summary_report += "==================================================\n"

    print(summary_report)
    log_to_file(log_file, summary_report)

    if args.test:
        print(f"[GRID] Test outputs are in {test_dir}/")
        print("[GRID] Run without --test to run the full production grid.")


if __name__ == "__main__":
    main()
