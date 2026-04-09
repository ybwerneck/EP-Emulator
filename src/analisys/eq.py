#!/usr/bin/env python3
"""
Post-process inverse problem runs and simulate early stopping.

Stopping rule:
- Stagnation detection (patience + tolerance)

Metrics reported at the stopping iteration:
- Parameter error
- Y error

Outputs:
- early_stop_metrics.csv
- inverse_summary_table.csv
- success_curves.png
- convergence_histograms.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob


# ============================================================
# Load runs
# ============================================================

def load_run_data(folder):

    summary_path = os.path.join(folder, "results_summary.json")
    yerr_path = os.path.join(folder, "y_error_full.npy")
    perr_path = os.path.join(folder, "param_error_full.npy")

    if not os.path.exists(summary_path):
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    y_err = np.load(yerr_path) if os.path.exists(yerr_path) else None
    p_err = np.load(perr_path) if os.path.exists(perr_path) else None

    return {
        "folder": folder,
        "summary": summary,
        "y_error": y_err,
        "param_error": p_err
    }


def aggregate_runs(root_folder):

    runs = []

    for folder in glob(os.path.join(root_folder, "*")):

        if os.path.isdir(folder):

            data = load_run_data(folder)

            if data is not None and data["y_error"] is not None:
                runs.append(data)

    return runs


# ============================================================
# Early stopping logic
# ============================================================

def compute_stop_iterations(y_err, patience=10, tol=1e-5):

    err = y_err.mean(axis=3)      # (it, batch, pop)
    best = np.min(err, axis=2)    # (it, batch)

    iters, batch = best.shape

    stop_iter = np.zeros(batch, dtype=int)

    for b in range(batch):

        best_so_far = best[0, b]
        stagnation = 0

        for i in range(1, iters):

            if best[i, b] < best_so_far - tol:
                best_so_far = best[i, b]
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= patience:
                stop_iter[b] = i
                break
        else:
            stop_iter[b] = iters

    return stop_iter, best


# ============================================================
# Parameter error at stopping iteration
# ============================================================

def get_param_error_at_stop(run, stop_iter):

    y_err = run["y_error"]
    p_err = run["param_error"]

    if p_err is None:
        return None

    err = y_err.mean(axis=3)
    best_idx = np.argmin(err, axis=2)

    batch = best_idx.shape[1]
    n_params = p_err.shape[3]

    param_errors = np.zeros((batch, n_params))

    for b in range(batch):

        it = stop_iter[b] - 1
        if it < 0:
            it = 0

        p = best_idx[it, b]

        param_errors[b] = p_err[it, b, p]

    return param_errors


# ============================================================
# Y error at stopping iteration
# ============================================================

def get_y_error_at_stop(run, stop_iter):

    y_err = run["y_error"]

    err = y_err.mean(axis=3)
    best_err = np.min(err, axis=2)

    batch = best_err.shape[1]

    y_errors = np.zeros(batch)

    for b in range(batch):

        it = stop_iter[b] - 1
        if it < 0:
            it = 0

        y_errors[b] = best_err[it, b]

    return y_errors


# ============================================================
# Metrics
# ============================================================

def compute_metrics(runs):

    rows = []
    stop_iters = {}

    for run in runs:

        name = os.path.basename(run["folder"])
        y_err = run["y_error"]

        stop_iter, best = compute_stop_iterations(y_err)

        rows.append({
            "emulator": name,
            "mean_iter": np.mean(stop_iter),
            "median_iter": np.median(stop_iter),
            "std_iter": np.std(stop_iter),
            "min_iter": np.min(stop_iter),
            "max_iter": np.max(stop_iter),
        })

        stop_iters[name] = stop_iter

    df = pd.DataFrame(rows)

    return df, stop_iters


# ============================================================
# Plots
# ============================================================

def plot_histograms(stop_iters, save_path):

    plt.figure(figsize=(8,6))

    for name, iters in stop_iters.items():

        plt.hist(
            iters,
            bins=30,
            alpha=0.5,
            label=name
        )

    plt.xlabel("Iterations to convergence")
    plt.ylabel("Number of samples")
    plt.title("Distribution of Convergence Iterations")
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Early stopping analysis")
    parser.add_argument("--root_folder", required=True)

    args = parser.parse_args()

    print("Loading runs...")
    runs = aggregate_runs(args.root_folder)

    print(f"{len(runs)} runs loaded")

    df, stop_iters = compute_metrics(runs)

    print("\nEarly stopping metrics:\n")
    print(df)

    df.to_csv(
        os.path.join(args.root_folder, "early_stop_metrics.csv"),
        index=False
    )

    plot_histograms(
        stop_iters,
        os.path.join(args.root_folder, "convergence_histograms.png")
    )


    # ============================================================
    # Final summary table
    # ============================================================

    summary_rows = []

    for run in runs:

        name = os.path.basename(run["folder"])

        y_err = run["y_error"]

        stop_iter, _ = compute_stop_iterations(y_err)

        total_iters = y_err.shape[0]

        # parameter error
        param_err = get_param_error_at_stop(run, stop_iter)

        if param_err is not None:
            param_mean = param_err.mean()
            param_std = param_err.std()
        else:
            param_mean = np.nan
            param_std = np.nan

        # Y error
        y_err_stop = get_y_error_at_stop(run, stop_iter)

        y_mean = y_err_stop.mean()
        y_std = y_err_stop.std()

        summary = run["summary"]

        if "mean_iter_time" in summary:
            time_per_iter = summary["mean_iter_time"]
        else:
            time_per_iter = np.nan

        summary_rows.append({

            "emulator": name,

            "iter_mean": np.mean(stop_iter),
            "iter_std": np.std(stop_iter),
            "iter_min": np.min(stop_iter),
            "iter_max": np.max(stop_iter),

            "param_error_mean": param_mean,
            "param_error_std": param_std,

            "y_error_mean": y_mean,
            "y_error_std": y_std,

            "time_per_iteration": time_per_iter
        })

    df_summary = pd.DataFrame(summary_rows)

    df_summary.to_csv(
        os.path.join(args.root_folder, "inverse_summary_table.csv"),
        index=False
    )

    print("\nFinal summary table:\n")
    print(df_summary)
