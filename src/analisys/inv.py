#!/usr/bin/env python3
"""
Post-process multiple inverse problem runs in a root folder and assemble visualizations.

- Reads multiple run subfolders under a root folder
- Aggregates losses, runtime, parameter errors
- Produces convergence plots, parameter error plots
- Saves summary CSV and plots
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def load_run_data(folder):
    """Load summary and parameter error history from one folder."""
    summary_path = os.path.join(folder, "results_summary.json")
    history_path = os.path.join(folder, "history_param_error.npy")

    if not os.path.exists(summary_path):
        print(f"Missing summary.json in {folder}")
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    history_param_error = None
    if os.path.exists(history_path):
        history_param_error = np.load(history_path, allow_pickle=True)

    return {
        "folder": folder,
        "summary": summary,
        "history_param_error": history_param_error
    }

def aggregate_runs(root_folder):
    """Load all runs from subfolders of the root folder."""
    runs = []
    for folder in glob(os.path.join(root_folder, "*")):
        if os.path.isdir(folder):
            data = load_run_data(folder)
            if data is not None:
                runs.append(data)
    return runs

def plot_convergence(runs, save_path):
    plt.figure(figsize=(8,6))
    for run in runs:
        hist = run["history_param_error"]
        if hist is not None:
            avg_history = np.mean(hist, axis=1)
            plt.plot(avg_history, alpha=0.7, label=os.path.basename(run["folder"]))
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Parameter Error")
    plt.title("Convergence Across Runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_final_param_error(runs, save_path):
    final_errors = [np.mean(run["history_param_error"][-1]) for run in runs if run["history_param_error"] is not None]
    labels = [os.path.basename(run["folder"]) for run in runs if run["history_param_error"] is not None]

    plt.figure(figsize=(8,6))
    plt.boxplot(final_errors, labels=labels)
    plt.ylabel("Mean Parameter Error (final)")
    plt.title("Final Parameter Error Across Runs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def save_summary_csv(runs, out_csv):
    rows = []
    for run in runs:
        summary = run["summary"]
        row = {
            "folder": os.path.basename(run["folder"]),
            "total_runtime_sec": summary.get("total_runtime_sec", np.nan),
            "final_best_loss": summary.get("final_best_loss", np.nan),
            "final_median_loss": summary.get("final_median_loss", np.nan),
            "mean_iter_time": summary.get("mean_iter_time", np.nan)
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregated metrics to {out_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process multiple inverse problem runs from a root folder")
    parser.add_argument("--root_folder", required=True,
                        help="Root folder containing run subfolders")
    args = parser.parse_args()
    args.out_dir=args.root_folder
    
    os.makedirs(args.out_dir, exist_ok=True)
    runs = aggregate_runs(args.root_folder)
    print(f"Loaded {len(runs)} runs from {args.root_folder}")

    plot_convergence(runs, save_path=os.path.join(args.out_dir, "convergence_summary.png"))
    plot_final_param_error(runs, save_path=os.path.join(args.out_dir, "final_param_error.png"))
    save_summary_csv(runs, out_csv=os.path.join(args.out_dir, "aggregated_results.csv"))
