#!/usr/bin/env python3
"""
Post-process multiple inverse problem runs in a root folder and assemble visualizations.

Outputs:
- One large plot: Y-error convergence (best individual per sample)
- Three small plots: parameter error convergence (for 3 parameters)
- Shaded bands: Q1–Q3 across samples (best individual per sample)

Requires files per run:
- y_error_full.npy   -> (it, batch, pop, n_outputs)
- param_error_full.npy -> (it, batch, pop, n_params)
- results_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# ============================================================
# Loading
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
            if data is not None:
                runs.append(data)
    return runs

# ============================================================
# Core statistics
# ============================================================


def best_individual_error(err_tensor):
    """
    err_tensor: (it, batch, pop, dim)
    returns: (it, batch, dim)  best individual per sample
    """
    return np.min(err_tensor, axis=2)


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_avg_param_y_error(runs, save_path):
    """Figure 1: 2 subplots — avg param error and Y error with Q1–Q3 shading"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    colors = sns.color_palette("Set1", len(runs))

    for i, run in enumerate(runs):
        # -----------------------------
        # Avg param error
        # -----------------------------
        p_err = run["param_error"]  # (it, batch, pop, n_params)
        if p_err is not None:
            # Step 1: mean over parameter dimension
            mean_over_params = p_err.mean(axis=3)       # (it, batch, pop)
            # Step 2: min over population per sample
            min_over_pop = np.min(mean_over_params, axis=2)  # (it, batch)
            # Step 3: average over batch samples
            avg_best_per_sample = min_over_pop.mean(axis=1)  # (it,)

            # Step 4: compute Q1/Q2/Q3 per iteration across batch
            q1 = np.quantile(min_over_pop, 0.25, axis=1)  # (it,)
            q2 = np.quantile(min_over_pop, 0.50, axis=1)
            q3 = np.quantile(min_over_pop, 0.75, axis=1)

            axes[0].plot(avg_best_per_sample, color=colors[i], label=os.path.basename(run["folder"]))
            axes[0].fill_between(np.arange(len(avg_best_per_sample)), q1, q3, alpha=0.2, color=colors[i])

        # -----------------------------
        # -----------------------------
        # Y error
        # -----------------------------
        y_err = run["y_error"]  # (it, batch, pop, n_outputs)
        if y_err is not None:
            mean_over_outputs = y_err.mean(axis=3)       # (it, batch, pop)
            min_over_pop = np.mean(mean_over_outputs, axis=2)  # (it, batch)
            avg_best_per_sample = min_over_pop.mean(axis=1)
            q1 = np.quantile(min_over_pop, 0.25, axis=1)
            q2 = np.quantile(min_over_pop, 0.50, axis=1)
            q3 = np.quantile(min_over_pop, 0.75, axis=1)
            axes[1].plot(avg_best_per_sample, color=colors[i], label=os.path.basename(run["folder"]))
            axes[1].fill_between(np.arange(len(avg_best_per_sample)), q1, q3, alpha=0.2, color=colors[i])
    axes[0].tick_params(labelsize=18)
    axes[1].tick_params(labelsize=18)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Avg Parameter Error",fontsize=18)
    axes[0].legend(fontsize=18,loc="upper right")

    axes[1].set_yscale("log")
    axes[1].set_ylabel("Avg Y Error",fontsize=18)
    axes[1].set_xlabel("Iteration",fontsize=18)
    plt.suptitle("Inverse Problem Convergence — Avg Parameter and Y Error", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

param_names = [
    "K_i",     # ps[0]
    "K_o",     # ps[1]
    "ATP",     # ps[2]
    "g_Na",    # ps[3]
    "g_CaL",   # ps[4]
    "g_K1",    # ps[5]
    "g_Kr",    # ps[6]
    "g_Ks",    # ps[7]
    "g_to",    # ps[8]
    "g_bca",   # ps[9]
    "g_pk",    # ps[10]
    "g_pca"    # ps[11]
]

def plot_three_params(runs, save_path, param_indices=None):
    """Figure 2: three subplots for selected parameters (best per sample, Q1–Q3)"""
    n_params = runs[0]["param_error"].shape[3]  # total number of params
    param_indices=[7,2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    colors = sns.color_palette("Set1", len(runs))

    for i, run in enumerate(runs):
        p_err = run["param_error"]  # (it, batch, pop, n_params)
        if p_err is None:
            continue
        for j, p_idx in enumerate(param_indices):
            # Take single parameter
            param_err = p_err[:, :, :, p_idx]  # (it, batch, pop)
            min_over_pop = np.mean(param_err, axis=2)  # best per sample (it, batch)
            q1 = np.quantile(min_over_pop, 0.25, axis=1)
            q2 = np.quantile(min_over_pop, 0.50, axis=1)
            q3 = np.quantile(min_over_pop, 0.75, axis=1)
            median_avg = min_over_pop.mean(axis=1)
            axes[j].plot(median_avg, color=colors[i], label=os.path.basename(run["folder"]))
            axes[j].fill_between(np.arange(len(median_avg)), q1, q3, alpha=0.2, color=colors[i])
            axes[j].set_yscale("log")
            axes[j].set_xlabel(f"Param {param_names[p_idx]} Error", fontsize=18)
            #axes[j].grid(True, which="both", linestyle="--", alpha=0.3)
            axes[j].tick_params(labelsize=18)

            axes[j].set_ylim(top=1, bottom=1e-2)  # set a reasonable lower limit for log scale
    axes[0].set_ylabel("Parameter Error",fontsize=22)
    axes[-1].legend(fontsize=18)
    plt.suptitle("Inverse Problem Convergence — Selected Parameter Errors", fontsize=22)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()




# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-process inverse problem runs")
    parser.add_argument("--root_folder", required=True,
                        help="Root folder containing run subfolders")
    args = parser.parse_args()

    runs = aggregate_runs(args.root_folder)
    print(f"Loaded {len(runs)} runs from {args.root_folder}")

    out_dir = args.root_folder
    os.makedirs(out_dir, exist_ok=True)

    plot_avg_param_y_error(runs, save_path=os.path.join(args.root_folder, "avg_param_y_error.png"))

    # Figure 2: 3 random parameters
    plot_three_params(runs, save_path=os.path.join(args.root_folder, "three_param_errors.png"))

