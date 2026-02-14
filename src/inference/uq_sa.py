#!/usr/bin/env python3
"""
Surrogate UQ and SA evaluation against true-model reference statistics.

This script:
- Loads a true-model definition (for distribution only)
- Loads true-model UQ and Sobol statistics from CSV
- Evaluates all surrogates in a folder
- Computes UQ + SA using the same protocol as the true model
- Compares surrogate statistics against true statistics
- Writes scalar comparison metrics + timing to CSV
- Measures total execution time
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import sys

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
import chaospy as cp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.EP.ModelC import TTCellModelFull as modelB
from src.EP.ModelA import TTCellModelExt as modelA
from src.MEC.ModelsCD import TisoModel as modelC
from src.MEC.ModelsCD import Ho8Model as modelD
from src.EP.wrapper import FullModelWrapper
from src.surrogate_models.gaussian_process import *
from src.surrogate_models.DD_Models import ModelInterface as Surrogate

# ---------------------------------------------------------------------
# MODEL REGISTRY (distribution only)
# ---------------------------------------------------------------------
def load_model(model_key):
    MODEL_REGISTRY = {"a": modelA, "b": modelB, "h": modelC, "t": modelD}
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Valid options: {list(MODEL_REGISTRY.keys())}")
    if model_key in ["a", "b"]:
        return FullModelWrapper(MODEL_REGISTRY[model_key])
    else:
        return MODEL_REGISTRY[model_key]()

# ---------------------------------------------------------------------
# UQ
# ---------------------------------------------------------------------
def compute_uq(model_fn, dist, n_samples):
    X = dist.sample(n_samples).T
    Y = model_fn(X)
    if Y.ndim == 1:
        Y = Y[:, None]
    return {
        "mean": np.mean(Y, axis=0),
        "std": np.std(Y, axis=0),

    }


def compute_uq_from_pce(surrogate,dist):
    poly = surrogate.model
    mean = cp.E(poly, dist)
    std = cp.Std(poly, dist)
    print(mean)
    mean,std=surrogate.denormalize_y(mean),surrogate.denormalize_y(std)
    print(mean)
    return {"mean": mean, "std": std}

def compare_uq(uq_emul, uq_true, eps=1e-12):
    """Relative mean absolute error (avoid division by zero)."""
    rel = lambda emul, true: np.mean(np.abs(emul - true) / (np.abs(true) + eps))
    return {
        "mean_rel": rel(uq_emul["mean"], uq_true["mean"]),
        "std_rel": rel(uq_emul["std"], uq_true["std"]),

    }

# ---------------------------------------------------------------------
# SOBOL / SA
# ---------------------------------------------------------------------
def compute_sobol(model_fn, dist, n_base=32):
    dim = len(dist)
    bounds = np.vstack([dist.lower, dist.upper]).T
    problem = {
        "num_vars": dim,
        "names": [f"x{i}" for i in range(dim)],
        "bounds": bounds.tolist(),
    }

    X = sobol_sample.sample(problem, n_base, calc_second_order=False)
    Y = model_fn(X)
    if Y.ndim == 1:
        Y = Y[:, None]

    n_outputs = Y.shape[1]
    S1 = np.zeros((dim, n_outputs))
    ST = np.zeros((dim, n_outputs))

    for j in range(n_outputs):
        Si = sobol_analyze.analyze(problem, Y[:, j], calc_second_order=False, print_to_console=False)
        S1[:, j] = Si["S1"]
        ST[:, j] = Si["ST"]

    return {"S1": S1, "ST": ST}

def compute_pce_sa_from_basis(surrogate,dist):
    """
    Compute first-order sensitivities from a PCE NDPoly surrogate.
    """
    poly = surrogate.model
    
    S1 = cp.Sens_m(poly, dist)
    return {"S1": S1}

def compare_sa(sa_emul, sa_true, eps=1e-12):
    """Relative L2 norm of SA indices."""
    rel = lambda emul, true: np.linalg.norm(emul - true) / (np.linalg.norm(true) + eps)
    return {"S1_rel": rel(sa_emul["S1"], sa_true["S1"])}

# ---------------------------------------------------------------------
# MAIN EVALUATION ROUTINE
# ---------------------------------------------------------------------
def evaluate_emulator_folder(emulator_folder, true_uq_csv, true_sa_csv, model, uq_samples, sa_base, output_csv):
    start_time = time.time()

    # Load true statistics
    true_uq_df = pd.read_csv(true_uq_csv)
    true_sa_df = pd.read_csv(true_sa_csv)

    uq_true = {key: true_uq_df[key].values for key in ["mean", "std"]}
    sa_true = {
        "S1": true_sa_df.pivot(index="param", columns="output", values="S1").values,
        "ST": true_sa_df.pivot(index="param", columns="output", values="ST").values,
    }

    dist = model.getDist()
    records = []

    for fname in os.listdir(emulator_folder):
        print(f"Evaluating surrogate: {fname}")
        if not fname.endswith((".pkl", ".pth")):
            continue
        path = os.path.join(emulator_folder, fname)

        try:
            emulator = Surrogate.load(path)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        records_emul = []

        # Check if emulator is a PCE
        is_pce = hasattr(emulator, "dist") and hasattr(emulator, "model")
        
        if is_pce:
            if(emulator.P>2):
                print(f"Skipping {fname} PCE with order {emulator.P} (too high for SA)")
                continue
            # -----------------------------
            # Entry 1: Use PCE basis
            # -----------------------------
            start_time = time.time()
            print(dist)
            uq_emul = compute_uq_from_pce(emulator,dist)
            uq_metrics = compare_uq(uq_emul, uq_true)
            sa_emul = compute_pce_sa_from_basis(emulator,dist)
            sa_metrics = compare_sa(sa_emul, sa_true)
            elapsed = time.time() - start_time

            records_emul.append({
                "model": fname + "_basis",
                **uq_metrics,
                **sa_metrics,
                "uq_time_s": elapsed / 2,  # optional split if you want separate timings
                "sa_time_s": elapsed / 2,
            })
            print(f"{fname} [PCE basis] metrics: {uq_metrics}, {sa_metrics} (time: {elapsed:.2f}s)")

            # -----------------------------
            # Entry 2: Use emulator normally
            # -----------------------------
            start_uq = time.time()
            uq_emul_mc =compute_uq(model_fn=emulator.predict, dist=emulator.dist, n_samples=uq_samples) 
            uq_metrics_mc = compare_uq(uq_emul_mc, uq_true)
            elapsed_uq = time.time() - start_uq

            start_sa = time.time()
            sa_emul_mc = compute_sobol(model_fn=emulator.predict, dist=dist, n_base=sa_base)
            sa_metrics_mc = compare_sa(sa_emul_mc, sa_true)
            elapsed_sa = time.time() - start_sa

            records_emul.append({
                "model": fname + "_mc",
                **uq_metrics_mc,
                **sa_metrics_mc,
                "uq_time_s": elapsed_uq,
                "sa_time_s": elapsed_sa,
            })
            print(f"{fname} [MC emulator] metrics: {uq_metrics_mc}, {sa_metrics_mc} (time: {elapsed_uq + elapsed_sa:.2f}s)")

        else:
            continue
            # Standard surrogate workflow
            start_uq = time.time()
            uq_emul = compute_uq(model_fn=emulator.predict, dist=dist, n_samples=uq_samples)
            uq_metrics = compare_uq(uq_emul, uq_true)
            elapsed_uq = time.time() - start_uq

            start_sa = time.time()
            sa_emul = compute_sobol(model_fn=emulator.predict, dist=dist, n_base=sa_base)
            sa_metrics = compare_sa(sa_emul, sa_true)
            elapsed_sa = time.time() - start_sa

            records_emul.append({
                "model": fname,
                **uq_metrics,
                **sa_metrics,
                "uq_time_s": elapsed_uq,
                "sa_time_s": elapsed_sa,
            })

        records.extend(records_emul)
        print(f"Evaluated surrogate: {fname}, total surrogate time: {sum(r.get('uq_time_s',0)+r.get('sa_time_s',0) for r in records_emul):.2f}s")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    total_elapsed = time.time() - start_time
    print(f"Total folder evaluation time: {total_elapsed:.2f}s")
    return df

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate surrogate models against true UQ and Sobol SA.")
    parser.add_argument("--model", choices=["a", "b", "h", "t"], required=True, help="Select true model (distribution source)")
    parser.add_argument("--emulator_folder", required=True, help="Folder containing surrogate models")
    parser.add_argument("--true_uq_csv", required=True, help="CSV with true-model UQ statistics")
    parser.add_argument("--true_sa_csv", required=True, help="CSV with true-model Sobol indices")
    parser.add_argument("--uq_samples", type=int, default=5000, help="Monte Carlo samples for surrogate UQ")
    parser.add_argument("--sa_base", type=int, default=128, help="Base Sobol sample size (Saltelli)")
    parser.add_argument("--output_csv", required=True, help="Output CSV with surrogate error metrics + timings")

    args = parser.parse_args()
    model = load_model(args.model)

    print(f"Evaluating surrogates for model {args.model}")
    print(f"UQ samples: {args.uq_samples}, Sobol base size: {args.sa_base}")

    total_start = time.time()
    evaluate_emulator_folder(
        emulator_folder=args.emulator_folder,
        true_uq_csv=args.true_uq_csv,
        true_sa_csv=args.true_sa_csv,
        model=model,
        uq_samples=args.uq_samples,
        sa_base=args.sa_base,
        output_csv=args.output_csv,
    )
    total_elapsed = time.time() - total_start
    print(f"Total script execution time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

