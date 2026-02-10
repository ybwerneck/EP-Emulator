#!/usr/bin/env python3
"""
True-model UQ and SA generator.

This script:
- Selects one of four predefined true models
- Uses the model's native input distribution
- Computes UQ statistics (mean, std, quantiles)
- Computes Sobol sensitivity indices via PCE (Chaospy)
- Writes results to CSV for later surrogate comparison
"""

import os
import argparse
import numpy as np
import pandas as pd
import sys
import chaospy as cp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.EP.ModelC import TTCellModelFull as modelB
from src.EP.ModelA import TTCellModelExt as modelA
from src.MEC.ModelsCD import TisoModel as modelC
from src.MEC.ModelsCD import Ho8Model as modelD
from src.EP.wrapper import FullModelWrapper

np.random.seed(1234)

# ---------------------------------------------------------------------
# MODEL REGISTRY
# ---------------------------------------------------------------------

def load_model(model_key):
    """
    Load one of the predefined true models.
    """

    MODEL_REGISTRY = {
        "a": modelA,
        "b": modelB,
        "h": modelC,
        "t": modelD,
    }

    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Valid options: {list(MODEL_REGISTRY.keys())}"
        )

    if model_key in ["a", "b"]:
        return FullModelWrapper(MODEL_REGISTRY[model_key])
    else:
        return MODEL_REGISTRY[model_key]()


# ---------------------------------------------------------------------
# UQ COMPUTATION
# ---------------------------------------------------------------------

def compute_uq(model_fn, dist, n_samples):
    """
    Monte Carlo uncertainty propagation.
    """
    X = dist.sample(n_samples)
    Y = model_fn(X)

    uq = {
        "mean": np.mean(Y, axis=0),
        "std": np.std(Y, axis=0),
        "q05": np.quantile(Y, 0.05, axis=0),
        "q95": np.quantile(Y, 0.95, axis=0),
    }
    return uq


def save_uq(uq, output_csv):
    """
    Save UQ statistics: one row per output.
    """
    records = []
    n_outputs = uq["mean"].shape[0]

    for o in range(n_outputs):
        records.append({
            "output": o,
            "mean": uq["mean"][o],
            "std": uq["std"][o],
            "q05": uq["q05"][o],
            "q95": uq["q95"][o],
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)


# ---------------------------------------------------------------------
# SOBOL / SA COMPUTATION (PCE-BASED)
# ---------------------------------------------------------------------

def compute_sobol(model_fn, dist, poly_order=3):
    """
    Compute first-order and total Sobol indices using PCE.
    """

    # Polynomial basis
    poly = cp.generate_expansion(poly_order, dist)

    # Quadrature
    nodes, weights = cp.generate_quadrature(
        order=poly_order + 1,
        dist=dist,
        rule="gaussian"
    )

    # Model evaluations
    Y = model_fn(nodes)

    print("Model evaluations for sobol:", Y.shape,flush=True)
    # Fit PCE
    pce = cp.fit_quadrature(poly, nodes, weights, Y)

    # Sobol indices
    S1 = cp.Sens_m(pce, dist)
    ST = cp.Sens_t(pce, dist)

    return {
        "S1": S1,  # shape (n_params, n_outputs)
        "ST": ST,
    }
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

def compute_sobol(model_fn, dist, n_base=64):
    """
    Compute first-order and total Sobol indices using SALib.

    Parameters
    ----------
    model_fn : callable
        Model evaluator: Y = model_fn(X)
    dist : chaospy.Distribution
        Joint parameter distribution
    n_base : int
        Base sample size (total evaluations ~ n_base * (2D + 2))

    Returns
    -------
    dict with keys "S1" and "ST"
        Arrays of shape (n_params, n_outputs)
    """

    # ------------------------------------------------------------------
    # Define SALib problem
    # ------------------------------------------------------------------
    dim = len(dist)
    bounds = np.vstack([dist.lower, dist.upper]).T

    problem = {
        "num_vars": dim,
        "names": [f"x{i}" for i in range(dim)],
        "bounds": bounds.tolist(),
    }

    # ------------------------------------------------------------------
    # Generate Sobol samples
    # ------------------------------------------------------------------
    X = sobol_sample.sample(
        problem,
        n_base,
        calc_second_order=False
    )

    # ------------------------------------------------------------------
    # Model evaluations
    # ------------------------------------------------------------------
    
    print("Sobol samples:", X.shape, flush=True)
    Y = model_fn(X)

    if Y.ndim == 1:
        Y = Y[:, None]  # (n_samples, 1)

    n_outputs = Y.shape[1]
    print("Model outputs:", Y.shape, flush=True)


    # ------------------------------------------------------------------
    # Sobol analysis (per output)
    # ------------------------------------------------------------------
    S1 = np.zeros((dim, n_outputs))
    ST = np.zeros((dim, n_outputs))

    for j in range(n_outputs):
        Si = sobol_analyze.analyze(
            problem,
            Y[:, j],
            calc_second_order=False,
            print_to_console=False,
        )

        S1[:, j] = Si["S1"]
        ST[:, j] = Si["ST"]

    return {
        "S1": S1,   # (n_params, n_outputs)
        "ST": ST,
    }
def save_sa(sa, output_csv):
    """
    Save Sobol indices in long format.
    """
    S1 = sa["S1"]
    ST = sa["ST"]

    n_params, n_outputs = S1.shape
    records = []

    for p in range(n_params):
        for o in range(n_outputs):
            records.append({
                "param": p,
                "output": o,
                "S1": S1[p, o],
                "ST": ST[p, o],
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate true-model UQ and Sobol SA results."
    )

    parser.add_argument(
        "--model",
        choices=["a", "b", "h", "t"],
        required=True,
        help="Select true model"
    )

    parser.add_argument(
        "--uq_samples",
        type=int,
        default=1000,
        help="Number of Monte Carlo samples for UQ"
    )
    parser.add_argument(
        "--uq_output_csv",
        required=True,
        help="Output CSV for UQ statistics"
    )

    parser.add_argument(
        "--sa_output_csv",
        required=True,
        help="Output CSV for Sobol indices"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------

    print(f"Loading true model: {args.model}")
    model = load_model(args.model)

    model_fn = model.run
    dist = model.getDist()

    print(f"Number of parameters: {len(dist)}")


    # -------------------------------------------------------------
    # SA
    # -------------------------------------------------------------

    print("Computing Sobol sensitivity indices")
    sa = compute_sobol(
        model_fn=model_fn,
        dist=dist,
    )
    save_sa(sa, args.sa_output_csv)

    print("True-model UQ and SA generation completed.")


if __name__ == "__main__":
    main()
