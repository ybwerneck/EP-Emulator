import numpy as np
import pandas as pd
import os, sys
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverse_problem import inverse_problem_DE
from src.surrogate_models.DD_Models import ModelInterface as surrogate
from src.surrogate_models.gaussian_process import *

from src.EP.ModelC import TTCellModelFull as modelC
from src.EP.ModelA import TTCellModelExt as modelA


def main(model_path, model_name, model_choice, grad_refine):

    # -----------------------------------------------------------
    # Select model (B maps to old ModelC)
    # -----------------------------------------------------------
    if model_choice == "A":
        model = modelA
    elif model_choice == "B":
        model = modelC
    else:
        raise ValueError("Model must be A or B")

    # -----------------------------------------------------------
    # Load dataset matching model
    # -----------------------------------------------------------
    data_dir = f"data/Generated_Data_5K/Model{model_choice}"

    X = pd.read_csv(os.path.join(data_dir, "X.csv")).values
    Y = pd.read_csv(os.path.join(data_dir, "Y.csv")).values

    # -----------------------------------------------------------
    # Load emulator
    # -----------------------------------------------------------
    emulator = surrogate.load(model_path)

    batch_size = 10

    np.random.seed(42)
    indices = np.random.choice(len(X), batch_size, replace=False)

    # -----------------------------------------------------------
    # Prior
    # -----------------------------------------------------------
    dist = model.getDist(low=0.75, high=1.25)

    # -----------------------------------------------------------
    # Results folder
    # -----------------------------------------------------------
    suffix = "_da" if grad_refine else ""

    results_dir = f"Results/InverseProblem/{model_choice}/{model_name}{suffix}"

    # -----------------------------------------------------------
    # Run inverse problem
    # -----------------------------------------------------------
    P_final, hist, S = inverse_problem_DE(
        emulator,
        X,
        Y,
        dist,
        batch_size=batch_size,
        checkpoint_interval=1,
        P_min=0 if model_choice == "A" else 0.75,
        P_max=1 if model_choice == "A" else 1.25,
        pop_size=150,
        num_iters=1000,
        results_dir=results_dir,
        grad_refine=grad_refine,
        indices=indices
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inverse problem with surrogate model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained surrogate model"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name used for results directory"
    )

    parser.add_argument(
        "--model",
        choices=["A", "B"],
        default="B",
        help="Underlying EP model"
    )

    parser.add_argument(
        "--grad_refine",
        action="store_true",
        help="Enable gradient refinement step"
    )

    args = parser.parse_args()

    main(args.model_path, args.model_name, args.model, args.grad_refine)