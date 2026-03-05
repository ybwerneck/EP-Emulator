import numpy as np
import pandas as pd
import os, sys
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverse_problem import inverse_problem_DE
from src.surrogate_models.DD_Models import ModelInterface as surrogate
from src.EP.ModelC import TTCellModelFull as modelB
from src.surrogate_models.gaussian_process import *

def main(model_path, model_name, grad_refine):

    # -----------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------
    X = pd.read_csv("data/Generated_Data_5K/ModelB/X.csv").values
    Y = pd.read_csv("data/Generated_Data_5K/ModelB/Y.csv").values[:, :]

    # -----------------------------------------------------------
    # Load emulator
    # -----------------------------------------------------------
    emulator = surrogate.load(model_path)

    batch_size = 15

    np.random.seed(42)
    indices = np.random.choice(len(X), batch_size, replace=False)

    dist = modelB.getDist(low=0.75, high=1.25)

    # -----------------------------------------------------------
    # Results folder
    # -----------------------------------------------------------
    suffix = "_da" if grad_refine else ""
    results_dir = f"Results/InverseProblem/{model_name}{suffix}"

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
        pop_size=150,
        num_iters=10000,
        results_dir=results_dir,
        grad_refine=grad_refine
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
        "--grad_refine",
        action="store_true",
        help="Enable gradient refinement step"
    )

    args = parser.parse_args()

    main(args.model_path, args.model_name, args.grad_refine)