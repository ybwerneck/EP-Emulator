import numpy as np
import pandas as pd
import os, sys, argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverse_problem import inverse_problem_DE
from src.EP.wrapper import FullModelWrapper

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["A","B"], default="B")
    parser.add_argument("--results_dir", default="Results/InverseProblem")

    args = parser.parse_args()

    from src.EP.ModelC import TTCellModelFull as modelC
    from src.EP.ModelA import TTCellModelExt as modelA

    # --------------------------------------------------------
    # Select model (B maps to old C)
    # --------------------------------------------------------

    if args.model == "A":
        model = modelA
    elif args.model == "B":
        model = modelC

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

# --------------------------------------------------------
# Dataset (match model)
# --------------------------------------------------------

    data_dir = f"data/Generated_Data_5K/Model{args.model}"

    X = pd.read_csv(os.path.join(data_dir, "X.csv")).values
    Y = pd.read_csv(os.path.join(data_dir, "Y.csv")).values
    # --------------------------------------------------------
    # Wrap full model
    # --------------------------------------------------------

    full_model = FullModelWrapper(model)

    # --------------------------------------------------------
    # Prior
    # --------------------------------------------------------

    dist = model.getDist(low=0.75, high=1.25)

    # --------------------------------------------------------
    # Batch selection
    # --------------------------------------------------------

    batch_size = 10

    np.random.seed(42)
    indices = np.random.choice(len(X), batch_size, replace=False)

    # --------------------------------------------------------
    # Results directory
    # --------------------------------------------------------

    results_dir = os.path.join(args.results_dir, f"Fmodel")

    # --------------------------------------------------------
    # Run inverse problem
    # --------------------------------------------------------

    P_final, hist = inverse_problem_DE(
        full_model,
        X,
        Y,
        dist,
        batch_size=batch_size,
        pop_size=150,
        checkpoint_interval=1,
        num_iters=1000,
        results_dir=results_dir,
        indices=indices
    )