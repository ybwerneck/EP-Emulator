import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverse_problem import inverse_problem_DE
import matplotlib.pyplot as plt


from src.EP.wrapper import FullModelWrapper

if __name__ == "__main__":
    from src.EP.ModelC import TTCellModelFull as modelC

    # Load dataset (for choosing ground truth cases)
    X = pd.read_csv("data/Generated_Data_5K/ModelB/X.csv").values
    Y = pd.read_csv("data/Generated_Data_5K/ModelB/Y.csv").values[:,0:]

    # Wrap the full model
    full_model = FullModelWrapper(modelC)

    # Prior
    dist = modelC.getDist(low=0.75, high=1.25)

    # Run inverse problem with DE
    batch_size=15
    np.random.seed(42)
    indices = np.random.choice(len(X), batch_size, replace=False)

    P_final, hist = inverse_problem_DE(full_model, X, Y, dist,
                                       batch_size=batch_size,pop_size=150,checkpoint_interval=1,  # try smaller batch for full model
                                       num_iters=2000,results_dir="Results/InverseProblem/Fmodel",indices=indices) # fewer iterations for speed
