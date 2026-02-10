import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverse_problem import inverse_problem_DE
import matplotlib.pyplot as plt
from src.surrogate_models.neural_networks import NModel
import pickle
# -----------------------------------------------------------
# Example usage
# -----------------------------------------------------------
if __name__ == "__main__":
    # Load dataset
    X = pd.read_csv("data/Generated_Data_5K/ModelB/X.csv").values
    Y = pd.read_csv("data/Generated_Data_5K/ModelB/Y.csv").values[:,0:]

    # Load emulator (NN in this case)
    with open("trainned_models/prob_B/nmodel_M_5K.pth", "rb") as f:
        nmodel: NModel = pickle.load(f)
    nmodel.model.eval()
    emulator = nmodel

    from src.EP.ModelC import TTCellModelFull as modelC
    dist = modelC.getDist(low=0.75, high=1.25)

    P_final, hist,S = inverse_problem_DE(emulator,X,Y,dist, batch_size=30,checkpoint_interval=10, pop_size=150, num_iters=1000,results_dir="Results/InverseProblem/NN_ModelPP",grad_refine=True)
    print(S)
    