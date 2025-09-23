import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from inverse_problem import inverse_problem_DE
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Wrapper to make full model compatible with .predict()
# -----------------------------------------------------------


def run_model_to_array(P,model):
        """
        Runs the TTCellModelChannel and returns a consistent NumPy array.
        
        Args:
            P: np.ndarray of shape (n_candidates, n_params)
            use_gpu: bool, whether to use GPU
            regen: bool, regenerate simulations
            name: str, file name used internally by the model

        Returns:
            np.ndarray of shape (n_candidates, q) where q is the number of QoIs.
            Missing results are filled with np.nan.
        """
        # Configuration
        ti = 1000
        tf = 2000
        dt = 0.01
        dtS = .1
        # Set size parameters for all models
        model.setSizeParameters(ti, tf, dt, dtS)
        results = model.run(P)
       # print(results)
        #dVmax,ADP90,ADP50,Vreps,tdV
        
        
        all_keys = [
            "V_rest",
            "V_peak",
            "dVdt_max",
            "Amplitude",
            "DPA",
            "APD80",
            "APD50",
            "APD30"
        ]        
        # Build array, fill missing entries with NaN
        Y_array = np.array([[res.get(k, 100) for k in all_keys if not k=='Wf'] for res in results])
        print("Model output shape:", Y_array.shape)
        return Y_array

class FullModelWrapper:
    def __init__(self, model):
        """
        Wrap a TTCellModel (ModelA, ModelB, ModelC) to emulate .predict(P).
        """
        self.model = model

    def predict(self, P: np.ndarray) -> np.ndarray:
        """
        Predict outputs for parameter matrix P.

        Args:
            P (np.ndarray): shape (n_candidates, n_params)

        Returns:
            np.ndarray: shape (n_candidates, n_qois)
        """
        return run_model_to_array(P, self.model)



if __name__ == "__main__":
    from src.EP.ModelC import TTCellModelFull as modelC

    # Load dataset (for choosing ground truth cases)
    X = pd.read_csv("data/Generated_Data_5K/ModelC/X.csv").values
    Y = pd.read_csv("data/Generated_Data_5K/ModelC/Y.csv").values[:,0:]

    # Wrap the full model
    full_model = FullModelWrapper(modelC)

    # Prior
    dist = modelC.getDist(low=0.75, high=1.25)

    # Run inverse problem with DE
    P_final, hist = inverse_problem_DE(full_model, X, Y, dist,
                                       batch_size=20,pop_size=150,checkpoint_interval=1,  # try smaller batch for full model
                                       num_iters=2000,results_dir="Results/InverseProblem/Fmodel") # fewer iterations for speed
