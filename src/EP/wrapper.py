import numpy as np
import pandas as pd
import os, sys
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
        ti = 11000
        tf = 12000
        dt = 0.01
        dtS = .01
        # Set size parameters for all models
        model.setSizeParameters(ti, tf, dt, dtS)
        results = model.run(P)
       # print(results)
        #dVmax,ADP90,ADP50,Vreps,tdV
        
        
        all_keys = [
            "V_rest",
            "V_peak",
            "dVdt_max",
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

    def run(self, P: np.ndarray) -> np.ndarray:
        """
        Alias for predict.
        """
        return self.predict(P)
    
    def getDist(self):
        return self.model.getDist()
