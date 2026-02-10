"""
This is a baseline model class to map real parameters to a TTModel implemented in CUDA.

@author: yanbw
"""


###Base e modelo 
import subprocess 
import sys
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import six,os
from scipy.signal import find_peaks



import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.EP.njit_tt import run_model_njit_gpu

class TTCellModel:
    tf = 100.0
    ti = 0.0
    dt = 0.1
    dtS = 1.0
    parametersN = ["ki","ko","atp",'g_Na','g_CaL','g_to','g_Kr','g_Ks','g_K1','g_bna','g_bca','g_pCa','g_pK']
    K_o_default=5.40e+00
    g_CaL_default=1.750e-04
    g_Na_default=1.48380e+01
    K_i_default=138.3
    atp_default=5.6
    g_K1_defaults=5.4050e+00
    g_Kr_defaults=0.096
    g_Ks_defaults=0.245
    g_to_defaults=2.940e-01
    g_bca_defaults=5.920e-04
    g_pca_defaults=1 ##coef
    g_pk_defaults=1.460e-02

    @staticmethod
    def setSizeParameters(ti, tf, dt, dtS):
        TTCellModel.ti = ti
        TTCellModel.tf = tf
        TTCellModel.dt = dt
        TTCellModel.dtS = dtS




    @staticmethod      
    def compute_apd_no_interp(V, dt, V_rest, V_peak, fractions=[0.9, 0.5, 0.3],upstroke_idx=10):
        
        if upstroke_idx is None:
            return {f: np.nan for f in fractions}
        
        t_up = upstroke_idx * dt
        APDs = {}
        #print(V_rest)
        for f in fractions:
            V_target = V_peak - f * (V_peak - V_rest)
            # Find first index after upstroke below threshold
            below = np.where(V[upstroke_idx:] <= V_target)[0]
           # print( below)
            # only keep indices >= 100
            below = below[below >= 1000]
            if len(below) == 0:
                APDs[f] = np.nan  # never repolarized to this fraction
            else:
                APDs[f] = below[0] * dt  # duration relative to upstroke
            if( APDs[f]==0):
                print(below[0:50])
                
                print(APDs)
                print(kkk)
      #  print(APDs)



        return APDs

    @staticmethod
    def run(P_array,batch_size=None):
        n_samples = P_array.shape[0]
        results = []

        if batch_size is None or n_samples <= batch_size:
            batch_size = n_samples  # process all at once

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            P_batch = P_array[start:end]

            # Run model on this batch
            out_batch = run_model_njit_gpu(
                P_batch,
                dt=TTCellModel.dt,
                tf=TTCellModel.tf,
                ti=TTCellModel.ti,
                dtS=TTCellModel.dtS,
            )

            # Process each sample in the batch
            for a, row in enumerate(out_batch):
                V = row[-100001:-4, 0]  # last 1000 points of voltage
                dt = TTCellModel.dt

                # Resting potential
                V_rest = np.mean(V[-50:-1])

                # Peak potential
                V_peak = np.max(V)

                # Amplitude
                amplitude = V_peak - V_rest

                # Derivative and activation time
                dVdt = np.diff(V) / dt
                upstroke_idx = np.argmax(dVdt)

                dVdt_max = np.max(dVdt)
                T_up = upstroke_idx * dt

                # Find minimum after repolarization (for DPA)
                V_repol_idx = np.argmin(V[upstroke_idx + 15:]) + upstroke_idx
                T_repol = V_repol_idx * dt
                DPA = T_repol - T_up

                # APDs
                APDs_dict = TTCellModel.compute_apd_no_interp(
                    V,
                    dt,
                    V_rest,
                    V_peak,
                    fractions=[0.8, 0.5, 0.3],
                    upstroke_idx=upstroke_idx,
                )
                APD80 = APDs_dict[0.8]
                APD50 = APDs_dict[0.5]
                APD30 = APDs_dict[0.3]

                results.append({
                    "Wf": V,
                    "V_rest": V_rest,
                    "V_peak": V_peak,
                    "dVdt_max": dVdt_max,
                    "APD80": APD80,
                    "APD50": APD50,
                    "APD30": APD30,
                })

        return results
            



    @staticmethod
    def getEvalPoints():
        t = TTCellModel.ti
        ep = []
        while t <= TTCellModel.tf:
            ep.append(t)
            t += TTCellModel.dtS
        return ep

    @staticmethod
    def getNPar():
        return len(TTCellModel.parametersN)
