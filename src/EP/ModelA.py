# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:43:30 2022

@author: yanbw
"""

###Base e modelo 
import subprocess 
import sys
import numpy as np
import sys
import numpy as np
import pandas as pd

import timeit
import re
import collections
import os
import six
import chaospy as cp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.EP.TTCellModel_njit import TTCellModel 

class TTCellModelExt(TTCellModel):
   

   

    @staticmethod
    def getDist(low=0,high=1):
            
        hypox=cp.Uniform(low,high)
        hyper=cp.Uniform(low,high)    
        acid=cp.Uniform(low,high) 
        dist = cp.J(hypox,hyper,acid)
        return dist

    @staticmethod
    def getNPar():
        return 3

    @staticmethod
    def cofs(ps):

        params=[
           (1-0.25*ps[2])*TTCellModel.g_Na_default,
           
           (1-0.25*ps[2])*TTCellModel.g_CaL_default,
            
           TTCellModel.K_i_default - 13.3*ps[2],
           
           TTCellModel.K_o_default + 4.6*ps[1],
           
           TTCellModel.atp_default - 3 * ps[0], ##Atp
           
           TTCellModel.g_K1_defaults, 
           TTCellModel.g_Kr_defaults, 
           TTCellModel.g_Ks_defaults,
            
           TTCellModel.g_to_defaults,
           TTCellModel.g_bca_defaults,
           TTCellModel.g_pk_defaults   ,
           TTCellModel.g_pca_defaults  ,
           
            
            ]
     
        return np.array(params)
         

    @staticmethod
    def run(P="",use_gpu=False, regen=True,name="out.txt"):  
        return TTCellModel.run(np.array([TTCellModelExt.cofs(p) for p in P]))


    @staticmethod
    def getNPar():
        return 12
         