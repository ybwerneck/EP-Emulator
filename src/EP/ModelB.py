# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 22:22:48 2022

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
from scipy.integrate import odeint

import chaospy as cp
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.EP.TTCellModel_njit import TTCellModel  # Ensure this matches your path


class TTCellModelChannel(TTCellModel):
   

   
    @staticmethod
    def cofs(ps):
        params=[
            
                #AA + coefs
               (1-0.25*ps[2])*TTCellModel.g_Na_default*(ps[3] ),               
               (1-0.25*ps[2])*TTCellModel.g_CaL_default*(ps[4] ),                
               TTCellModel.K_i_default - 13.3*ps[2],               
               TTCellModel.K_o_default + 4.6*ps[1],
               TTCellModel.atp_default - 3 * ps[0], 
               
               
               TTCellModel.g_K1_defaults *(ps[5] ), 
               TTCellModel.g_Kr_defaults *( ps[6] ), 
               TTCellModel.g_Ks_defaults *(ps[7] ) ,                
               TTCellModel.g_to_defaults *( ps[8] )  ,
               TTCellModel.g_bca_defaults *(ps[9] )  ,
               TTCellModel.g_pk_defaults *(ps[10] )  ,
               TTCellModel.g_pca_defaults *(ps[11] )  ,

                
         ]
              
            
          
     
        return np.array(params)
    
    @staticmethod
    def getDist(low=0,high=1,disse_l=0,disse_h=0.0001):
       
        hypo=cp.Uniform(disse_l,disse_h) 
        hyper=cp.Uniform(disse_l,disse_h)    
        acid=cp.Uniform(disse_l,disse_h) 

        gk1=cp.Uniform(low,high)    
        gkr=cp.Uniform(low,high) 
        gks=cp.Uniform(low,high)    
        gto=cp.Uniform(low,high) 
        gbca=cp.Uniform(low,high)
        gna=cp.Uniform(low,high)
        gcal=cp.Uniform(low,high)
        gpk=cp.Uniform(low,high)
        g_pca=cp.Uniform(low,high)
        
        
        dist = cp.J(hypo,hyper,acid,gna,gcal,gk1,gkr,gks,gto,gbca,gpk,g_pca)
        return dist



    @staticmethod
    def run(P="",use_gpu=False, regen=True,name="out.txt"):  
        return TTCellModel.run(np.array([TTCellModelChannel.cofs(p) for p in P]))


    @staticmethod
    def getNPar():
        return 12
         