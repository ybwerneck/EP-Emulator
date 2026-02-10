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
from TTCellModel_njit import run_model_njit_gpu



class TTCellModel:
    tf=100
    ti=0
    dt=0.1
    dtS=1
    parametersN=["ki","ko","gna","gca","atp"]
    
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
    def parseR(name="out.txt"):  
        
      
       
      
      
        X=[]
        file = open(name, 'r')
        for row in file:
           aux=[]
           for x  in row.split(' '):
               try:
                   aux.append(float(x))
               except:
                   aux.append(-100)
           ads=TTCellModel.ads(aux[:-10],[0.5,0.9],aux[-10] )
         
           try:
               k={"Wf": aux[:-2] ,"dVmax":aux[-1],"ADP90":ads[1],"ADP50":ads[0],"Vreps":aux[-10],"tdV":aux[-2]}
           except:
             k={"Wf": aux[:-2] }
             #plt.plot(aux[:-2])
             #plt.show()
            # print("ADCALCERROR ",ads)
           X.append(k)
   
        return X
    
      

    
    @staticmethod 
    def prepareinput(P,cofs):
        parametersS=[cofs(p) for p in P]
        with open('m.txt','wb') as f:
             np.savetxt(f,parametersS, fmt='%.8f')
             
    @staticmethod
    def run(P_array,cofsF=0):  
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
                V = row[-10001:-4, 0]  # last 1000 points of voltage
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
                    fractions=[0.9, 0.5, 0.3],
                    upstroke_idx=upstroke_idx,
                )
                APD90 = APDs_dict[0.9]
                APD50 = APDs_dict[0.5]
                APD30 = APDs_dict[0.3]

                results.append({
                    "Wf": V,
                    "V_rest": V_rest,
                    "V_peak": V_peak,
                    "dVdt_max": dVdt_max,
                    "Amplitude": amplitude,
                    "APD80": APD90,
                    "APD50": APD50,
                    "APD30": APD30,
                })

        return results
            

    @staticmethod
    def setParametersOfInterest(parametersN):
        TTCellModel.parametersN=parametersN
      
    

    @staticmethod
    def getSimSize(): #Returns size of result vector for given simulation size parameters, usefull for knowing beforehand the number of datapoints to compare
        n=0#(tf-ti)/dt
        return n
    
    @staticmethod
    def setSizeParameters(ti,tf,dt,dtS):
        TTCellModel.ti=ti
        TTCellModel.tf=tf
        TTCellModel.dt=dt
        TTCellModel.dtS=dtS
        
    @staticmethod   #returns the time points at wich there is evalution
    def getEvalPoints():
        
        t=TTCellModel.ti
        ts=0
        ep=[]
        while(t<TTCellModel.tf):
            if (ts >= TTCellModel.dtS) :
                ep.append(t)
                ts = 0
            t=t+TTCellModel.dtS
            ts=ts+TTCellModel.dtS
            
        return ep    
    
    @staticmethod      
    def ads(sol,repoCofs,repos): ##calculo da velocidade de repolarização
        k=0
        i=0;
        out={}
        x=sol
        flag=0
        #plt.plot(sol)
        #plt.savefig("a.png")
        x=np.array(sol)
        index=0
        idxmax=0

        for value in x:
           index+=1  
           if(value==x.max()):
                        flag=1                
                        out[len(repoCofs)]=index  + TTCellModel.ti
                        idxmax=index
           if(flag==1):
                        k+=1
           if(flag==1 and repoCofs[i]*repos >= value):
                        out[i]= (k)
                        i+=1
           if(i>=len(repoCofs)):
               
                        break

         
        return out

    @staticmethod
    def callCppmodel(N,use_gpu=False,outpt="out.txt",inpt="m.txt"):  
     #   print("Calling solver")
        name="./src/EP/gpu_tt.out "
        args=name +" --tf="+str(TTCellModel.tf)+" --ti="+str(TTCellModel.ti)+" --dt="+str(TTCellModel.dt)+" --dt_save="+str(TTCellModel.dtS) +" --n="+str(N)+" --i="+inpt+" --o="+outpt  
       
        if(use_gpu):
            args=args+" --use_gpu=1"
     
        
     #   print("  kernel call:",args)
        output = subprocess.Popen(args,stdout=subprocess.PIPE,shell=True)
        string = output.stdout.read().decode("utf-8")
        # Delete the file "out.txt" if it exists
        if os.path.exists(inpt):
            os.remove(inpt)
          #  print("  Input File deleted.")
        else:
            print("File 'out.txt' does not exist.")


    @staticmethod
    def getDist(low=0,high=1):
            
        hypox=cp.Uniform(low,high)
        hyper=cp.Uniform(low,high)    
        acid=cp.Uniform(low,high) 
        dist = cp.J(hypox,hyper,acid)
        return -1

    @staticmethod
    def getNPar():
        return -1

    @staticmethod
    def cofs(ps):


     
        return []
    
    