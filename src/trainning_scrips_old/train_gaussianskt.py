import pandas as pd
from Models import SklearnMultipleSingleGPs as GPModel  # Ensure this matches the location of your GPModel class
import numpy as np
import matplotlib.pyplot as plt
import sys




# General PyTorch utilities
import torch.nn as nn

if len(sys.argv) != 3:
    print("Usage: python load_model.py <SET> <PROB>")
    sys.exit(1)
 
    
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel as C
import numpy as np

def SmallGPkernel(input_dim=3):
    """
    Kernel polinomial grau 2 via DotProduct^2, com ARD.
    """
    # DotProduct já inclui bias: (σ0^2 + X·X')
    # Ao elevar a potência 2 simulamos polinomial grau 2
    poly_kernel = DotProduct(sigma_0=1.0) ** 2
    return C(1.0, (1e-3, 1e3)) * poly_kernel

def MediumGPkernel(input_dim=3):
    """
    Kernel Poly(grau 2) + RBF (ARD).
    """
    poly_kernel = DotProduct(sigma_0=1.0) ** 2
    rbf_kernel = RBF(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3))
    return C(1.0, (1e-3, 1e3)) * (poly_kernel + rbf_kernel)

def LargeGPkernel(input_dim=3):
    """
    Kernel Poly(grau 2) * RBF (ARD).
    """
    poly_kernel = DotProduct(sigma_0=1.0) ** 2
    rbf_kernel = RBF(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3))
    return C(1.0, (1e-3, 1e3)) * (poly_kernel * rbf_kernel)

    


    

# Parse arguments from the bash command line
data_set = sys.argv[1]
prb = sys.argv[2]
sets=[data_set]
kerneis={
    "gp_skt_Large":LargeGPkernel(),
    "gp_skt_Medium":MediumGPkernel(),
    "gp_skt_Small":SmallGPkernel(),
}
import os

for seti in sets:
 for name,kernel in kerneis.items():
    # Load datasets from specified paths
    x = pd.read_csv(f'data/Generated_Data_{seti}/Model{prb}/X.csv').values  # Shape [n, 3]
    y = pd.read_csv(f'data/Generated_Data_{seti}/Model{prb}/Y.csv').values  # Shape [n, 5]
    # Define a kernel for the Gaussian Process
    #Radial Basis Function kernel
    print(kernel)
    # Create a Gaussian Process model instance
    gp_model = GPModel(5,kernel=kernel)

    # Train the model
    gp_model.train(x, y, n_restarts_optimizer=10,epochs=-1,patience=1000,learning_rate=0.001)

    # Save the trained model to a file
    gp_model.save(f'trainned_models/models/prob_{prb}/{name}_{seti}.pkl')


    gp_model=GPModel.load(f'trainned_models/models/prob_{prb}/{name}_{seti}.pkl')
    print("Gaussian Process model trained and saved as 'trained_gp_model.pkl'.")

    ypred=gp_model.predict(x)
    yval=y

    print(np.shape(yval),print(np.shape(ypred)))
    # Plot each feature into a subplot
    fig, axes = plt.subplots(5, 1, figsize=(10, 15))

    for i in range(5):
    # axes[i].plot(yval[:, i], label=f'yval Feature {i+1}', linestyle='-', marker='o', markersize=4)
        axes[i].set_title(f'Feature {i+1}')
        #axes[i].legend()
        axes[i].grid(True)
        axes[i].set_xlim(np.min(ypred[:, i]),np.max(ypred[:, i]))
        axes[i].set_ylim(np.min(ypred[:, i]),np.max(ypred[:, i]))
        axes[i].scatter(ypred[:, i],yval[:, i],label=f'ypred Feature {i+1}')

    plt.xlabel('Sample Index')
    plt.tight_layout()
    os.makedirs(f"Results/{seti}/prob_{prb}", exist_ok=True)
    plt.savefig(f"Results/{seti}/prob_{prb}/{name}.png")