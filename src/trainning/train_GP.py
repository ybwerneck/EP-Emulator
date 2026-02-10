import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.surrogate_models.gaussian_process import GPyTorchMultipleSingleModels as GPModel 

# GPyTorch imports for Gaussian Process modeling
import torch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, MaternKernel, LinearKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

# General PyTorch utilities
import torch.nn as nn

if len(sys.argv) != 3:
    print("Usage: python load_model.py <SET> <PROB>")
    sys.exit(1)
 
import torch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, MaternKernel, LinearKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal

# General PyTorch utilities    
import gpytorch
from gpytorch.means import LinearMean
from gpytorch.kernels import RBFKernel, MaternKernel, LinearKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP


class SmallGPModel(ExactGP):
    """
    GP_S model: Linear mean + RBF kernel.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        d = train_x.shape[-1]
        self.mean_module = LinearMean(d)
        self.covar_module = ScaleKernel(
            RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MediumGPModel(ExactGP):
    """
    GP_M model: Linear mean + RBF + Matérn(ν=2.5).
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        d = train_x.shape[-1]
        self.mean_module = LinearMean(d)
        self.covar_module = ScaleKernel(
            RBFKernel() + MaternKernel(nu=1.5)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class LargeGPModel(ExactGP):
    """
    GP_L model: Linear mean + RBF + Matérn(ν=1.5) + Linear kernel.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        d = train_x.shape[-1]
        self.mean_module = LinearMean(d)
        self.covar_module = ScaleKernel(
            RBFKernel() + MaternKernel(nu=1.5) + LinearKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)




# Parse arguments from the bash command line
data_set = sys.argv[1]
prb = sys.argv[2]
sets=[data_set]
kerneis={
    "gp_L":LargeGPModel,
    "gp_M":MediumGPModel,
    "gp_S":SmallGPModel,
}
for seti in sets:
 for name,kernel in kerneis.items():
    print(kernel)

    # Load datasets from specified paths
    x = pd.read_csv(f'data/Generated_Data_{seti}/Model{prb}/X.csv').values  # Shape [n, 3]
    y = pd.read_csv(f'data/Generated_Data_{seti}/Model{prb}/Y.csv').values  # Shape [n, 5]
    # Define a kernel for the Gaussian Process
    #Radial Basis Function kernel

    # Create a Gaussian Process model instance
    gp_model = GPModel(6,kernel=kernel,name=name)

    # Train the model
    gp_model.train(x, y, n_restarts_optimizer=1,epochs=-1,patience=100,learning_rate=0.01)

    # Save the trained model to a file
    gp_model.save(f'trainned_models/prob_{prb}/{name}_{seti}.pkl')


    gp_model=GPModel.load(f'trainned_models/prob_{prb}/{name}_{seti}.pkl')
    print("Gaussian Process model trained and saved as 'trained_gp_model.pkl'.")

    ypred=gp_model.predict(x)
    yval=y

    print(np.shape(yval),print(np.shape(ypred)))
    # Plot each feature into a subplot
    fig, axes = plt.subplots(6, 1, figsize=(10, 15))

    for i in range(6):
    # axes[i].plot(yval[:, i], label=f'yval Feature {i+1}', linestyle='-', marker='o', markersize=4)
        axes[i].set_title(f'Feature {i+1}')
        #axes[i].legend()
        axes[i].grid(True)
        axes[i].set_xlim(np.min(ypred[:, i]),np.max(ypred[:, i]))
        axes[i].set_ylim(np.min(ypred[:, i]),np.max(ypred[:, i]))
        axes[i].scatter(ypred[:, i],yval[:, i],label=f'ypred Feature {i+1}')

    os.makedirs(f"Results/Model_fits/{data_set}/prob_{prb}", exist_ok=True)
    plt.savefig(f"Results/Model_fits/{data_set}/prob_{prb}/gp_{name}.png")
    plt.close()  # Close the figure after saving to free memory
    print(f"Plot saved as Results/Model_fits/{data_set}/prob_{prb}/gp_{name}.png")