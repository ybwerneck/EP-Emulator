import pandas as pd
from Models import GPyTorchMultipleSingleModels as GPModel  # Ensure this matches the location of your GPModel class
import numpy as np
import matplotlib.pyplot as plt
import sys
from gpytorch.models import ExactGP
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
 
    
class SmallGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SmallGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())  # RBF kernel with scale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
class MediumGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MediumGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(train_x.shape[-1])  # Trainable linear mean
        self.covar_module = ScaleKernel(
            RBFKernel() + MaternKernel(nu=2.5)
        )  # Composite kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
class LargeGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LargeGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()  # Optionally replace with a more complex mean
        self.covar_module = ScaleKernel(
            RBFKernel() + MaternKernel(nu=1.5) + LinearKernel()
        )  # Composite kernel with linear term
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    




# Parse arguments from the bash command line
data_set = sys.argv[1]
prb = sys.argv[2]
sets=[data_set]
kerneis={
    "gp_Large":LargeGPModel,
    "gp_Medium":MediumGPModel,
    "gp_Small":SmallGPModel,
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
    gp_model = GPModel(5,kernel=kernel)

    # Train the model
    gp_model.train(x, y, n_restarts_optimizer=10,epochs=-1,patience=1000,learning_rate=0.001)

    # Save the trained model to a file
    gp_model.save(f'trainned_models/prob_{prb}/{name}_{seti}.pkl')


    gp_model=GPModel.load(f'trainned_models/prob_{prb}/{name}_{seti}.pkl')
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
    plt.savefig(f"Results/{seti}/prob_{prb}/{name}.png")