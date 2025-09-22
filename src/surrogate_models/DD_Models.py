from functools import partial
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import chaospy as cp
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

import logging
import time
import pickle

from gpytorch.kernels import RBFKernel, MaternKernel, PeriodicKernel, ScaleKernel, MultitaskKernel
from gpytorch.means import MultitaskMean, LinearMean

# Set up logging (this configuration can be adapted as needed)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def instrument(method):
    """Decorator to log the execution of methods and measure their runtime."""
    def wrapper(*args, **kwargs):
        pr = kwargs.get("meas_time", False)
        
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
      #  print(pr)
        if(pr):
            return result,end_time-start_time
        else:
            return result
    return wrapper

class ModelInterface:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'train' in cls.__dict__:
            raise TypeError("Subclasses should not override 'train'. Override '_train' instead.")
        if 'predict' in cls.__dict__:
            raise TypeError("Subclasses should not override 'predict'. Override '_predict' instead.")

    @instrument
    def train(self, x, y, **kwargs):
        self.y_min, self.y_max = y.min(axis=0), y.max(axis=0)
        logging.debug(f"train: y_min={self.y_min}, y_max={self.y_max}")
        y_normalized = self.normalize_y(y)
        
        start_time = time.time()
        self._train(x, y_normalized, **kwargs)
        end_time = time.time()
        
        self.metadata["time_train"] = end_time - start_time
        self.metadata["ds_train"]=np.shape(y)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            self.metadata["gpu_memory_MB"] = mem_allocated

    @instrument
    def predict(self, X, **kwargs):
        """
        High-level prediction: normalized, denormalized, instrumented.
        """
        y_normalized = self._predict(X, **kwargs)
        return self.denormalize_y(y_normalized)

    def forward(self, X, **kwargs):
        """
        Low-level gradient enable forward.
        """
        if hasattr(self, "_forward"):
            return self.denormalize_y_forward(self._forward(X, **kwargs))
        else:
            raise NotImplementedError("This emulator does not implement low-level forward (no grad).")

    @instrument
    def normalize_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)
    
    @instrument
    def denormalize_y(self, y):
            return y * (self.y_max - self.y_min) + self.y_min

    def denormalize_y_forward(self, y):
            """
            Gradient-enabled denormalization for use in forward passes.
            Works with torch tensors and supports autograd.
            """
            if isinstance(y, torch.Tensor):
                y_min = torch.tensor(self.y_min, dtype=y.dtype, device=y.device)
                y_max = torch.tensor(self.y_max, dtype=y.dtype, device=y.device)
                return y * (y_max - y_min) + y_min
            else:
                # fallback to numpy
                return y * (self.y_max - self.y_min) + self.y_min

    def _train(self, x, y, **kwargs):
        raise NotImplementedError("Subclasses must implement _train")

    def _predict(self, X, **kwargs):
        raise NotImplementedError("Subclasses must implement _predict")

    @instrument
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        logging.debug(f"Model saved to {filename}")

    @staticmethod
    @instrument
    def load(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        logging.debug(f"Model loaded from {filename}")
        return model


