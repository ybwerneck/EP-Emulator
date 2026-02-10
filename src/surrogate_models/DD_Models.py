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

import time
import pickle
import logging
import numpy as np
import torch

class ModelInterface:
    def __init__(self, name=None):
        """
        Base model interface.
        Args:
            name (str): Name of the model.
        """
        self.type = "Generic"
        self.name = name or self.__class__.__name__
        self.metadata = {}
        self.n_train_samples = 0  # Number of samples used to train
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'train' in cls.__dict__:
            raise TypeError("Subclasses should not override 'train'. Override '_train' instead.")
        if 'predict' in cls.__dict__:
            raise TypeError("Subclasses should not override 'predict'. Override '_predict' instead.")
    
    @instrument
    def train(self, x, y, normalize_x=True, **kwargs):
        """
        High-level training function that normalizes y (and optionally x)
        and tracks metadata.
        """
        # Normalize y
        self.y_min, self.y_max = y.min(axis=0), y.max(axis=0)
        y_normalized = self.normalize_y(y)

        # Optionally normalize x
        if normalize_x:
            self.x_min, self.x_max = x.min(axis=0), x.max(axis=0)
            x = self.normalize_x(x)

        start_time = time.time()
        self._train(x, y_normalized, **kwargs)
        end_time = time.time()

        self.metadata["time_train"] = end_time - start_time
        self.metadata["ds_train"] = np.shape(y)
        self.n_train_samples = y.shape[0]
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            self.metadata["gpu_memory_MB"] = torch.cuda.memory_allocated(device) / (1024 ** 2)

    @instrument
    def predict(self, X, **kwargs):
        """
        High-level prediction method. Automatically normalizes X if the model
        was trained with normalization.
        """
        if hasattr(self, "x_min") and self.x_min is not None and self.x_max is not None:
            X_norm = self.normalize_x(X)
        else:
            X_norm = Xsd


        print(np.shape(X_norm))
        y_normalized = self._predict(X_norm, **kwargs)
        return self.denormalize_y(y_normalized)

    def forward(self, X, **kwargs):
        if hasattr(self, "_forward"):
            return self.denormalize_y_forward(self._forward(X, **kwargs))
        else:
            raise NotImplementedError("This emulator does not implement low-level forward (no grad).")

    def normalize_x(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min)

    def denormalize_x(self, x):
        return x * (self.x_max - self.x_min) + self.x_min

    def normalize_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min)

    def denormalize_y(self, y):
        return y * (self.y_max - self.y_min) + self.y_min

    def denormalize_y_forward(self, y):
        if isinstance(y, torch.Tensor):
            y_min = torch.tensor(self.y_min, dtype=y.dtype, device=y.device)
            y_max = torch.tensor(self.y_max, dtype=y.dtype, device=y.device)
            return y * (y_max - y_min) + y_min
        else:
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
    def load(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        logging.debug(f"Model loaded from {filename}")
        return model
