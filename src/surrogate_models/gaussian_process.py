
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
import gpytorch
import torch
import numpy as np
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.surrogate_models.DD_Models import ModelInterface 

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


class SingleTaskGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SingleTaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    



class GPyTorchMultipleSingleModels(ModelInterface):
    def __init__(self, num_tasks,kernel=SingleTaskGPModel,name=None):
        super().__init__(name=name)
        """
        Initializes the model with multiple independent single-task GP models.
        Each output dimension is modeled by its own GP.
        """

        self.num_tasks = num_tasks
        self.likelihoods = [GaussianLikelihood() for _ in range(num_tasks)]
        self.models = [None] * num_tasks
        self.type = "Gaussian"
        self.metadata = {"type": self.type, "num_tasks": num_tasks}
        self.kernel=kernel
        print(kernel)

    def _train(self, x, y, **kwargs):
        tol = kwargs.get("tol", 1e-5)
        patience = kwargs.get("patience", 10)
        best_mae = float("inf")
        epochs_without_improvement = 0

        # Convert input data to torch tensors on GPU if available
        device = torch.device("cuda:0")
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        # Initialize individual models for each task
        for t in range(self.num_tasks):
            y_t = y_tensor[:, t]
            self.models[t] = self.kernel(x_tensor, y_t, self.likelihoods[t]).to(device)

        # Set up optimizers and schedulers for each model
        optimizers, schedulers = [], []
        lr = kwargs.get("learning_rate", 0.1)
        for t in range(self.num_tasks):
            opt = torch.optim.Adam(self.models[t].parameters(), lr=lr)
            optimizers.append(opt)
            schedulers.append(
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt,
                    mode="min",
                    factor=0.9,  # reduce LR multiplicatively
                    patience=patience//10,  # smaller patience for LR
                    threshold=tol,
#verbose=True,
                )
            )

        # Determine maximum epochs
        max_epochs = kwargs.get("epochs", 50)
        if max_epochs == -1:
            max_epochs = int(1e9)

        for epoch in range(int(max_epochs)):
            # Train each task model
            for t in range(self.num_tasks):
                self.models[t].train()
                self.likelihoods[t].train()
                optimizers[t].zero_grad()
                
                output = self.models[t](x_tensor)
                mll = ExactMarginalLogLikelihood(self.likelihoods[t], self.models[t])
                loss = -mll(output, y_tensor[:, t])
                loss.backward()
                optimizers[t].step()

            # Compute training MAE across tasks
            mae_list = []
            for t in range(self.num_tasks):
                self.models[t].eval()
                self.likelihoods[t].eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    preds = self.models[t](x_tensor)
                    task_mae = torch.mean(torch.abs(preds.mean - y_tensor[:, t]))
                    mae_list.append(task_mae.item())
            avg_mae = np.mean(mae_list)

            # Step schedulers with the average MAE
            for scheduler in schedulers:
                scheduler.step(avg_mae)

            if epoch % 10 == 0:  # print every 10 epochs
                current_lrs = [opt.param_groups[0]["lr"] for opt in optimizers]
                print(f"Epoch {epoch+1}, Training MAE: {avg_mae:.4f}, LRs: {current_lrs[0]}")

            # Early stopping
            if avg_mae < best_mae - tol:
                best_mae = avg_mae
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} with training MAE {avg_mae:.4f}")
                break

        self.metadata["training_epochs"] = epoch

    def _predict(self, x, batch_size=4096, **kwargs):
        """
        Predicts the posterior mean for each task using the trained GP models.
        Returns an array of shape (N, num_tasks).
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # storage for predictions
        preds_tasks = [[] for _ in range(self.num_tasks)]

        # put all models in eval mode
        for t in range(self.num_tasks):
            self.models[t].eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(x), batch_size):
                xb = x[i:i + batch_size]
                xb = torch.tensor(xb, dtype=torch.float32, device=device)

                for t in range(self.num_tasks):
                    out = self.models[t](xb)        # skip likelihood for speed
                    mean = out.mean                 # posterior mean f(x)
                    preds_tasks[t].append(mean.cpu().numpy())

        # concatenate per task
        preds = []
        for t in range(self.num_tasks):
            preds.append(np.concatenate(preds_tasks[t], axis=0).reshape(-1, 1))

        return np.concatenate(preds, axis=1)

