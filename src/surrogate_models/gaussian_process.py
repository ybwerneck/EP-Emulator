
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
import torch
import numpy as np
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.surrogate_models.DD_Models import ModelInterface 


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
    def __init__(self, num_tasks,kernel=SingleTaskGPModel):
        """
        Initializes the model with multiple independent single-task GP models.
        Each output dimension is modeled by its own GP.
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.likelihoods = [GaussianLikelihood() for _ in range(num_tasks)]
        self.models = [None] * num_tasks
        self.type = "Gaussian"
        self.metadata = {"type": self.type, "num_tasks": num_tasks}
        self.kernel=kernel
        print(kernel)

    def _train(self, x, y, **kwargs):
        tol = kwargs.get("tol", 1e-6)
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

        # Set up optimizers for each model
        optimizers = []
        lr = kwargs.get("learning_rate", 0.1)
        for t in range(self.num_tasks):
            optimizers.append(torch.optim.Adam(self.models[t].parameters(), lr=lr))

        # Determine maximum epochs: if epochs is -1, run indefinitely until early stopping.
        max_epochs = kwargs.get("epochs", 50)
        if max_epochs == -1:
            max_epochs = int(1e9)

        for epoch in range(int(max_epochs)):
            # Set models to training mode and perform an update step per task
            for t in range(self.num_tasks):
                self.models[t].train()
                self.likelihoods[t].train()
                optimizers[t].zero_grad()
                
                output = self.models[t](x_tensor)
                # Use negative marginal log likelihood for parameter update (optimizer step)
                mll = ExactMarginalLogLikelihood(self.likelihoods[t], self.models[t])
                loss = -mll(output, y_tensor[:, t])
                loss.backward()
                optimizers[t].step()

            # After updating, evaluate training MAE for early stopping.
            mae_list = []
            for t in range(self.num_tasks):
                self.models[t].eval()
                self.likelihoods[t].eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    preds = self.models[t](x_tensor)
                    # Compute MAE for this task
                    task_mae = torch.mean(torch.abs(preds.mean - y_tensor[:, t]))
                    mae_list.append(task_mae.item())
            # Average MAE across tasks
            avg_mae = np.mean(mae_list)
            print(f"Epoch {epoch+1}, Training MAE: {avg_mae:.4f}")

            # Early stopping based on training MAE improvement
            if avg_mae < best_mae - tol:
                best_mae = avg_mae
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1} with training MAE {avg_mae:.4f}")
                break

        self.metadata["training_epochs"] = epoch

    def _predict(self, x, batch_size=256*16, **kwargs):
        """
        Predict outputs for the given input features in batches.
        Aggregates predictions from all individual models.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predictions = [[] for _ in range(self.num_tasks)]

        for t in range(self.num_tasks):
            self.models[t].eval()
            self.likelihoods[t].eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i + batch_size]
                x_tensor = torch.tensor(x_batch, dtype=torch.float32).to(device)
                
                for t in range(self.num_tasks):
                    preds = self.likelihoods[t](self.models[t](x_tensor))
                    predictions[t].append(preds.mean.cpu().numpy())

        preds_all = []
        for t in range(self.num_tasks):
            preds_task = np.concatenate(predictions[t], axis=0)
            preds_all.append(preds_task.reshape(-1, 1))
        
        return np.concatenate(preds_all, axis=1)
