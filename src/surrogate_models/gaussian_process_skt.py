import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.surrogate_models.DD_Models import ModelInterface 
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error




    


class SklearnMultipleSingleGPs(ModelInterface):
    def __init__(self, num_tasks, kernel=None):
        """
        Multiple independent single-task GPs using scikit-learn.
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.kernel = kernel or (C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0))
        self.models = [
            GaussianProcessRegressor(kernel=self.kernel,
                                     n_restarts_optimizer=3,
                                     normalize_y=True)
            for _ in range(num_tasks)
        ]
        self.type = "Gaussian"
        self.metadata = {"type": self.type, "num_tasks": num_tasks}

    def _train(self, x, y, **kwargs):
        """
        Train each GP model independently for its task.
        """
        # y should be shape (n_samples, num_tasks)
        for t in range(self.num_tasks):
            self.models[t].fit(x, y[:, t])
        
        # Compute average training MAE for metadata
        avg_mae = np.mean([
            mean_absolute_error(y[:, t], self.models[t].predict(x))
            for t in range(self.num_tasks)
        ])
        self.metadata["training_mae"] = avg_mae
        print(f"Training MAE: {avg_mae:.4f}")

    def _predict(self, x, batch_size=None, **kwargs):
        """
        Predict for all tasks.
        Batch size is ignored since sklearn GPR handles data in one go.
        """
        preds = []
        for t in range(self.num_tasks):
            pred_t = self.models[t].predict(x)
            preds.append(pred_t.reshape(-1, 1))
        return np.hstack(preds)
