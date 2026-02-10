
import chaospy as cp
import numpy as np
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.surrogate_models.DD_Models import ModelInterface 


import numpy as np
import chaospy as cp

class PCEModel(ModelInterface):
    def __init__(self, dist, P=2, regressor=None,name=None):
        """
        Polynomial Chaos Expansion (PCE) surrogate model.

        Parameters
        ----------
        dist : chaospy.Distribution
            Input distribution defining the stochastic domain.
        P : int, optional
            Total polynomial order of the expansion.
        regressor : optional
            Optional regression model passed to Chaospy for coefficient fitting.
        """
        super().__init__(name=name)
        self.dist = dist
        self.P = P
        self.regressor = regressor
        self.model = None
        self.type = f"PCE_{P}"

        self.metadata = {
            "type": self.type,
            "dist": str(dist),
            "P": P,
            "regressor": regressor
        }
    def _train(self, x, y, **kwargs):
        """
        Fit the PCE surrogate using regression.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_params)
            Training inputs (physical space).
        y : array-like, shape (n_samples, n_outputs)
            Training outputs.
        rule : str, optional
            Polynomial construction rule.
        """
        rule = kwargs.get("rule", "three_terms")

        # Polynomial expansion
        poly_exp = cp.generate_expansion(self.P, self.dist)

        # Convert inputs to stochastic space
        x = np.asarray(x)
        u = x.T

        # Fit PCE coefficients
        self.model = cp.fit_regression(poly_exp, u, y, model=self.regressor)

        # Compute predictions on training points
        y_hat = np.array([self.model(*pt) for pt in u.T])


    def _predict(self, X, **kwargs):
        """
        Predict outputs using the trained PCE model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_params)
            Input samples in physical space.

        Returns
        -------
        y_hat : ndarray, shape (n_samples, n_outputs)
            Numeric predictions.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")

        X = np.asarray(X)
        # Map physical inputs to stochastic space
        u = X.T

        # Evaluate numerically
        y_hat = np.array([self.model(*pt) for pt in u.T])

        return y_hat
