
import chaospy as cp
import numpy as np
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.surrogate_models.DD_Models import ModelInterface 


class PCEModel(ModelInterface):
    def __init__(self, dist, P=2, regressor=None):
        self.dist = dist
        self.P = P
        self.regressor = regressor
        self.model = None
        self.type = f"PCE_{P}"
        self.metadata = {"type": self.type, "dist": str(dist), "P": P, "regressor": regressor}

    def _train(self, x, y, **kwargs):
        rule = kwargs.get('rule', 'three_terms_recurrence')
        poly_exp = cp.generate_expansion(self.P, self.dist)
        self.model = cp.fit_regression(poly_exp, np.array(x).T, y, model=self.regressor)

    def _predict(self, X, **kwargs):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return cp.call(self.model, np.array(X).T).T