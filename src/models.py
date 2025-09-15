from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class LinearRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class PolynomialRegressor:
    def __init__(self, degree: int = 3, include_bias: bool = False):
        self.model = Pipeline(steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ("lin", LinearRegression())
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

class LogisticRegressor:
    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressor":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict_label(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)
