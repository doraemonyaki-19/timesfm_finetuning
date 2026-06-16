"""NumPy ridge regression for TimesFM external regressors (xreg).

Reimplements the core ridge regression from xreg_lib.py (which uses JAX)
in pure NumPy, for use with the PyTorch-based evaluation pipeline.

Usage:
  from xreg_numpy import NumpyXRegLinear
  model = NumpyXRegLinear(ridge_lambda=1.0)
  model.fit(X_train, y_train)
  corrections = model.predict(X_test)
"""
from __future__ import annotations

import numpy as np


def one_hot(categories: np.ndarray, n_classes: int) -> np.ndarray:
  """One-hot encode integer categories into (N, n_classes) matrix."""
  n = len(categories)
  out = np.zeros((n, n_classes), dtype=np.float64)
  out[np.arange(n), categories.astype(int)] = 1.0
  return out


def build_feature_matrix(
  dynamic_categoricals: dict[str, np.ndarray] | None = None,
  dynamic_numericals: dict[str, np.ndarray] | None = None,
  static_numericals: dict[str, float] | None = None,
  n_steps: int | None = None,
  category_sizes: dict[str, int] | None = None,
) -> np.ndarray:
  """Build a feature matrix from covariates.

  Args:
    dynamic_categoricals: {name: int array of shape (n_steps,)}
    dynamic_numericals: {name: float array of shape (n_steps,)}
    static_numericals: {name: float scalar} — broadcast across all steps
    n_steps: number of time steps (inferred from dynamic features if None)
    category_sizes: {name: n_classes} for one-hot encoding

  Returns:
    X: (n_steps, n_features) feature matrix with intercept column appended.
  """
  dynamic_categoricals = dynamic_categoricals or {}
  dynamic_numericals = dynamic_numericals or {}
  static_numericals = static_numericals or {}
  category_sizes = category_sizes or {}

  if n_steps is None:
    for v in dynamic_categoricals.values():
      n_steps = len(v)
      break
    if n_steps is None:
      for v in dynamic_numericals.values():
        n_steps = len(v)
        break
  if n_steps is None:
    raise ValueError("Cannot infer n_steps — provide at least one dynamic feature.")

  parts = []

  # One-hot encoded categoricals
  for name in sorted(dynamic_categoricals):
    arr = dynamic_categoricals[name]
    n_classes = category_sizes.get(name, int(arr.max()) + 1)
    parts.append(one_hot(arr, n_classes))

  # Dynamic numericals
  for name in sorted(dynamic_numericals):
    parts.append(dynamic_numericals[name].reshape(-1, 1).astype(np.float64))

  # Static numericals (broadcast)
  for name in sorted(static_numericals):
    parts.append(np.full((n_steps, 1), static_numericals[name], dtype=np.float64))

  # Intercept
  parts.append(np.ones((n_steps, 1), dtype=np.float64))

  return np.hstack(parts)


class NumpyXRegLinear:
  """Ridge regression for correcting TimesFM forecast residuals.

  Mirrors the logic of BatchedInContextXRegLinear from xreg_lib.py,
  but uses NumPy instead of JAX.
  """

  def __init__(self, ridge_lambda: float = 1.0):
    self.ridge_lambda = ridge_lambda
    self.beta: np.ndarray | None = None

  def fit(self, X: np.ndarray, y: np.ndarray) -> None:
    """Fit ridge regression: beta = (X'X + λI)^{-1} X'y.

    Args:
      X: (n_train, n_features) feature matrix.
      y: (n_train,) target residuals.
    """
    XtX = X.T @ X
    reg = self.ridge_lambda * np.eye(XtX.shape[0], dtype=np.float64)
    self.beta = np.linalg.solve(XtX + reg, X.T @ y.astype(np.float64))

  def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict corrections: X @ beta.

    Args:
      X: (n_test, n_features) feature matrix.

    Returns:
      corrections: (n_test,) array.
    """
    if self.beta is None:
      raise RuntimeError("Must call fit() before predict().")
    return X @ self.beta
