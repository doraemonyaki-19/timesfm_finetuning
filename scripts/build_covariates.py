"""Build covariates for TimesFM xreg evaluation.

Constructs calendar-based dynamic features and statistical static features
from dates and prices arrays.

Usage:
  from build_covariates import build_covariates
  covs = build_covariates(dates, prices, context_len=512, horizon=120)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_covariates(
  dates: np.ndarray,
  prices: np.ndarray,
  context_len: int,
  horizon: int,
) -> dict:
  """Build covariate arrays for a single ticker.

  The last `horizon` dates/prices are the test (horizon) period.
  The preceding `context_len` dates/prices are the train (context) period.

  Args:
    dates: array of datetime64 or similar, shape (context_len + horizon,).
    prices: float array, shape (context_len + horizon,).
    context_len: number of context time steps.
    horizon: number of forecast time steps.

  Returns:
    Dict with keys:
      train_dynamic_categoricals: {name: int array (context_len,)}
      test_dynamic_categoricals: {name: int array (horizon,)}
      static_numericals: {name: float}
      category_sizes: {name: int}
  """
  total = context_len + horizon
  assert len(dates) >= total, f"Need {total} dates, got {len(dates)}"
  assert len(prices) >= total, f"Need {total} prices, got {len(prices)}"

  # Use the last (context_len + horizon) points
  dates = dates[-total:]
  prices = prices[-total:]

  train_dates = dates[:context_len]
  test_dates = dates[context_len:]
  train_prices = prices[:context_len]

  # Convert to pandas for easy date attribute extraction
  train_dt = pd.DatetimeIndex(train_dates)
  test_dt = pd.DatetimeIndex(test_dates)

  # Dynamic categorical: day_of_week (0=Mon, 6=Sun)
  train_dow = train_dt.dayofweek.values.astype(np.int64)
  test_dow = test_dt.dayofweek.values.astype(np.int64)

  # Dynamic categorical: month (1-12) → remap to 0-11
  train_month = (train_dt.month.values - 1).astype(np.int64)
  test_month = (test_dt.month.values - 1).astype(np.int64)

  # Static numerical: historical volatility (std of log returns in context)
  log_returns = np.diff(np.log(train_prices.astype(np.float64).clip(min=1e-8)))
  hist_vol = float(np.std(log_returns)) if len(log_returns) > 1 else 0.0
  avg_log_ret = float(np.mean(log_returns)) if len(log_returns) > 1 else 0.0

  return {
    "train_dynamic_categoricals": {
      "day_of_week": train_dow,
      "month": train_month,
    },
    "test_dynamic_categoricals": {
      "day_of_week": test_dow,
      "month": test_month,
    },
    "static_numericals": {
      "hist_volatility": hist_vol,
      "avg_log_return": avg_log_ret,
    },
    "category_sizes": {
      "day_of_week": 7,
      "month": 12,
    },
  }
