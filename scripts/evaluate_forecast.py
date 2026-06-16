"""
Evaluate TimesFM 2.5 forecasts: pretrained baseline vs finetuned model.

Downloads held-out ticker data, runs predictions with both models,
and computes comparison metrics (MSE, MAE, MAPE, RMSE, Directional Accuracy).

With --use-covariates, also applies xreg (ridge regression on calendar/statistical
covariates) to correct forecasts, producing a 4-way comparison.

Usage:
  python scripts/evaluate_forecast.py \
    --tickers "NVDA,UNH,GS,CVX,KO,BA" \
    --checkpoint finetune_usa/checkpoints/best \
    --horizons "7,30,60,120" \
    --max-context 512 \
    --use-covariates
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

# Ensure src and scripts are importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
  sys.path.insert(0, str(SCRIPTS))

try:
  import yfinance as yf
except ImportError:
  raise RuntimeError("yfinance is required: pip install yfinance")

import timesfm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(pred: np.ndarray, actual: np.ndarray) -> float:
  return float(np.mean((pred - actual) ** 2))


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
  return float(np.mean(np.abs(pred - actual)))


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
  return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mape(pred: np.ndarray, actual: np.ndarray) -> float:
  mask = np.abs(actual) > 1e-8
  if not mask.any():
    return float("nan")
  return float(np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])) * 100)


def directional_accuracy(pred: np.ndarray, actual: np.ndarray, context_last: float) -> float:
  """Percentage of steps where predicted direction matches actual direction."""
  pred_full = np.concatenate([[context_last], pred])
  actual_full = np.concatenate([[context_last], actual])
  pred_dir = np.sign(np.diff(pred_full))
  actual_dir = np.sign(np.diff(actual_full))
  return float(np.mean(pred_dir == actual_dir) * 100)


def compute_metrics(pred: np.ndarray, actual: np.ndarray, context_last: float) -> dict:
  """Compute all metrics for a single prediction."""
  return {
    "mse": mse(pred, actual),
    "mae": mae(pred, actual),
    "rmse": rmse(pred, actual),
    "mape": mape(pred, actual),
    "dir_acc": directional_accuracy(pred, actual, context_last),
  }


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_ticker(ticker: str, years: int = 20) -> tuple[np.ndarray, np.ndarray]:
  """Download ticker data and return (dates, prices)."""
  end = datetime.datetime.now(datetime.timezone.utc).date()
  start = end - datetime.timedelta(days=365 * years)
  print(f"Downloading {ticker} ({start} to {end})...")
  df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
  if df is None or df.empty:
    raise RuntimeError(f"Failed to download {ticker}")
  # Handle MultiIndex columns from yfinance
  if isinstance(df.columns, __import__("pandas").MultiIndex):
    df = df.droplevel("Ticker", axis=1)
  close = df["Close"].values.astype(np.float32).flatten()
  dates = df.index.values  # datetime64
  # Remove NaN/zero/negative
  mask = ~np.isnan(close) & (close > 0)
  close = close[mask]
  dates = dates[mask]
  print(f"  {ticker}: {len(close)} data points")
  return dates, close


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, max_context: int, max_horizon: int):
  """Load a TimesFM model and compile it for forecasting."""
  import os
  if os.path.isdir(model_id):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
      model_id, local_files_only=True
    )
  else:
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)

  model.compile(
    timesfm.ForecastConfig(
      max_context=max_context,
      max_horizon=max_horizon,
      normalize_inputs=True,
      use_continuous_quantile_head=True,
      force_flip_invariance=True,
      infer_is_positive=True,
      fix_quantile_crossing=True,
    )
  )
  return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_EPS = 1e-8

def evaluate_model(model, contexts: list[np.ndarray], actuals: list[np.ndarray],
                   horizon: int, label: str,
                   log_transform: bool = False) -> tuple[dict, list]:
  """Run forecast and compute metrics for each ticker.

  When log_transform=True the model receives log(context) as input and
  predictions are exp-transformed back to price-space before metric
  computation.  Use this for checkpoints trained with the Fu et al. log
  transform; the baseline model always runs in raw price-space.
  """
  if log_transform:
    model_inputs = [np.log(np.maximum(c, _EPS)).astype(np.float32) for c in contexts]
  else:
    model_inputs = contexts

  point_forecast, _ = model.forecast(horizon=horizon, inputs=model_inputs)

  results = {}
  all_pred, all_actual = [], []
  for i, (pred_row, actual_row) in enumerate(zip(point_forecast, actuals)):
    h = min(horizon, len(actual_row))
    pred = pred_row[:h]
    if log_transform:
      pred = np.exp(pred)
    actual = actual_row[:h]
    context_last = contexts[i][-1]
    results[i] = compute_metrics(pred, actual, context_last)
    all_pred.append(pred)
    all_actual.append(actual)

  # Overall metrics
  all_pred_cat = np.concatenate(all_pred)
  all_actual_cat = np.concatenate(all_actual)
  results["overall"] = {
    "mse": mse(all_pred_cat, all_actual_cat),
    "mae": mae(all_pred_cat, all_actual_cat),
    "rmse": rmse(all_pred_cat, all_actual_cat),
    "mape": mape(all_pred_cat, all_actual_cat),
    "dir_acc": float(np.mean([results[i]["dir_acc"] for i in range(len(contexts))])),
  }
  return results, point_forecast


def apply_xreg_corrections(
  contexts: list[np.ndarray],
  base_preds: list,
  actuals_h: list[np.ndarray],
  all_dates: list[np.ndarray],
  horizon: int,
  max_context: int,
  max_horizon: int,
) -> tuple[dict, list]:
  """Apply xreg ridge regression corrections to base predictions.

  Fits ridge regression on context residuals (actual - predicted trend),
  then applies corrections to horizon forecasts.

  Returns (results_dict, corrected_predictions).
  """
  from build_covariates import build_covariates
  from xreg_numpy import NumpyXRegLinear, build_feature_matrix

  corrected_preds = []
  results = {}
  all_pred, all_actual = [], []

  for i in range(len(contexts)):
    ctx = contexts[i]
    h = min(horizon, len(actuals_h[i]))
    base_pred = np.array(base_preds[i][:h], dtype=np.float64)
    actual = actuals_h[i][:h]

    dates = all_dates[i]
    ctx_len = len(ctx)

    # Build covariates for context + horizon period
    covs = build_covariates(dates, np.concatenate([ctx, actual]),
                            ctx_len, h)

    # For training the ridge regression, we use the context period.
    # The "residual" is a simple trend: difference between actual context
    # values and a naive last-value forecast (since we don't have backcast).
    # Instead, we fit the ridge regression to predict the context prices
    # directly (centered), then use the learned coefficients to adjust
    # the horizon forecast.

    # Build feature matrices
    cat_sizes = covs["category_sizes"]
    X_train = build_feature_matrix(
      dynamic_categoricals=covs["train_dynamic_categoricals"],
      static_numericals=covs["static_numericals"],
      n_steps=ctx_len,
      category_sizes=cat_sizes,
    )
    X_test = build_feature_matrix(
      dynamic_categoricals=covs["test_dynamic_categoricals"],
      static_numericals=covs["static_numericals"],
      n_steps=h,
      category_sizes=cat_sizes,
    )

    # Target for ridge: context residuals (actual context - linear trend)
    # We use a simple detrending: subtract mean to center
    ctx_f64 = ctx.astype(np.float64)
    ctx_mean = np.mean(ctx_f64)
    y_train = ctx_f64 - ctx_mean

    # Fit ridge regression
    xreg = NumpyXRegLinear(ridge_lambda=1.0)
    xreg.fit(X_train, y_train)

    # Predict corrections for the horizon
    # The correction = what the ridge model predicts for test features
    # minus what it predicts for the last context features (to get delta)
    test_correction = xreg.predict(X_test)
    # Use average of last 5 context points' predictions as anchor
    anchor = np.mean(xreg.predict(X_train[-min(5, len(X_train)):]))
    correction = test_correction - anchor

    # Apply correction to base prediction
    corrected = base_pred + correction
    # Ensure positive (stock prices)
    corrected = np.maximum(corrected, 0.01)

    corrected_preds.append(corrected.astype(np.float32))

    context_last = ctx[-1]
    results[i] = compute_metrics(corrected.astype(np.float32), actual, context_last)
    all_pred.append(corrected.astype(np.float32))
    all_actual.append(actual)

  all_pred_cat = np.concatenate(all_pred)
  all_actual_cat = np.concatenate(all_actual)
  results["overall"] = {
    "mse": mse(all_pred_cat, all_actual_cat),
    "mae": mae(all_pred_cat, all_actual_cat),
    "rmse": rmse(all_pred_cat, all_actual_cat),
    "mape": mape(all_pred_cat, all_actual_cat),
    "dir_acc": float(np.mean([results[i]["dir_acc"] for i in range(len(contexts))])),
  }
  return results, corrected_preds


def print_table(ticker_names, model_results: dict[str, dict]):
  """Print a formatted comparison table for multiple models.

  model_results: {model_name: results_dict}
  """
  model_names = list(model_results.keys())
  header = f"{'Ticker':<10} | {'Model':<18} | {'MSE':>10} | {'MAE':>8} | {'RMSE':>8} | {'MAPE%':>7} | {'DirAcc%':>7}"
  sep = "-" * len(header)
  print(f"\n{sep}")
  print(header)
  print(sep)

  keys = list(range(len(ticker_names))) + ["overall"]
  names = ticker_names + ["OVERALL"]

  for key, name in zip(keys, names):
    for mi, mname in enumerate(model_names):
      r = model_results[mname][key]
      print(f"{name if mi == 0 else '':<10} | {mname:<18} | {r['mse']:>10.2f} | {r['mae']:>8.2f} | {r['rmse']:>8.2f} | {r['mape']:>6.1f}% | {r['dir_acc']:>6.1f}%")
    # Show MAPE deltas vs first model (baseline)
    base_mape = model_results[model_names[0]][key]["mape"]
    deltas = []
    for mname in model_names[1:]:
      d = model_results[mname][key]["mape"] - base_mape
      deltas.append(f"{mname}: {d:+.1f}pp")
    print(f"{'':10} | {'MAPE delta':<18} | {', '.join(deltas)}")
    if key != "overall":
      print(f"{'':10} |{'-' * (len(sep) - 12)}")
  print(sep)


def save_plots(ticker_names, contexts, actuals, all_preds: dict[str, list],
               horizon, output_path):
  """Save comparison plots for multiple models.

  all_preds: {model_name: list of prediction arrays}
  """
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib not installed, skipping plot generation.")
    return

  colors = {
    "Baseline": "blue",
    "Baseline+xreg": "cyan",
    "Finetuned": "red",
    "Finetuned+xreg": "orange",
  }
  default_colors = ["blue", "red", "green", "purple", "orange", "cyan"]

  n = len(ticker_names)
  fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

  for i, (ticker, ctx, actual) in enumerate(zip(ticker_names, contexts, actuals)):
    ax = axes[i, 0]
    h = min(horizon, len(actual))

    show_ctx = min(100, len(ctx))
    ctx_x = np.arange(-show_ctx, 0)
    forecast_x = np.arange(0, h)

    ax.plot(ctx_x, ctx[-show_ctx:], color="gray", label="Context", linewidth=1)
    ax.plot(forecast_x, actual[:h], color="black", label="Actual", linewidth=2)

    for ci, (mname, preds) in enumerate(all_preds.items()):
      color = colors.get(mname, default_colors[ci % len(default_colors)])
      ax.plot(forecast_x, preds[i][:h], color=color, label=mname,
              linewidth=1.5, linestyle="--", alpha=0.8)

    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"{ticker}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")

  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  plt.close(fig)
  print(f"\nPlot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description="Evaluate TimesFM baseline vs finetuned")
  parser.add_argument("--tickers", type=str, default="NVDA,UNH,GS,CVX,KO,BA",
                      help="Comma-separated tickers for evaluation (should differ from training)")
  parser.add_argument("--checkpoint", type=str, default="finetune_usa/checkpoints/best",
                      help="Path to finetuned checkpoint directory")
  parser.add_argument("--baseline-id", type=str, default="google/timesfm-2.5-200m-pytorch",
                      help="HuggingFace model ID for baseline")
  parser.add_argument("--max-context", type=int, default=512)
  parser.add_argument("--horizons", type=str, default="7,30,60,120",
                      help="Comma-separated forecast horizons to evaluate (e.g., '7,30,60,120')")
  parser.add_argument("--use-covariates", action="store_true",
                      help="Apply xreg covariates (calendar + statistical features) to correct forecasts")
  parser.add_argument("--plot", type=str, default="evaluation_results.png",
                      help="Output plot base filename (horizon suffix added for multi-horizon)")
  parser.add_argument("--no-baseline", action="store_true",
                      help="Skip loading the baseline model (saves memory; use cached baseline numbers)")
  parser.add_argument("--eval-start-date", type=str, default=None,
                      help="ISO date (YYYY-MM-DD). When set, only data from this date onwards "
                           "is used for evaluation (held-out period after training cutoff).")
  parser.add_argument("--log-transform", action="store_true",
                      help="Apply log transform to context before finetuned inference and "
                           "exp transform to predictions. Use when the checkpoint was trained "
                           "with the Fu et al. log-price objective.")
  args = parser.parse_args()

  ticker_list = [t.strip() for t in args.tickers.split(",")]
  horizons = [int(h.strip()) for h in args.horizons.split(",")]
  max_horizon = max(horizons)

  # Download data
  print("=" * 60)
  print("DOWNLOADING EVALUATION DATA")
  print("=" * 60)
  all_data = {}  # ticker -> (dates, prices)
  for ticker in ticker_list:
    try:
      all_data[ticker] = download_ticker(ticker)
    except Exception as e:
      print(f"  WARNING: Skipping {ticker}: {e}")

  ticker_list = [t for t in ticker_list if t in all_data]
  if not ticker_list:
    print("ERROR: No tickers downloaded successfully.")
    return

  # Split context vs actuals.
  # Default: last max_horizon points = actuals, preceding max_context = context.
  # With --eval-start-date: find the first date >= cutoff; everything before is
  # context (capped at max_context), everything from that date on is actuals
  # (capped at max_horizon).  This ensures training data never leaks into eval.
  contexts = []
  actuals = []
  all_dates = []  # full date arrays for xreg
  valid_tickers = []
  for ticker in ticker_list:
    dates, series = all_data[ticker]
    if args.eval_start_date is not None:
      cutoff = np.datetime64(args.eval_start_date)
      split_idx = int(np.searchsorted(dates, cutoff))
      if split_idx >= len(series):
        print(f"  WARNING: {ticker}: eval_start_date {args.eval_start_date} is after all available data. Skipping.")
        continue
      if split_idx == 0:
        print(f"  WARNING: {ticker}: no context data before eval_start_date. Skipping.")
        continue
      ctx = series[max(0, split_idx - args.max_context):split_idx]
      act = series[split_idx:split_idx + max_horizon]
      ctx_dates = dates[max(0, split_idx - args.max_context):split_idx + max_horizon]
    else:
      needed = args.max_context + max_horizon
      if len(series) < needed:
        print(f"  WARNING: {ticker} only has {len(series)} points, need {needed}. Using what's available.")
        ctx_len = len(series) - max_horizon
        ctx = series[:ctx_len]
        act = series[ctx_len:]
        ctx_dates = dates
      else:
        ctx = series[-(args.max_context + max_horizon):-max_horizon]
        act = series[-max_horizon:]
        ctx_dates = dates[-(args.max_context + max_horizon):]
    contexts.append(ctx)
    actuals.append(act)
    all_dates.append(ctx_dates)
    valid_tickers.append(ticker)
  ticker_list = valid_tickers
  if not ticker_list:
    print("ERROR: No tickers have data in the requested eval window.")
    return

  # NOTE: TimesFM_2p5_200M_torch uses a class-level nn.Module, so all
  # instances share the same weights.  We must load + evaluate one model
  # at a time to avoid the second load overwriting the first.

  # --- Evaluate each horizon ---
  plot_base = Path(args.plot)
  multi = len(horizons) > 1

  for horizon in horizons:
    print("\n" + "#" * 60)
    print(f"HORIZON = {horizon} days")
    print("#" * 60)

    # Trim actuals to this horizon
    actuals_h = [a[:horizon] for a in actuals]

    # Collect results and predictions for all models
    model_results = {}
    model_preds = {}

    # --- Baseline ---
    if not args.no_baseline:
      print("\n" + "=" * 60)
      print("LOADING BASELINE MODEL")
      print("=" * 60)
      baseline_model = load_model(args.baseline_id, args.max_context, max_horizon)

      print("EVALUATING BASELINE")
      baseline_results, baseline_preds = evaluate_model(
        baseline_model, contexts, actuals_h, horizon, "Baseline"
      )
      del baseline_model
      model_results["Baseline"] = baseline_results
      model_preds["Baseline"] = baseline_preds
    else:
      print("\n[--no-baseline] Skipping baseline model load.")
      baseline_preds = None

    # --- Baseline + xreg ---
    if args.use_covariates and not args.no_baseline:
      print("\nAPPLYING XREG TO BASELINE")
      bx_results, bx_preds = apply_xreg_corrections(
        contexts, baseline_preds, actuals_h, all_dates,
        horizon, args.max_context, max_horizon,
      )
      model_results["Baseline+xreg"] = bx_results
      model_preds["Baseline+xreg"] = bx_preds

    # --- Finetuned ---
    print("\n" + "=" * 60)
    print("LOADING FINETUNED MODEL")
    print("=" * 60)
    finetuned_model = load_model(args.checkpoint, args.max_context, max_horizon)

    print("EVALUATING FINETUNED")
    finetuned_results, finetuned_preds = evaluate_model(
      finetuned_model, contexts, actuals_h, horizon, "Finetuned",
      log_transform=args.log_transform,
    )
    del finetuned_model
    model_results["Finetuned"] = finetuned_results
    model_preds["Finetuned"] = finetuned_preds

    # --- Finetuned + xreg ---
    if args.use_covariates:
      print("\nAPPLYING XREG TO FINETUNED")
      fx_results, fx_preds = apply_xreg_corrections(
        contexts, finetuned_preds, actuals_h, all_dates,
        horizon, args.max_context, max_horizon,
      )
      model_results["Finetuned+xreg"] = fx_results
      model_preds["Finetuned+xreg"] = fx_preds

    # Print comparison
    print_table(ticker_list, model_results)

    # Save plots
    if multi:
      plot_path = str(plot_base.with_stem(f"{plot_base.stem}_h{horizon}"))
    else:
      plot_path = str(plot_base)
    save_plots(ticker_list, contexts, actuals_h, model_preds, horizon, plot_path)

    # Summary
    print(f"\nSUMMARY (horizon={horizon}):")
    for mname, mres in model_results.items():
      o = mres["overall"]
      print(f"  {mname:<18} MAPE: {o['mape']:.1f}%  DirAcc: {o['dir_acc']:.1f}%")

    # Show improvement summary
    base_mape = model_results["Baseline"]["overall"]["mape"]
    for mname in list(model_results.keys())[1:]:
      delta = model_results[mname]["overall"]["mape"] - base_mape
      direction = "IMPROVED" if delta < 0 else "did not improve"
      print(f"  => {mname} {direction} MAPE by {abs(delta):.1f}pp ({delta:+.1f}pp)")


if __name__ == "__main__":
  main()
