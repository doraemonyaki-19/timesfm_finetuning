"""
Evaluate TimesFM 2.5 forecasts: pretrained baseline vs finetuned model.

Downloads held-out ticker data, runs predictions with both models,
and computes comparison metrics (MSE, MAE, MAPE, RMSE, Directional Accuracy).

Usage:
  python scripts/evaluate_forecast.py \
    --tickers "NVDA,UNH,GS,CVX,KO,BA" \
    --checkpoint checkpoints/best \
    --horizons "7,30,60,120" \
    --max-context 512
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
  sys.path.insert(0, str(SRC))

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


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_ticker(ticker: str, years: int = 20) -> np.ndarray:
  end = datetime.datetime.now(datetime.timezone.utc).date()
  start = end - datetime.timedelta(days=365 * years)
  print(f"Downloading {ticker} ({start} to {end})...")
  df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
  if df is None or df.empty:
    raise RuntimeError(f"Failed to download {ticker}")
  close = df["Close"].values.astype(np.float32).flatten()
  close = close[~np.isnan(close)]
  close = close[close > 0]
  print(f"  {ticker}: {len(close)} data points")
  return close


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

def evaluate_model(model, contexts: list[np.ndarray], actuals: list[np.ndarray],
                   horizon: int, label: str) -> dict:
  """Run forecast and compute metrics for each ticker."""
  point_forecast, _ = model.forecast(horizon=horizon, inputs=contexts)

  results = {}
  all_pred, all_actual = [], []
  for i, (pred_row, actual_row) in enumerate(zip(point_forecast, actuals)):
    h = min(horizon, len(actual_row))
    pred = pred_row[:h]
    actual = actual_row[:h]
    context_last = contexts[i][-1]

    results[i] = {
      "mse": mse(pred, actual),
      "mae": mae(pred, actual),
      "rmse": rmse(pred, actual),
      "mape": mape(pred, actual),
      "dir_acc": directional_accuracy(pred, actual, context_last),
    }
    all_pred.append(pred)
    all_actual.append(actual)

  # Overall metrics
  all_pred = np.concatenate(all_pred)
  all_actual = np.concatenate(all_actual)
  results["overall"] = {
    "mse": mse(all_pred, all_actual),
    "mae": mae(all_pred, all_actual),
    "rmse": rmse(all_pred, all_actual),
    "mape": mape(all_pred, all_actual),
    "dir_acc": float(np.mean([results[i]["dir_acc"] for i in range(len(contexts))])),
  }
  return results, point_forecast


def print_table(ticker_names, baseline_res, finetuned_res):
  """Print a formatted comparison table."""
  header = f"{'Ticker':<10} | {'Model':<12} | {'MSE':>10} | {'MAE':>8} | {'RMSE':>8} | {'MAPE%':>7} | {'DirAcc%':>7}"
  sep = "-" * len(header)
  print(f"\n{sep}")
  print(header)
  print(sep)

  keys = list(range(len(ticker_names))) + ["overall"]
  names = ticker_names + ["OVERALL"]

  for key, name in zip(keys, names):
    b = baseline_res[key]
    f = finetuned_res[key]
    print(f"{name:<10} | {'Baseline':<12} | {b['mse']:>10.2f} | {b['mae']:>8.2f} | {b['rmse']:>8.2f} | {b['mape']:>6.1f}% | {b['dir_acc']:>6.1f}%")
    print(f"{'':10} | {'Finetuned':<12} | {f['mse']:>10.2f} | {f['mae']:>8.2f} | {f['rmse']:>8.2f} | {f['mape']:>6.1f}% | {f['dir_acc']:>6.1f}%")
    # Show delta
    mape_delta = f['mape'] - b['mape']
    dir_delta = f['dir_acc'] - b['dir_acc']
    print(f"{'':10} | {'Delta':<12} | {'':>10} | {'':>8} | {'':>8} | {mape_delta:>+6.1f}% | {dir_delta:>+6.1f}%")
    if key != "overall":
      print(f"{'':10} |{'-'*68}")
  print(sep)


def save_plots(ticker_names, contexts, actuals, baseline_preds, finetuned_preds,
               horizon, output_path):
  """Save comparison plots."""
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("matplotlib not installed, skipping plot generation.")
    return

  n = len(ticker_names)
  fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)

  for i, (ticker, ctx, actual) in enumerate(zip(ticker_names, contexts, actuals)):
    ax = axes[i, 0]
    h = min(horizon, len(actual))

    # Show last 100 context points + forecast
    show_ctx = min(100, len(ctx))
    ctx_x = np.arange(-show_ctx, 0)
    forecast_x = np.arange(0, h)

    ax.plot(ctx_x, ctx[-show_ctx:], color="gray", label="Context", linewidth=1)
    ax.plot(forecast_x, actual[:h], color="black", label="Actual", linewidth=2)
    ax.plot(forecast_x, baseline_preds[i][:h], color="blue", label="Baseline",
            linewidth=1.5, linestyle="--", alpha=0.8)
    ax.plot(forecast_x, finetuned_preds[i][:h], color="red", label="Finetuned",
            linewidth=1.5, linestyle="--", alpha=0.8)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"{ticker}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")

  plt.tight_layout()
  plt.savefig(output_path, dpi=150)
  print(f"\nPlot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description="Evaluate TimesFM baseline vs finetuned")
  parser.add_argument("--tickers", type=str, default="NVDA,UNH,GS,CVX,KO,BA",
                      help="Comma-separated tickers for evaluation (should differ from training)")
  parser.add_argument("--checkpoint", type=str, default="checkpoints/best",
                      help="Path to finetuned checkpoint directory")
  parser.add_argument("--baseline-id", type=str, default="google/timesfm-2.5-200m-pytorch",
                      help="HuggingFace model ID for baseline")
  parser.add_argument("--max-context", type=int, default=512)
  parser.add_argument("--horizons", type=str, default="7,30,60,120",
                      help="Comma-separated forecast horizons to evaluate (e.g., '7,30,60,120')")
  parser.add_argument("--plot", type=str, default="evaluation_results.png",
                      help="Output plot base filename (horizon suffix added for multi-horizon)")
  args = parser.parse_args()

  ticker_list = [t.strip() for t in args.tickers.split(",")]
  horizons = [int(h.strip()) for h in args.horizons.split(",")]
  max_horizon = max(horizons)

  # Download data
  print("=" * 60)
  print("DOWNLOADING EVALUATION DATA")
  print("=" * 60)
  all_series = {}
  for ticker in ticker_list:
    try:
      all_series[ticker] = download_ticker(ticker)
    except Exception as e:
      print(f"  WARNING: Skipping {ticker}: {e}")

  ticker_list = [t for t in ticker_list if t in all_series]
  if not ticker_list:
    print("ERROR: No tickers downloaded successfully.")
    return

  # Split: last `max_horizon` days = ground truth, preceding `max_context` days = input
  contexts = []
  actuals = []
  for ticker in ticker_list:
    series = all_series[ticker]
    needed = args.max_context + max_horizon
    if len(series) < needed:
      print(f"  WARNING: {ticker} only has {len(series)} points, need {needed}. Using what's available.")
      ctx_len = len(series) - max_horizon
      contexts.append(series[:ctx_len])
      actuals.append(series[ctx_len:])
    else:
      contexts.append(series[-(args.max_context + max_horizon):-max_horizon])
      actuals.append(series[-max_horizon:])

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

    # --- Baseline ---
    print("\n" + "=" * 60)
    print("LOADING BASELINE MODEL")
    print("=" * 60)
    baseline_model = load_model(args.baseline_id, args.max_context, max_horizon)

    print("EVALUATING BASELINE")
    baseline_results, baseline_preds = evaluate_model(
      baseline_model, contexts, actuals_h, horizon, "Baseline"
    )
    del baseline_model

    # --- Finetuned ---
    print("\n" + "=" * 60)
    print("LOADING FINETUNED MODEL")
    print("=" * 60)
    finetuned_model = load_model(args.checkpoint, args.max_context, max_horizon)

    print("EVALUATING FINETUNED")
    finetuned_results, finetuned_preds = evaluate_model(
      finetuned_model, contexts, actuals_h, horizon, "Finetuned"
    )
    del finetuned_model

    # Print comparison
    print_table(ticker_list, baseline_results, finetuned_results)

    # Save plots
    if multi:
      plot_path = str(plot_base.with_stem(f"{plot_base.stem}_h{horizon}"))
    else:
      plot_path = str(plot_base)
    save_plots(ticker_list, contexts, actuals_h, baseline_preds, finetuned_preds,
               horizon, plot_path)

    # Summary
    b_overall = baseline_results["overall"]
    f_overall = finetuned_results["overall"]
    print(f"\nSUMMARY (horizon={horizon}):")
    print(f"  Baseline  MAPE: {b_overall['mape']:.1f}%  DirAcc: {b_overall['dir_acc']:.1f}%")
    print(f"  Finetuned MAPE: {f_overall['mape']:.1f}%  DirAcc: {f_overall['dir_acc']:.1f}%")
    if f_overall['mape'] < b_overall['mape']:
      print(f"  => Finetuning IMPROVED MAPE by {b_overall['mape'] - f_overall['mape']:.1f}pp")
    else:
      print(f"  => Finetuning did not improve MAPE (delta: {f_overall['mape'] - b_overall['mape']:+.1f}pp)")


if __name__ == "__main__":
  main()
