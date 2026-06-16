"""
Evaluate a finetuned regional checkpoint on BOTH eval tickers and training tickers.

This runs evaluate_forecast.py twice:
  1. On held-out eval tickers (the real test — different from training data)
  2. On a subset of training tickers (diagnostic — did the model learn anything?)

Usage:
  cd timesfm
  python scripts/evaluate_regional.py --region china --checkpoint finetune_china/checkpoints/run_glia_2a/best
  python scripts/evaluate_regional.py --region europe --checkpoint finetune_europe/checkpoints/run_glia_v2_1a/best
  python scripts/evaluate_regional.py --region japan --checkpoint finetune_japan/checkpoints/run_glia_v2_1a/best
  python scripts/evaluate_regional.py --region us --checkpoint finetune_usa/checkpoints/best
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


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
  sys.exit("yfinance required: pip install yfinance")

from timesfm.timesfm_2p5 import timesfm_2p5_torch
from timesfm.configs import ForecastConfig

# ---------------------------------------------------------------------------
# Region definitions: eval tickers + training ticker subset for diagnostics
# ---------------------------------------------------------------------------

REGIONS = {
  "us": {
    "eval": ["NVDA", "UNH", "GS", "CVX", "KO", "BA"],
    "eval_training": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM"],
  },
  "china": {
    "eval": ["0388.HK", "2382.HK", "1177.HK"],
    "eval_training": ["0700.HK", "0941.HK", "2318.HK", "1299.HK", "0005.HK", "0001.HK"],
  },
  "japan": {
    "eval": ["8035.T", "6902.T", "4661.T"],
    "eval_training": ["7203.T", "6758.T", "9984.T", "8306.T", "4502.T", "7974.T"],
  },
  "europe": {
    "eval": ["UL", "EADSF", "VWAGY"],
    "eval_training": ["ASML", "SAP", "NVO", "AZN", "SHEL", "DEO"],
  },
}

HORIZONS = [5, 14, 30, 60, 90, 120]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mape(pred: np.ndarray, actual: np.ndarray) -> float:
  mask = np.abs(actual) > 1e-8
  if not mask.any():
    return float("nan")
  return float(np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])) * 100)


def directional_accuracy(pred: np.ndarray, actual: np.ndarray, context_last: float) -> float:
  pred_full = np.concatenate([[context_last], pred])
  actual_full = np.concatenate([[context_last], actual])
  pred_dir = np.sign(np.diff(pred_full))
  actual_dir = np.sign(np.diff(actual_full))
  return float(np.mean(pred_dir == actual_dir) * 100)

def variance_ratio(pred: np.ndarray, actual: np.ndarray) -> float:
  """Ratio of forecast variance to actuals variance."""
  pred_var = np.var(pred)
  actual_var = np.var(actual)
  if actual_var < 1e-8:
    return float("nan")
  return pred_var / actual_var


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_ticker(ticker: str, years: int = 20) -> np.ndarray | None:
  end = datetime.date.today()
  start = end - datetime.timedelta(days=365 * years)
  try:
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False, auto_adjust=True)
    if df is None or df.empty:
      return None
    if isinstance(df.columns, __import__("pandas").MultiIndex):
      df = df.droplevel("Ticker", axis=1)
    close = df["Close"].values.astype(np.float32).flatten()
    close = close[~np.isnan(close) & (close > 0)]
    return close if len(close) >= 252 else None
  except Exception as e:
    print(f"  ERROR downloading {ticker}: {e}")
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tickers(
    model,
    tickers: list[str],
    ticker_data: dict[str, np.ndarray],
    horizon: int,
    max_context: int,
) -> dict[str, dict]:
  """Evaluate model on a set of tickers at a given horizon."""
  results = {}
  for ticker in tickers:
    data = ticker_data.get(ticker)
    if data is None:
      continue

    if len(data) < max_context + horizon:
      print(f"  {ticker}: insufficient data ({len(data)} pts, need {max_context + horizon})")
      continue

    context = data[-(max_context + horizon):-horizon]
    actual = data[-horizon:]
    context_last = float(context[-1])

    point_forecast, _ = model.forecast(
      horizon=horizon,
      inputs=[context.tolist()],
    )
    pred = np.array(point_forecast[0][:horizon], dtype=np.float32)

    results[ticker] = {
      "mape": mape(pred, actual),
      "dir_acc": directional_accuracy(pred, actual, context_last),
      "variance_ratio": variance_ratio(pred, actual),
    }

  return results


def evaluate_all_horizons(
    model,
    tickers: list[str],
    ticker_data: dict[str, np.ndarray],
    horizons: list[int],
    max_context: int,
) -> dict[int, dict[str, dict]]:
  """Evaluate a single model on all tickers and horizons. Returns {horizon: {ticker: metrics}}."""
  results = {}
  for horizon in horizons:
    results[horizon] = evaluate_tickers(model, tickers, ticker_data, horizon, max_context)
  return results


def print_comparison(
    baseline_results: dict[int, dict[str, dict]],
    finetuned_results: dict[int, dict[str, dict]],
    tickers: list[str],
    horizons: list[int],
    label: str,
) -> list[dict]:
  """Print comparison table and return summary rows."""
  print(f"\n{'='*90}")
  print(f"  {label}")
  print(f"  Tickers: {', '.join(tickers)}")
  print(f"  Horizons: {horizons}")
  print(f"{'='*90}")

  summary_rows = []

  for horizon in horizons:
    print(f"\n--- Horizon: {horizon}d ---")

    b_horizon = baseline_results.get(horizon, {})
    f_horizon = finetuned_results.get(horizon, {})

    print(f"\n  {'Ticker':<12} {'Baseline MAPE':>15} {'Finetuned MAPE':>15} {'Delta':>10} | {'Var Ratio':>12}")
    print(f"  {'--'*42}")

    b_mapes, f_mapes, f_var_ratios = [], [], []
    for ticker in tickers:
      b = b_horizon.get(ticker, {})
      f = f_horizon.get(ticker, {})
      if not b or not f:
        continue
      bm, fm = b["mape"], f["mape"]
      fvr = f["variance_ratio"]
      delta = fm - bm
      b_mapes.append(bm)
      f_mapes.append(fm)
      f_var_ratios.append(fvr)
      sign = "+" if delta >= 0 else ""
      print(f"  {ticker:<12} {bm:>14.1f}% {fm:>14.1f}% {sign}{delta:>8.1f}pp | {fvr:>11.2f}")

    if b_mapes and f_mapes:
      bm_avg = np.mean(b_mapes)
      fm_avg = np.mean(f_mapes)
      fvr_avg = np.mean([vr for vr in f_var_ratios if not np.isnan(vr)]) if any(not np.isnan(vr) for vr in f_var_ratios) else float('nan')
      delta_avg = fm_avg - bm_avg
      sign = "+" if delta_avg >= 0 else ""
      print(f"  {'OVERALL':<12} {bm_avg:>14.1f}% {fm_avg:>14.1f}% {sign}{delta_avg:>8.1f}pp | {fvr_avg:>11.2f}")

      summary_rows.append({
        "horizon": horizon,
        "baseline_mape": bm_avg,
        "finetuned_mape": fm_avg,
        "delta_pp": delta_avg,
        "variance_ratio": fvr_avg,
        "n_tickers": len(b_mapes),
      })

  return summary_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description="Regional evaluation: eval + training tickers")
  parser.add_argument("--region", required=True, choices=list(REGIONS.keys()))
  parser.add_argument("--checkpoint", required=True, help="Path to finetuned checkpoint")
  parser.add_argument("--max-context", type=int, default=512)
  parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS),
                      help="Comma-separated horizons (default: 5,14,30,60,90,120)")
  parser.add_argument("--output", help="Output markdown file path")
  args = parser.parse_args()

  horizons = [int(h) for h in args.horizons.split(",")]
  region_cfg = REGIONS[args.region]
  all_tickers = list(set(region_cfg["eval"] + region_cfg["eval_training"]))

  # Download data
  print(f"\nDownloading data for {len(all_tickers)} tickers...")
  ticker_data = {}
  for ticker in all_tickers:
    print(f"  {ticker}...", end=" ", flush=True)
    data = download_ticker(ticker)
    if data is not None:
      ticker_data[ticker] = data
      print(f"{len(data)} pts")
    else:
      print("SKIPPED")

  # IMPORTANT: TimesFM_2p5_200M_torch uses a class-level nn.Module, so loading
  # a second checkpoint overwrites the first. We must evaluate each model fully
  # before loading the next one.

  # --- Evaluate baseline first ---
  baseline_id = "google/timesfm-2.5-200m-pytorch"
  local_model_path = ROOT / "model"
  if (local_model_path / "model.safetensors").exists():
      baseline_id = str(local_model_path)
  print(f"\nLoading baseline model from {baseline_id}...")
  model = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(baseline_id)
  model.compile(ForecastConfig(max_context=args.max_context, max_horizon=max(horizons)))

  print("Evaluating baseline on eval tickers...")
  baseline_eval = evaluate_all_horizons(model, region_cfg["eval"], ticker_data, horizons, args.max_context)
  print("Evaluating baseline on training tickers...")
  baseline_train = evaluate_all_horizons(model, region_cfg["eval_training"], ticker_data, horizons, args.max_context)

  # --- Now load finetuned (overwrites shared class-level weights) ---
  finetuned_path = Path(args.checkpoint)
  if not (finetuned_path / "model.safetensors").exists():
    sys.exit(f"Finetuned checkpoint not found at: {finetuned_path}")

  print(f"\nLoading finetuned model from {finetuned_path}...")
  model = timesfm_2p5_torch.TimesFM_2p5_200M_torch.from_pretrained(str(finetuned_path))
  model.compile(ForecastConfig(max_context=args.max_context, max_horizon=max(horizons)))

  print("Evaluating finetuned on eval tickers...")
  finetuned_eval = evaluate_all_horizons(model, region_cfg["eval"], ticker_data, horizons, args.max_context)
  print("Evaluating finetuned on training tickers...")
  finetuned_train = evaluate_all_horizons(model, region_cfg["eval_training"], ticker_data, horizons, args.max_context)

  # --- Print comparisons ---
  eval_summary = print_comparison(
    baseline_eval, finetuned_eval,
    region_cfg["eval"], horizons,
    label=f"EVAL TICKERS ({args.region.upper()}) — held-out, different from training",
  )

  train_summary = print_comparison(
    baseline_train, finetuned_train,
    region_cfg["eval_training"], horizons,
    label=f"TRAINING TICKERS ({args.region.upper()}) — diagnostic, held-out windows",
  )

  print(f"\n{'='*90}")
  print(f"  COMBINED SUMMARY — {args.region.upper()}")
  print(f"{'='*90}")
  print(f"\n  {'Horizon':>8} | {'Eval MAPE':>12} {'Delta':>8} {'Var Ratio':>12} | {'Train MAPE':>12} {'Delta':>8} {'Var Ratio':>12}")
  print(f"  {'--'*42}")
  for e, t in zip(eval_summary, train_summary):
    h = e["horizon"]
    e_sign = "+" if e["delta_pp"] >= 0 else ""
    t_sign = "+" if t["delta_pp"] >= 0 else ""
    print(f"  {h:>5}d  | {e['finetuned_mape']:>9.1f}%   {e_sign}{e['delta_pp']:>6.1f}pp {e['variance_ratio']:>11.2f} | {t['finetuned_mape']:>9.1f}%   {t_sign}{t['delta_pp']:>6.1f}pp {t['variance_ratio']:>11.2f}")

  # Write output file
  out_path = args.output or f"finetune_expts/{args.region}_eval_summary.md"
  out_file = ROOT / out_path
  out_file.parent.mkdir(parents=True, exist_ok=True)

  with open(out_file, "w") as f:
    f.write(f"# {args.region.upper()} — Regional Evaluation Summary\n\n")
    f.write(f"Date: {datetime.date.today()}\n")
    f.write(f"Checkpoint: `{args.checkpoint}`\n\n")

    f.write("## Eval Tickers (held-out)\n\n")
    f.write(f"Tickers: {', '.join(region_cfg['eval'])}\n\n")
    f.write("| Horizon | Baseline | Finetuned | Delta | Var Ratio |\n")
    f.write("|---------|----------|-----------|-------|-----------|\n")
    for row in eval_summary:
      sign = "+" if row["delta_pp"] >= 0 else ""
      f.write(f"| {row['horizon']}d | {row['baseline_mape']:.1f}% | {row['finetuned_mape']:.1f}% | {sign}{row['delta_pp']:.1f}pp | {row['variance_ratio']:.2f} |\n")

    f.write("\n## Training Tickers (held-out windows — diagnostic)\n\n")
    f.write(f"Tickers: {', '.join(region_cfg['eval_training'])}\n\n")
    f.write("| Horizon | Baseline | Finetuned | Delta | Var Ratio |\n")
    f.write("|---------|----------|-----------|-------|-----------|\n")
    for row in train_summary:
      sign = "+" if row["delta_pp"] >= 0 else ""
      f.write(f"| {row['horizon']}d | {row['baseline_mape']:.1f}% | {row['finetuned_mape']:.1f}% | {sign}{row['delta_pp']:.1f}pp | {row['variance_ratio']:.2f} |\n")

    f.write(f"\n---\n*Generated by evaluate_regional.py*\n")

  print(f"\nResults written to: {out_file}")


if __name__ == "__main__":
  main()
