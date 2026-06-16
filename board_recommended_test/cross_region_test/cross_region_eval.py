"""
Test C: Cross-Region Evaluation.

Uses existing finetuned checkpoints to forecast tickers from OTHER regions.
Compares finetuned vs baseline MAPE to test whether the edge generalizes
beyond the training region.

Tests:
  1. Europe (EUv2) checkpoint  -> US tickers (AAPL, MSFT, AMZN, GOOGL, META, JPM)
  2. Japan (hdecay1.0) checkpoint -> Europe tickers (UL, EADSF, VWAGY)
  3. US (Run4) checkpoint -> Japan tickers (8035.T, 6902.T, 4661.T)
  4. Baseline (pretrained) -> all tickers (reference)

For each test: download price data, run sliding-window MAPE evaluation at
60d and 120d horizons, compare finetuned vs baseline.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT / "scripts")]:
  if p not in sys.path:
    sys.path.insert(0, p)

import timesfm
import yfinance as yf
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────

_CONFIG_PATH = ROOT / "sector_routing_config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
  _cfg = yaml.safe_load(_f)

MAX_CONTEXT = _cfg["max_context"]
FORECAST_CONFIG = _cfg["forecast_config"]
REGIONS = _cfg["regions"]

BASELINE_PATH = str(ROOT / "model") if (ROOT / "model" / "model.safetensors").exists() \
  else "google/timesfm-2.5-200m-pytorch"

# Cross-region test matrix
TESTS = [
  {
    "name": "EU_ckpt -> US_tickers",
    "checkpoint": str(ROOT / REGIONS["europe"]["checkpoint"]),
    "label": REGIONS["europe"]["label"],
    "tickers": REGIONS["us"]["eval_tickers"] + REGIONS["us"]["train_tickers"],
  },
  {
    "name": "JP_ckpt -> EU_tickers",
    "checkpoint": str(ROOT / REGIONS["japan"]["checkpoint"]),
    "label": REGIONS["japan"]["label"],
    "tickers": REGIONS["europe"]["eval_tickers"] + REGIONS["europe"]["train_tickers"],
  },
  {
    "name": "US_ckpt -> JP_tickers",
    "checkpoint": str(ROOT / REGIONS["us"]["checkpoint"]),
    "label": REGIONS["us"]["label"],
    "tickers": REGIONS["japan"]["eval_tickers"] + REGIONS["japan"]["train_tickers"],
  },
]

HORIZONS = [60, 120]
N_WINDOWS = 15
WINDOW_STEP = 5


def download_ticker(ticker, years=20):
  end = datetime.datetime.now(datetime.timezone.utc).date()
  start = end - datetime.timedelta(days=365 * years)
  df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
  if isinstance(df.columns, pd.MultiIndex):
    df = df.droplevel("Ticker", axis=1)
  close = df["Close"].values.astype(np.float32).flatten()
  mask = ~np.isnan(close) & (close > 0)
  return close[mask]


def load_model(checkpoint_path, max_horizon):
  import os
  if os.path.isdir(checkpoint_path):
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
      checkpoint_path, local_files_only=True)
  else:
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint_path)
  model.compile(timesfm.ForecastConfig(
    max_context=MAX_CONTEXT, max_horizon=max_horizon, **FORECAST_CONFIG))
  return model


def sliding_window_mape(series, model, horizon, n_windows, step):
  """Compute MAPE over sliding windows."""
  min_needed = MAX_CONTEXT + horizon + (n_windows - 1) * step
  if len(series) < min_needed:
    usable = (len(series) - MAX_CONTEXT - horizon) // step + 1
    if usable < 1:
      return float("nan"), 0
    n_windows = usable

  mapes = []
  for w in range(n_windows):
    end_ctx = len(series) - horizon - (n_windows - 1 - w) * step
    start_ctx = end_ctx - MAX_CONTEXT
    if start_ctx < 0:
      start_ctx = 0
    context = series[start_ctx:end_ctx]
    actual = series[end_ctx:end_ctx + horizon]

    pf, _ = model.forecast(horizon=horizon, inputs=[context.tolist()])
    pred = np.array(pf[0][:horizon])

    mask = actual != 0
    if mask.sum() == 0:
      continue
    mape = np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
    mapes.append(mape)

  if not mapes:
    return float("nan"), 0
  return float(np.mean(mapes)), len(mapes)


def run_test(test, all_series):
  """Run one cross-region test: finetuned vs baseline on foreign tickers."""
  print(f"\n{'=' * 70}")
  print(f"TEST: {test['name']}")
  print(f"Checkpoint: {test['label']} ({test['checkpoint']})")
  print(f"Tickers: {', '.join(test['tickers'])}")
  print(f"{'=' * 70}")

  results = {}
  max_h = max(HORIZONS)

  # Load finetuned model
  print(f"\nLoading finetuned model ({test['label']})...")
  ft_model = load_model(test["checkpoint"], max_h)

  for h in HORIZONS:
    print(f"\n--- Horizon {h}d ---")
    for t in test["tickers"]:
      if t not in all_series:
        print(f"  {t}: SKIP (no data)")
        continue
      s = all_series[t]
      mape, nw = sliding_window_mape(s, ft_model, h, N_WINDOWS, WINDOW_STEP)
      key = (t, h)
      results.setdefault(key, {})["ft_mape"] = mape
      results[key]["ft_windows"] = nw
      print(f"  {t}: FT MAPE = {mape:.3f}% ({nw} windows)")

  del ft_model

  # Load baseline model
  print(f"\nLoading baseline model...")
  bl_model = load_model(BASELINE_PATH, max_h)

  for h in HORIZONS:
    print(f"\n--- Horizon {h}d (baseline) ---")
    for t in test["tickers"]:
      if t not in all_series:
        continue
      s = all_series[t]
      mape, nw = sliding_window_mape(s, bl_model, h, N_WINDOWS, WINDOW_STEP)
      key = (t, h)
      results.setdefault(key, {})["bl_mape"] = mape
      results[key]["bl_windows"] = nw
      print(f"  {t}: BL MAPE = {mape:.3f}% ({nw} windows)")

  del bl_model

  return results


def main():
  print("Cross-Region Evaluation (Test C)")
  print(f"Date: {datetime.date.today()}")
  print(f"Horizons: {HORIZONS}")
  print(f"Windows: {N_WINDOWS}, Step: {WINDOW_STEP}d")

  # Download all tickers once
  all_tickers = set()
  for test in TESTS:
    all_tickers.update(test["tickers"])

  print(f"\nDownloading {len(all_tickers)} tickers...")
  all_series = {}
  for t in sorted(all_tickers):
    print(f"  {t}...", end=" ", flush=True)
    try:
      s = download_ticker(t)
      all_series[t] = s
      print(f"{len(s)} pts")
    except Exception as e:
      print(f"FAILED: {e}")

  # Run all tests
  all_results = {}
  for test in TESTS:
    results = run_test(test, all_series)
    all_results[test["name"]] = results

  # Summary table
  print("\n" + "=" * 70)
  print("SUMMARY: Cross-Region Evaluation")
  print("=" * 70)
  print(f"{'Test':<25} {'Ticker':<10} {'Hz':>4} {'BL MAPE':>10} {'FT MAPE':>10} {'Delta':>8} {'Result':>10}")
  print("-" * 80)

  summary_rows = []
  for test_name, results in all_results.items():
    for (t, h), r in sorted(results.items()):
      bl = r.get("bl_mape", float("nan"))
      ft = r.get("ft_mape", float("nan"))
      delta = ft - bl
      result = "IMPROVED" if delta < 0 else "REGRESSED"
      print(f"{test_name:<25} {t:<10} {h:>4}d {bl:>9.3f}% {ft:>9.3f}% {delta:>+7.3f}% {result:>10}")
      summary_rows.append({
        "test": test_name,
        "ticker": t,
        "horizon": h,
        "bl_mape": round(bl, 3),
        "ft_mape": round(ft, 3),
        "delta_pp": round(delta, 3),
        "result": result,
      })

  # Per-test aggregates
  print("\n" + "-" * 80)
  print(f"{'Test':<25} {'Hz':>4} {'Avg BL':>10} {'Avg FT':>10} {'Avg Delta':>10} {'Improved':>10}")
  print("-" * 80)

  for test_name, results in all_results.items():
    for h in HORIZONS:
      h_results = {k: v for k, v in results.items() if k[1] == h}
      if not h_results:
        continue
      bl_avg = np.nanmean([v["bl_mape"] for v in h_results.values()])
      ft_avg = np.nanmean([v["ft_mape"] for v in h_results.values()])
      delta = ft_avg - bl_avg
      n_imp = sum(1 for v in h_results.values()
                  if v.get("ft_mape", 999) < v.get("bl_mape", 999))
      n_tot = len(h_results)
      print(f"{test_name:<25} {h:>4}d {bl_avg:>9.3f}% {ft_avg:>9.3f}% {delta:>+9.3f}% {n_imp}/{n_tot:>8}")

  # Save results
  out_path = Path(__file__).resolve().parent / "cross_region_results.json"
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
      "date": str(datetime.date.today()),
      "horizons": HORIZONS,
      "n_windows": N_WINDOWS,
      "window_step": WINDOW_STEP,
      "tests": summary_rows,
    }, f, indent=2)
  print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
  main()
