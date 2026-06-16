"""
Sector Router: Bootstrap-based ticker classification and selective forecasting.

Two modes:
  1. Build routing table: evaluate finetuned vs baseline on all tickers,
     classify each into Tier 1 (finetuned), Tier 2 (baseline), Tier 3 (indeterminate)
     using bootstrap 95% CIs.

  2. Forecast: load routing table, forecast new data using the appropriate model
     per ticker.

Usage:
  # Build routing table for a region (in-sample, original mode)
  python scripts/sector_router.py build \
    --region china \
    --horizons "5,14,30,60,120" \
    --n-windows 15 --n-boot 10000

  # Build with temporal OOS validation (honest split)
  python scripts/sector_router.py build \
    --region china \
    --honest-split \
    --n-windows 30 --n-boot 10000

  # Build with wider window spacing to reduce overlap
  python scripts/sector_router.py build \
    --region china \
    --honest-split \
    --n-windows 30 --window-step 5 --n-boot 10000

  # Forecast using routing table
  python scripts/sector_router.py forecast \
    --region china \
    --tickers "0388.HK,2382.HK,0700.HK" \
    --horizon 120

  # Show current routing table
  python scripts/sector_router.py show --region china
"""
from __future__ import annotations
print("Executing sector_router.py")

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# ── Load configuration from YAML ──────────────────────────────────────────
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "sector_routing_config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

REGION_POLICY = _cfg["region_policy"]
SIMPLE_RULE_FT_MIN_HORIZON = _cfg["simple_rule_ft_min_horizon"]
NOISE_HORIZON_MAX = _cfg["noise_horizon_max"]
BLOCKED_CELLS = {r: set(hzs) for r, hzs in _cfg.get("blocked_cells", {}).items()}
NO_SHORT_BLOCKLIST = set(_cfg.get("no_short_blocklist", []))
DEAD_ZONE_THRESHOLDS = _cfg.get("dead_zone_thresholds", [0.0, 0.005, 0.01, 0.02, 0.05])
MAX_CONTEXT = _cfg["max_context"]
FORECAST_CONFIG = _cfg["forecast_config"]
REGIONS = _cfg["regions"]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yfinance as yf
import pandas as pd
import timesfm


# ── Helpers ─────────────────────────────────────────────────────────────────

def mape(pred, actual):
    mask = np.abs(actual) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])) * 100)


def long_flat_pnl(pred, actual, last_price):
    """P&L (%) from a long/flat strategy.

    Decision rule: go long if forecast endpoint > entry price, else flat (0%).
    Returns the realized return in percentage points.
    """
    if last_price < 1e-8:
        return float("nan")
    predicted_return = (pred[-1] - last_price) / last_price
    actual_return = (actual[-1] - last_price) / last_price
    if predicted_return > 0:
        return float(actual_return * 100)
    return 0.0


def long_short_flat_pnl(pred, actual, last_price):
    """P&L (%) from a long/short/flat strategy.

    Decision rule: go long if forecast endpoint > entry price,
    go short if forecast endpoint < entry.
    """
    if last_price < 1e-8:
        return float("nan")
    predicted_return = (pred[-1] - last_price) / last_price
    actual_return = (actual[-1] - last_price) / last_price
    if predicted_return > 0:
        return float(actual_return * 100)
    elif predicted_return < 0:
        return float(-actual_return * 100)
    return 0.0


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
    if checkpoint_path:
        model = timesfm.load_model(checkpoint_path)
    else:
        # Initialize the model structure
        model = timesfm.TimesFM_2p5_200M_torch(
            context_len=MAX_CONTEXT,
            horizon_len=max_horizon,
            num_layers=20,
            model_dims=1280,
            backend="torch",
        )
    
    # Compile the model with the forecast configuration
    model.compile(timesfm.ForecastConfig(
        max_context=MAX_CONTEXT, max_horizon=max_horizon, **FORECAST_CONFIG
    ))
    
    return model

# ── Core Logic ──────────────────────────────────────────────────────────────

def get_eval_windows(
    data: np.ndarray,
    n_windows: int,
    horizon: int,
    context: int,
    honest_split: bool,
    step: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate evaluation windows from time series data."""
    contexts, actuals = [], []
    max_start = len(data) - context - horizon
    if honest_split:
        # Temporal OOS: windows are contiguous and start from the end
        starts = range(max_start, max_start - n_windows * step, -step)
    else:
        # Original mode: random sampling of windows
        starts = np.random.randint(0, max_start + 1, n_windows)

    for start in starts:
        contexts.append(data[start : start + context])
        actuals.append(data[start + context : start + context + horizon])
    return contexts, actuals


def bootstrap_ci(
    data: np.ndarray, n_boot: int = 10000, alpha: float = 0.05
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval for the mean."""
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    return np.quantile(boot_means, alpha / 2), np.quantile(boot_means, 1 - alpha / 2)


def build_routing_table(
    region: str,
    horizons: list[int],
    n_windows: int,
    n_boot: int,
    honest_split: bool,
    window_step: int,
    seed: int,
    checkpoint: str,
    tier_metric: str,
):
    """Build and save the routing table for a given region."""
    np.random.seed(seed)
    max_horizon = max(horizons)
    tickers = REGIONS[region]["eval_tickers"] + REGIONS[region]["train_tickers"]
    metric_func = {"mape": mape, "long_flat_pnl": long_flat_pnl, "long_short_flat_pnl": long_short_flat_pnl}[tier_metric]

    print("Loading models...")
    ft_model = load_model(checkpoint, max_horizon)
    base_model = load_model(None, max_horizon) # None for baseline

    routing_table = {}
    for ticker in tqdm(tickers, desc=f"Processing tickers for {region}"):
        try:
            data = download_ticker(ticker)
            if len(data) < MAX_CONTEXT + max_horizon + n_windows:
                print(f"Skipping {ticker}: not enough data.")
                continue
        except Exception as e:
            print(f"Skipping {ticker}: could not download data ({e})")
            continue

        routing_table[ticker] = {}
        for h in horizons:
            contexts, actuals = get_eval_windows(
                data, n_windows, h, MAX_CONTEXT, honest_split, window_step
            )

            ft_preds, _ = ft_model.forecast(contexts)
            base_preds, _ = base_model.forecast(contexts)

            ft_metrics, base_metrics = [], []
            for i in range(n_windows):
                last_price = contexts[i][-1]
                if tier_metric == "mape":
                    ft_metrics.append(metric_func(ft_preds[i][:h], actuals[i][:h]))
                    base_metrics.append(metric_func(base_preds[i][:h], actuals[i][:h]))
                else:
                    ft_metrics.append(metric_func(ft_preds[i][:h], actuals[i][:h], last_price))
                    base_metrics.append(metric_func(base_preds[i][:h], actuals[i][:h], last_price))
            
            ft_metrics = np.array(ft_metrics)
            base_metrics = np.array(base_metrics)
            
            # Filter out NaNs
            valid_mask = ~np.isnan(ft_metrics) & ~np.isnan(base_metrics)
            if not np.any(valid_mask):
                routing_table[ticker][h] = 3 # Not enough valid windows
                continue

            diffs = base_metrics[valid_mask] - ft_metrics[valid_mask]
            lower, upper = bootstrap_ci(diffs, n_boot)

            if lower > 0:
                tier = 1  # Finetuned is better
            elif upper < 0:
                tier = 2  # Baseline is better
            else:
                tier = 3  # Indeterminate

            routing_table[ticker][h] = tier

    output_path = ROOT / f"routing_table_{region}.json"
    with open(output_path, "w") as f:
        json.dump(routing_table, f, indent=4)
    print(f"Routing table saved to {output_path}")

if __name__ == "__main__":
    print("Inside main block")
    parser = argparse.ArgumentParser(description="Sector-based ticker routing and forecasting.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Build command
    build_parser = subparsers.add_parser("build", help="Build the routing table.")
    build_parser.add_argument("--region", type=str, required=True, choices=REGIONS.keys(), help="Region to build the table for.")
    build_parser.add_argument("--horizons", type=str, default="5,14,30,60,120", help="Comma-separated list of horizons.")
    build_parser.add_argument("--n-windows", type=int, default=15, help="Number of evaluation windows.")
    build_parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap samples.")
    build_parser.add_argument("--honest-split", action="store_true", help="Use temporal out-of-sample split.")
    build_parser.add_argument("--window-step", type=int, default=1, help="Step size between windows.")
    build_parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap.")
    build_parser.add_argument("--checkpoint", type=str, required=True, help="Path to the finetuned model checkpoint.")
    build_parser.add_argument("--tier-metric", type=str, default="mape", choices=["mape", "long_flat_pnl", "long_short_flat_pnl"], help="Metric for tiering.")


    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Forecast using the routing table.")
    forecast_parser.add_argument("--region", type=str, required=True, choices=REGIONS.keys(), help="Region for the forecast.")
    forecast_parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of tickers.")
    forecast_parser.add_argument("--horizon", type=int, required=True, help="Forecast horizon.")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show the current routing table.")
    show_parser.add_argument("--region", type=str, required=True, choices=REGIONS.keys(), help="Region to show the table for.")

    args = parser.parse_args()

    if args.command == "build":
        horizons = [int(h) for h in args.horizons.split(",")]
        build_routing_table(
            region=args.region,
            horizons=horizons,
            n_windows=args.n_windows,
            n_boot=args.n_boot,
            honest_split=args.honest_split,
            window_step=args.window_step,
            seed=args.seed,
            checkpoint=args.checkpoint,
            tier_metric=args.tier_metric,
        )
    elif args.command == "forecast":
        tickers = args.tickers.split(",")
        # This is where the forecast function would be called
        print(f"Forecasting for tickers {tickers} in {args.region} with horizon {args.horizon}")
    elif args.command == "show":
        # This is where the show_routing_table function would be called
        print(f"Showing routing table for {args.region}")
