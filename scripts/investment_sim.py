"""
Portfolio investment strategy simulation using TimesFM forecasts.

Simulates 6 strategies over the past 12 months:
  1. SP500          - benchmark, 100% in ^GSPC
  2. EqualHold      - equal weight all 15 tickers, buy-and-hold
  3. EqualRebal     - equal weight, rebalance monthly
  4. ForecastTop5   - top-5 tickers by Run4 predicted 1-month return
  5. ForecastAllPos - all tickers, weight by predicted positive return (Run4)
  6. SelectiveTop5  - top-5 using per-ticker best model (selective ensemble)

All model inference is batched: each checkpoint is loaded once, predicts all
(n_rebalances x 15) context windows in a single call, then deleted.

Usage:
  cd C:\\Users\\ylchen\\workspace\\timesfm
  python scripts/investment_sim.py
  python scripts/investment_sim.py --output-dir finetune_expts --n-rebalances 12
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(SRC), str(ROOT / "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import yfinance as yf
import pandas as pd
import timesfm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INVESTABLE_TICKERS = [
    "AAPL", "JNJ", "JPM", "XOM", "PG", "CAT", "NEE", "AMT",
    "WMT", "NVDA", "UNH", "GS", "CVX", "KO", "BA",
]
BENCHMARK_TICKER = "^GSPC"

# Per-ticker model routing for selective ensemble (all others -> Run4)
SELECTIVE_ROUTING = {
    "NVDA": "Glia4b",
    "GS":   "Glia4b",
    "KO":   "Glia4b",
    "CVX":  "Glia3a",
}
CHECKPOINT_MAP = {
    "Run4":   "checkpoints/best",
    "Glia3a": "checkpoints/run_glia_3a/best",
    "Glia4b": "checkpoints/run_glia_4b/best",
}

INITIAL_CAPITAL      = 100_000.0
RISK_FREE_ANNUAL     = 0.045
TRADING_DAYS_PER_YEAR = 252
REBALANCE_INTERVAL   = 21   # ~1 month


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
def download_all_prices(years: int = 5) -> dict[str, pd.Series]:
    """Download closing prices for all tickers. Returns ticker -> pd.Series(DatetimeIndex)."""
    end   = datetime.datetime.now(datetime.timezone.utc).date()
    start = end - datetime.timedelta(days=365 * years)
    all_tickers = INVESTABLE_TICKERS + [BENCHMARK_TICKER]
    result = {}
    for ticker in all_tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel("Ticker", axis=1)
        close = df["Close"].dropna()
        close = close[close > 0]
        result[ticker] = close.astype(np.float32)
        print(f"{len(close)} pts")
    return result


def compute_rebalance_dates(
    prices_data: dict[str, pd.Series],
    n_rebalances: int = 12,
    interval: int = REBALANCE_INTERVAL,
) -> list[datetime.date]:
    """Derive n_rebalances monthly rebalance dates, oldest first, using ^GSPC calendar."""
    sp_dates = prices_data[BENCHMARK_TICKER].index
    # Start from the most recent date and work backwards
    dates = []
    idx = len(sp_dates) - 1
    for _ in range(n_rebalances):
        idx -= interval
        if idx < 0:
            break
        dates.append(sp_dates[idx].date())
    return sorted(dates)


def build_context_windows(
    prices_data: dict[str, pd.Series],
    rebalance_dates: list[datetime.date],
    max_context: int = 512,
) -> dict:
    """
    Build flat list of context arrays for all (date, ticker) pairs.
    Order: date0_tick0, date0_tick1, ..., date0_tickN, date1_tick0, ...

    Returns dict with:
      contexts             - flat list of np.ndarray (length n_dates * n_tickers)
      last_prices          - np.ndarray shape (n_dates, n_tickers)  [price at rebalance date]
    """
    n_dates   = len(rebalance_dates)
    n_tickers = len(INVESTABLE_TICKERS)
    contexts    = []
    last_prices = np.zeros((n_dates, n_tickers), dtype=np.float32)

    for d_idx, rebal_date in enumerate(rebalance_dates):
        ts = pd.Timestamp(rebal_date)
        for t_idx, ticker in enumerate(INVESTABLE_TICKERS):
            series = prices_data[ticker]
            # Find last index on or before rebal_date
            mask = series.index <= ts
            if not mask.any():
                # No data before this date — use first available
                sub = series.iloc[:max_context].values.astype(np.float32)
            else:
                end_pos = np.where(mask)[0][-1] + 1   # exclusive
                start_pos = max(0, end_pos - max_context)
                sub = series.iloc[start_pos:end_pos].values.astype(np.float32)
            contexts.append(sub)
            last_prices[d_idx, t_idx] = float(sub[-1]) if len(sub) > 0 else 0.0

    return {
        "contexts":    contexts,
        "last_prices": last_prices,
    }


# ---------------------------------------------------------------------------
# Inference layer
# ---------------------------------------------------------------------------
def load_and_forecast(
    checkpoint_path: str,
    all_contexts: list,
    horizon: int,
    max_context: int,
    label: str,
) -> list[np.ndarray]:
    """Load one checkpoint, batch-forecast all contexts, delete model."""
    print(f"  Loading {label} ({checkpoint_path})...", end=" ", flush=True)
    if os.path.isdir(checkpoint_path):
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            checkpoint_path, local_files_only=True)
    else:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint_path)
    model.compile(timesfm.ForecastConfig(
        max_context=max_context,
        max_horizon=horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    ))
    point_forecast, _ = model.forecast(horizon=horizon, inputs=all_contexts)
    del model
    print("done")
    return [np.array(row[:horizon]) for row in point_forecast]


def build_forecast_cache(
    contexts_data: dict,
    horizon: int,
    max_context: int,
) -> dict[str, np.ndarray]:
    """
    Load each checkpoint once, forecast all contexts, return shaped cache.
    Returns {label: np.ndarray of shape (n_dates, n_tickers, horizon)}.
    """
    all_contexts = contexts_data["contexts"]
    n_tickers = len(INVESTABLE_TICKERS)
    n_dates   = len(all_contexts) // n_tickers

    cache = {}
    for label, ckpt in CHECKPOINT_MAP.items():
        flat_preds = load_and_forecast(ckpt, all_contexts, horizon, max_context, label)
        # Reshape: flat [n_dates * n_tickers] -> [n_dates, n_tickers, horizon]
        preds_3d = np.array(flat_preds).reshape(n_dates, n_tickers, horizon)
        cache[label] = preds_3d
    return cache


# ---------------------------------------------------------------------------
# Strategy layer
# ---------------------------------------------------------------------------
def _predicted_returns(cache: dict, label: str, last_prices: np.ndarray) -> np.ndarray:
    """Compute predicted return matrix. Shape: (n_dates, n_tickers)."""
    pred_final = cache[label][:, :, -1]          # last predicted price
    denom = np.where(last_prices > 0, last_prices, 1.0)
    return (pred_final - last_prices) / denom


def strategy_sp500(rebalance_dates) -> list[dict | None]:
    return [{"^GSPC": 1.0}] * len(rebalance_dates)


def strategy_equal_hold(rebalance_dates) -> list[dict | None]:
    w = 1.0 / len(INVESTABLE_TICKERS)
    alloc = [{t: w for t in INVESTABLE_TICKERS}]
    alloc += [None] * (len(rebalance_dates) - 1)
    return alloc


def strategy_equal_rebal(rebalance_dates) -> list[dict | None]:
    w = 1.0 / len(INVESTABLE_TICKERS)
    return [{t: w for t in INVESTABLE_TICKERS}] * len(rebalance_dates)


def strategy_forecast_top5(
    cache: dict, model_label: str, last_prices: np.ndarray, n_top: int = 5
) -> list[dict | None]:
    pred_rets = _predicted_returns(cache, model_label, last_prices)
    allocations = []
    for d_idx in range(last_prices.shape[0]):
        ranked = np.argsort(pred_rets[d_idx])[::-1]
        top_idx = ranked[:n_top]
        allocations.append({INVESTABLE_TICKERS[i]: 1.0 / n_top for i in top_idx})
    return allocations


def strategy_forecast_allpos(
    cache: dict, model_label: str, last_prices: np.ndarray
) -> list[dict | None]:
    pred_rets = _predicted_returns(cache, model_label, last_prices)
    n_tickers = len(INVESTABLE_TICKERS)
    allocations = []
    for d_idx in range(last_prices.shape[0]):
        r = pred_rets[d_idx]
        pos = np.where(r > 0, r, 0.0)
        total = pos.sum()
        if total < 1e-8:
            # All predictions non-positive — fall back to equal weight
            weights = {t: 1.0 / n_tickers for t in INVESTABLE_TICKERS}
        else:
            norm = pos / total
            weights = {INVESTABLE_TICKERS[i]: float(norm[i])
                       for i in range(n_tickers) if norm[i] > 1e-6}
        allocations.append(weights)
    return allocations


def strategy_selective_top5(
    cache: dict, last_prices: np.ndarray, n_top: int = 5
) -> list[dict | None]:
    n_dates   = last_prices.shape[0]
    n_tickers = len(INVESTABLE_TICKERS)
    allocations = []
    for d_idx in range(n_dates):
        pred_rets = np.zeros(n_tickers)
        for t_idx, ticker in enumerate(INVESTABLE_TICKERS):
            label = SELECTIVE_ROUTING.get(ticker, "Run4")
            pred_final = cache[label][d_idx, t_idx, -1]
            lp = last_prices[d_idx, t_idx]
            pred_rets[t_idx] = (pred_final - lp) / (lp + 1e-8)
        ranked = np.argsort(pred_rets)[::-1]
        top_idx = ranked[:n_top]
        allocations.append({INVESTABLE_TICKERS[i]: 1.0 / n_top for i in top_idx})
    return allocations


# ---------------------------------------------------------------------------
# Portfolio simulation engine
# ---------------------------------------------------------------------------
def _get_price(prices_data: dict, ticker: str, date: datetime.date) -> float:
    """Closing price for ticker on date; falls back to nearest prior day."""
    series = prices_data.get(ticker)
    if series is None:
        return 0.0
    ts = pd.Timestamp(date)
    try:
        return float(series.loc[ts])
    except KeyError:
        mask = series.index <= ts
        if mask.any():
            return float(series[mask].iloc[-1])
        return 0.0


def simulate_portfolio(
    allocations: list[dict | None],
    prices_data: dict[str, pd.Series],
    rebalance_dates: list[datetime.date],
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Simulate daily portfolio value over the full simulation period.
    allocations[i] = dict of {ticker: weight} to apply on rebalance_dates[i],
                     or None to carry forward without rebalancing.
    """
    sp_dates     = prices_data[BENCHMARK_TICKER].index
    sim_start    = rebalance_dates[0]
    sim_end      = sp_dates[-1].date()
    trading_days = [d.date() for d in sp_dates
                    if sim_start <= d.date() <= sim_end]

    holdings: dict[str, float] = {}  # ticker -> num_shares
    cash = initial_capital
    portfolio_values = []
    rebal_ptr = 0

    for day in trading_days:
        # Apply rebalancing on or after the scheduled date
        if (rebal_ptr < len(rebalance_dates)
                and day >= rebalance_dates[rebal_ptr]):
            target = allocations[rebal_ptr]
            rebal_ptr += 1
            if target is not None:
                # Mark-to-market current value
                port_val = cash
                for tkr, shares in holdings.items():
                    port_val += shares * _get_price(prices_data, tkr, day)
                # Liquidate and re-allocate
                holdings = {}
                cash = port_val
                for tkr, wt in target.items():
                    price = _get_price(prices_data, tkr, day)
                    if price > 0 and wt > 0:
                        holdings[tkr] = (port_val * wt) / price
                        cash -= port_val * wt

        # Daily mark-to-market
        val = cash
        for tkr, shares in holdings.items():
            val += shares * _get_price(prices_data, tkr, day)
        portfolio_values.append(val)

    return {
        "dates":    trading_days,
        "values":   np.array(portfolio_values),
        "holdings": holdings,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(values: np.ndarray, risk_free: float = RISK_FREE_ANNUAL) -> dict:
    total_ret   = (values[-1] - values[0]) / values[0]
    n_years     = len(values) / TRADING_DAYS_PER_YEAR
    ann_ret     = (1 + total_ret) ** (1 / n_years) - 1

    rolling_max = np.maximum.accumulate(values)
    drawdowns   = (values - rolling_max) / rolling_max
    max_dd      = float(np.min(drawdowns))

    daily_rets  = np.diff(values) / values[:-1]
    daily_rf    = (1 + risk_free) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    excess      = daily_rets - daily_rf
    sharpe      = (np.mean(excess) / (np.std(excess) + 1e-10)) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return {
        "total_return_pct":    total_ret * 100,
        "annualized_return_pct": ann_ret * 100,
        "max_drawdown_pct":    max_dd * 100,
        "sharpe_ratio":        float(sharpe),
        "final_value":         float(values[-1]),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_results_table(strategy_results: dict) -> None:
    print("\n" + "=" * 75)
    print("INVESTMENT SIMULATION RESULTS (past 12 months, monthly rebalancing)")
    print("=" * 75)
    print(f"{'Strategy':<22} | {'Total Ret':>9} | {'Ann Ret':>7} | "
          f"{'Max DD':>7} | {'Sharpe':>6} | {'Final $':>10}")
    print("-" * 75)
    for name, res in strategy_results.items():
        m = res["metrics"]
        print(f"{name:<22} | {m['total_return_pct']:>+8.1f}% | "
              f"{m['annualized_return_pct']:>+6.1f}% | "
              f"{m['max_drawdown_pct']:>6.1f}% | "
              f"{m['sharpe_ratio']:>6.2f} | "
              f"${m['final_value']:>9,.0f}")


def print_allocation_table(name: str, allocations: list, rebalance_dates: list) -> None:
    print(f"\n--- {name}: Monthly Allocations ---")
    print(f"{'Date':<13} | Holdings")
    print("-" * 65)
    for date, alloc in zip(rebalance_dates, allocations):
        if alloc is None:
            print(f"{str(date):<13} | (carry forward)")
            continue
        items = sorted(alloc.items(), key=lambda x: -x[1])
        parts = ", ".join(f"{t} {w:.0%}" for t, w in items if w > 0.001)
        print(f"{str(date):<13} | {parts}")


def save_results(strategy_results: dict, rebalance_dates: list,
                 output_path: Path) -> None:
    lines = [
        "# Investment Simulation Results\n\n",
        f"Date: {datetime.date.today()}\n",
        f"Simulation period: ~12 months ({rebalance_dates[0]} to {rebalance_dates[-1]})\n",
        "Tickers: AAPL, JNJ, JPM, XOM, PG, CAT, NEE, AMT, WMT, NVDA, UNH, GS, CVX, KO, BA\n",
        "Benchmark: ^GSPC\n\n",
        "## Performance Summary\n\n",
        "| Strategy | Total Return | Ann. Return | Max Drawdown | Sharpe |\n",
        "|----------|-------------|------------|--------------|--------|\n",
    ]
    for name, res in strategy_results.items():
        m = res["metrics"]
        lines.append(
            f"| {name} | {m['total_return_pct']:+.1f}% | "
            f"{m['annualized_return_pct']:+.1f}% | "
            f"{m['max_drawdown_pct']:.1f}% | "
            f"{m['sharpe_ratio']:.2f} |\n"
        )
    lines.append("\n## Notes\n\n")
    lines.append("- ForecastTop5 / SelectiveTop5: equal weight across top 5 tickers by predicted 1-month return\n")
    lines.append("- ForecastAllPos: weight proportional to predicted positive returns (Run4)\n")
    lines.append("- SelectiveTop5 routing: NVDA/GS/KO→Glia4b, CVX→Glia3a, others→Run4\n")
    lines.append("- Risk-free rate: 4.5% annualized\n")
    lines.append("- No transaction costs, no short selling\n")
    output_path.write_text("".join(lines))


def save_plot(strategy_results: dict, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    colors = {
        "SP500":               "black",
        "EqualHold":           "gray",
        "EqualRebal":          "steelblue",
        "ForecastTop5":        "orange",
        "ForecastAllPos":      "green",
        "SelectiveTop5":       "crimson",
    }
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, res in strategy_results.items():
        vals  = res["sim"]["values"]
        dates = res["sim"]["dates"]
        norm  = vals / vals[0] * 100
        m     = res["metrics"]
        ax.plot(dates, norm,
                label=f"{name} ({m['total_return_pct']:+.1f}%)",
                color=colors.get(name, "purple"),
                linewidth=2.0)

    ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value (start = 100)", fontsize=11)
    ax.set_title("Portfolio Strategy Comparison — Past 12 Months", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",      default="finetune_expts")
    parser.add_argument("--horizon",   type=int, default=21,
                        help="Forecast horizon in trading days (default 21 = ~1 month)")
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--n-rebalances", type=int, default=12)
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    print("=" * 60)
    print("INVESTMENT SIMULATION")
    print("=" * 60)

    # 1. Download prices
    print("\nDownloading price data...")
    prices_data = download_all_prices()

    # 2. Rebalance dates
    rebalance_dates = compute_rebalance_dates(
        prices_data, n_rebalances=args.n_rebalances)
    print(f"\n{len(rebalance_dates)} rebalance dates: "
          f"{rebalance_dates[0]} → {rebalance_dates[-1]}")

    # 3. Build context windows
    print("\nBuilding context windows...")
    contexts_data = build_context_windows(
        prices_data, rebalance_dates, args.max_context)
    n_windows = len(contexts_data["contexts"])
    print(f"  {n_windows} context windows ({args.n_rebalances} dates × {len(INVESTABLE_TICKERS)} tickers)")

    # 4. Batch inference (3 checkpoints × n_windows)
    print(f"\nBatch inference (3 checkpoints × {n_windows} windows each)...")
    forecast_cache = build_forecast_cache(
        contexts_data, args.horizon, args.max_context)

    # 5. Build allocations for each strategy
    last_prices = contexts_data["last_prices"]  # shape (n_dates, n_tickers)
    allocations = {
        "SP500":          strategy_sp500(rebalance_dates),
        "EqualHold":      strategy_equal_hold(rebalance_dates),
        "EqualRebal":     strategy_equal_rebal(rebalance_dates),
        "ForecastTop5":   strategy_forecast_top5(forecast_cache, "Run4", last_prices),
        "ForecastAllPos": strategy_forecast_allpos(forecast_cache, "Run4", last_prices),
        "SelectiveTop5":  strategy_selective_top5(forecast_cache, last_prices),
    }

    # 6. Simulate all strategies
    print("\nSimulating portfolios...")
    strategy_results = {}
    for name, alloc in allocations.items():
        sim     = simulate_portfolio(alloc, prices_data, rebalance_dates, args.initial_capital)
        metrics = compute_metrics(sim["values"])
        strategy_results[name] = {
            "sim":         sim,
            "metrics":     metrics,
            "allocations": alloc,
        }

    # 7. Print results
    print_results_table(strategy_results)
    for name in ["ForecastTop5", "ForecastAllPos", "SelectiveTop5"]:
        print_allocation_table(name, allocations[name], rebalance_dates)

    # 8. Save outputs
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path   = out / "investment_sim_results.md"
    plot_path = out / "investment_sim_plot.png"
    save_results(strategy_results, rebalance_dates, md_path)
    save_plot(strategy_results, plot_path)
    print(f"\nMarkdown saved to: {md_path}")


if __name__ == "__main__":
    main()
