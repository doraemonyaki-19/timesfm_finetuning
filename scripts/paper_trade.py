"""
Paper Trade Simulator for TimesFM Sector Router.

Simulates the board-approved dual-strategy portfolio with per-cell dead zones,
governance limits, borrow costs, and position tracking.

Usage:
  # Run paper trade with $1M notional
  python scripts/paper_trade.py --notional 1000000

  # Dry run (no model inference, uses random signals for testing)
  python scripts/paper_trade.py --notional 1000000 --dry-run

  # Custom borrow rate (annualized, default 1.5%)
  python scripts/paper_trade.py --notional 1000000 --borrow-rate 0.02
"""
from __future__ import annotations

import argparse
import csv
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

import yfinance as yf
import pandas as pd

# ── Load config ───────────────────────────────────────────────────────────

_CONFIG_PATH = ROOT / "sector_routing_config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

REGIONS = _cfg["regions"]
NO_SHORT_BLOCKLIST = set(_cfg.get("no_short_blocklist", []))
BLOCKED_CELLS = {r: set(hzs) for r, hzs in _cfg.get("blocked_cells", {}).items()}
MAX_CONTEXT = _cfg["max_context"]
FORECAST_CONFIG = _cfg["forecast_config"]
SIMPLE_RULE_FT_MIN_HORIZON = _cfg["simple_rule_ft_min_horizon"]
REGION_POLICY = _cfg["region_policy"]

# ── Board-approved allocation (2026-04-11) ────────────────────────────────

ALLOCATION = [
    # (region, horizon, weight, strategy)
    # strategy: "lsf" = long/short/flat, "lf" = long/flat
    ("europe", 120, 0.25, "lsf"),
    ("japan",  120, 0.20, "lsf"),
    ("europe",  60, 0.15, "lsf"),
    ("china",  120, 0.10, "lf"),
    ("japan",   60, 0.10, "lsf"),
    ("us",      60, 0.10, "lf"),
    # Cash: 10% held in reserve, not allocated
]
CASH_WEIGHT = 0.10

# ── Governance limits ─────────────────────────────────────────────────────

SINGLE_NAME_SHORT_CAP = 0.03      # 3% of notional per short position
STOP_LOSS_PCT = -0.10             # -10% stop-loss on any short
MAX_NET_SHORT_EXPOSURE = 0.40     # 40% of notional max net short

# ── Borrow cost defaults ─────────────────────────────────────────────────

DEFAULT_BORROW_RATE = 0.015       # 1.5% annualized


def routing_table_path(region):
    dirname = "finetune_usa" if region == "us" else f"finetune_{region}"
    return ROOT / dirname / "routing_table.json"


def download_ticker(ticker, years=20):
    end = datetime.datetime.now(datetime.timezone.utc).date()
    start = end - datetime.timedelta(days=365 * years)
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                     progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    close = df["Close"].values.astype(np.float32).flatten()
    mask = ~np.isnan(close) & (close > 0)
    return close[mask]


def load_model(checkpoint_path, max_horizon):
    import timesfm
    import os
    if os.path.isdir(checkpoint_path):
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            checkpoint_path, local_files_only=True)
    else:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint_path)
    model.compile(timesfm.ForecastConfig(
        max_context=MAX_CONTEXT, max_horizon=max_horizon, **FORECAST_CONFIG))
    return model


def get_baseline_path():
    local = ROOT / "model"
    if (local / "model.safetensors").exists():
        return str(local)
    return "google/timesfm-2.5-200m-pytorch"


def forecast_tickers(region, horizon, tickers, dry_run=False):
    """Forecast tickers using the routing table logic.

    Returns dict: ticker -> {model, point_forecast, last_price, predicted_return}
    """
    if dry_run:
        results = {}
        rng = np.random.default_rng(42)
        for t in tickers:
            pred_ret = rng.normal(0, 0.05)
            results[t] = {
                "model": "dry-run",
                "last_price": 100.0,
                "predicted_return": pred_ret,
                "final_price": 100.0 * (1 + pred_ret),
            }
        return results

    # Load routing table
    table_path = routing_table_path(region)
    with open(table_path) as f:
        table = json.load(f)

    policy = REGION_POLICY.get(region, "router")
    h_key = str(horizon)

    # Determine model per ticker
    ft_tickers = []
    bl_tickers = []
    if policy == "simple":
        use_ft = horizon >= SIMPLE_RULE_FT_MIN_HORIZON
        for t in tickers:
            (ft_tickers if use_ft else bl_tickers).append(t)
    else:
        for t in tickers:
            if (t in table["tickers"]
                    and h_key in table["tickers"][t]["horizons"]
                    and table["tickers"][t]["horizons"][h_key]["tier"] == "T1"):
                ft_tickers.append(t)
            else:
                bl_tickers.append(t)

    # Download data
    contexts = {}
    for t in tickers:
        s = download_ticker(t)
        ctx = s[-MAX_CONTEXT:] if len(s) >= MAX_CONTEXT else s
        contexts[t] = ctx

    results = {}

    # Baseline forecasts
    if bl_tickers:
        bl_path = get_baseline_path()
        model = load_model(bl_path, horizon)
        bl_contexts = [contexts[t] for t in bl_tickers]
        pf, _ = model.forecast(horizon=horizon, inputs=bl_contexts)
        del model
        for i, t in enumerate(bl_tickers):
            forecast = np.array(pf[i][:horizon])
            last_price = float(contexts[t][-1])
            final_price = float(forecast[-1])
            results[t] = {
                "model": "baseline",
                "last_price": last_price,
                "predicted_return": (final_price - last_price) / last_price,
                "final_price": final_price,
            }

    # Finetuned forecasts
    if ft_tickers:
        ft_path = table["checkpoint"]
        model = load_model(ft_path, horizon)
        ft_contexts = [contexts[t] for t in ft_tickers]
        pf, _ = model.forecast(horizon=horizon, inputs=ft_contexts)
        del model
        for i, t in enumerate(ft_tickers):
            forecast = np.array(pf[i][:horizon])
            last_price = float(contexts[t][-1])
            final_price = float(forecast[-1])
            results[t] = {
                "model": table["label"],
                "last_price": last_price,
                "predicted_return": (final_price - last_price) / last_price,
                "final_price": final_price,
            }

    return results


def compute_positions(notional, borrow_rate, dry_run=False):
    """Compute the full paper trade portfolio.

    Returns (positions, summary) where positions is a list of dicts.
    """
    today = datetime.date.today()
    positions = []

    # Collect all tickers per cell, run forecasts
    for region, horizon, weight, strategy in ALLOCATION:
        cell_label = f"{region.upper()} {horizon}d"

        # Check blocked cells
        blocked = BLOCKED_CELLS.get(region, set())
        if horizon in blocked:
            print(f"  {cell_label}: BLOCKED by board -- skipping, "
                  f"reallocating {weight*100:.0f}% to cash")
            continue

        # Get tickers: eval + train
        region_cfg = REGIONS[region]
        all_tickers = region_cfg["eval_tickers"] + region_cfg["train_tickers"]

        # Load dead zone from routing table
        table_path = routing_table_path(region)
        dz_threshold = 0.0
        if table_path.exists():
            with open(table_path) as f:
                table = json.load(f)
            odz = table.get("optimal_dead_zones", {})
            h_key = str(horizon)
            if h_key in odz:
                dz_threshold = odz[h_key]["dead_zone"]

        print(f"\n  {cell_label} ({weight*100:.0f}%, {strategy.upper()}, "
              f"DZ={dz_threshold*100:.1f}%):")

        # Forecast
        forecasts = forecast_tickers(region, horizon, all_tickers,
                                     dry_run=dry_run)

        cell_notional = notional * weight
        n_tickers = len(all_tickers)
        per_ticker_notional = cell_notional / n_tickers

        for t in all_tickers:
            fc = forecasts[t]
            pred_ret = fc["predicted_return"]

            # Dead zone -> signal
            if abs(pred_ret) < dz_threshold:
                signal = "FLAT"
            elif pred_ret > 0:
                signal = "LONG"
            else:
                signal = "SHORT"

            # No-short blocklist
            if signal == "SHORT" and t in NO_SHORT_BLOCKLIST:
                signal = "SHORT-BLOCKED"

            # Strategy filter: L/F strategy can't short
            if strategy == "lf" and signal == "SHORT":
                signal = "FLAT"

            # Position sizing
            if signal == "LONG":
                position_usd = per_ticker_notional
            elif signal == "SHORT":
                # Cap at single-name short limit
                raw = per_ticker_notional
                cap = notional * SINGLE_NAME_SHORT_CAP
                position_usd = -min(raw, cap)
            else:
                position_usd = 0.0

            # Borrow cost for shorts (annualized, prorated to horizon)
            borrow_cost = 0.0
            if position_usd < 0:
                days_held = horizon
                borrow_cost = abs(position_usd) * borrow_rate * (days_held / 365)

            positions.append({
                "date": str(today),
                "region": region,
                "horizon": horizon,
                "cell": cell_label,
                "strategy": strategy.upper(),
                "ticker": t,
                "model": fc["model"],
                "last_price": round(fc["last_price"], 2),
                "predicted_return": round(pred_ret * 100, 3),
                "dead_zone": round(dz_threshold * 100, 1),
                "signal": signal,
                "position_usd": round(position_usd, 2),
                "weight_pct": round(weight * 100, 1),
                "borrow_cost": round(borrow_cost, 2),
            })

            tag = ""
            if signal == "SHORT-BLOCKED":
                tag = " [BLOCKLIST]"
            elif signal == "FLAT" and abs(pred_ret) < dz_threshold:
                tag = " [DEAD ZONE]"
            print(f"    {t:>10s}  {signal:>14s}  "
                  f"pred={pred_ret*100:+6.2f}%  "
                  f"${position_usd:>+12,.2f}"
                  f"{'  borrow=$'+str(round(borrow_cost,2)) if borrow_cost > 0 else ''}"
                  f"{tag}")

    return positions


def main():
    parser = argparse.ArgumentParser(
        description="Paper Trade Simulator for TimesFM Sector Router")
    parser.add_argument("--notional", type=float, default=1_000_000,
                        help="Portfolio notional in USD (default: 1,000,000)")
    parser.add_argument("--borrow-rate", type=float, default=DEFAULT_BORROW_RATE,
                        help=f"Annualized borrow rate for shorts "
                             f"(default: {DEFAULT_BORROW_RATE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use random signals instead of model inference")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: paper_trade_YYYY-MM-DD.csv)")
    args = parser.parse_args()

    today = datetime.date.today()
    notional = args.notional
    borrow_rate = args.borrow_rate

    print("=" * 70)
    print("PAPER TRADE SIMULATOR")
    print(f"Date: {today}")
    print(f"Notional: ${notional:,.2f}")
    print(f"Cash reserve: {CASH_WEIGHT*100:.0f}% (${notional * CASH_WEIGHT:,.2f})")
    print(f"Borrow rate: {borrow_rate*100:.2f}% annualized")
    if args.dry_run:
        print("MODE: DRY RUN (random signals)")
    print("=" * 70)

    print("\nAllocation:")
    for region, horizon, weight, strategy in ALLOCATION:
        blocked = BLOCKED_CELLS.get(region, set())
        status = " [BLOCKED]" if horizon in blocked else ""
        print(f"  {region.upper():>8s} {horizon:>3d}d  "
              f"{weight*100:>5.1f}%  {strategy.upper():>3s}{status}")
    print(f"  {'CASH':>8s}       {CASH_WEIGHT*100:>5.1f}%")

    print("\nGenerating positions...")
    positions = compute_positions(notional, borrow_rate, dry_run=args.dry_run)

    # ── Portfolio summary ─────────────────────────────────────────────────

    n_long = sum(1 for p in positions if p["signal"] == "LONG")
    n_short = sum(1 for p in positions if p["signal"] == "SHORT")
    n_flat = sum(1 for p in positions if p["signal"] in ("FLAT", "SHORT-BLOCKED"))
    total_long = sum(p["position_usd"] for p in positions if p["position_usd"] > 0)
    total_short = sum(p["position_usd"] for p in positions if p["position_usd"] < 0)
    total_borrow = sum(p["borrow_cost"] for p in positions)
    net_exposure = total_long + total_short
    gross_exposure = total_long + abs(total_short)
    net_short_pct = abs(total_short) / notional * 100

    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"  Positions: {n_long} long, {n_short} short, {n_flat} flat")
    print(f"  Long exposure:   ${total_long:>12,.2f}  "
          f"({total_long/notional*100:.1f}%)")
    print(f"  Short exposure:  ${total_short:>12,.2f}  "
          f"({abs(total_short)/notional*100:.1f}%)")
    print(f"  Net exposure:    ${net_exposure:>12,.2f}  "
          f"({net_exposure/notional*100:.1f}%)")
    print(f"  Gross exposure:  ${gross_exposure:>12,.2f}  "
          f"({gross_exposure/notional*100:.1f}%)")
    print(f"  Cash:            ${notional * CASH_WEIGHT:>12,.2f}  "
          f"({CASH_WEIGHT*100:.1f}%)")
    print(f"  Total borrow:    ${total_borrow:>12,.2f}")

    # Governance checks
    print("\n  Governance Checks:")
    violations = []

    # Single-name short cap
    for p in positions:
        if p["position_usd"] < 0:
            pct = abs(p["position_usd"]) / notional
            if pct > SINGLE_NAME_SHORT_CAP + 0.001:
                violations.append(
                    f"    FAIL: {p['ticker']} short "
                    f"${abs(p['position_usd']):,.2f} = "
                    f"{pct*100:.1f}% > {SINGLE_NAME_SHORT_CAP*100:.0f}% cap")

    # Net short exposure
    if net_short_pct > MAX_NET_SHORT_EXPOSURE * 100:
        violations.append(
            f"    FAIL: Net short {net_short_pct:.1f}% > "
            f"{MAX_NET_SHORT_EXPOSURE*100:.0f}% limit")

    if violations:
        for v in violations:
            print(v)
    else:
        print("    PASS: Single-name short cap (3%)")
        print(f"    PASS: Net short exposure ({net_short_pct:.1f}% <= "
              f"{MAX_NET_SHORT_EXPOSURE*100:.0f}%)")
        print("    PASS: No-short blocklist enforced")

    # ── Per-cell breakdown ────────────────────────────────────────────────

    print("\n  Per-Cell Breakdown:")
    cells = {}
    for p in positions:
        cell = p["cell"]
        if cell not in cells:
            cells[cell] = {"long": 0, "short": 0, "flat": 0,
                           "long_usd": 0, "short_usd": 0, "borrow": 0}
        if p["signal"] == "LONG":
            cells[cell]["long"] += 1
            cells[cell]["long_usd"] += p["position_usd"]
        elif p["signal"] == "SHORT":
            cells[cell]["short"] += 1
            cells[cell]["short_usd"] += p["position_usd"]
        else:
            cells[cell]["flat"] += 1
        cells[cell]["borrow"] += p["borrow_cost"]

    for cell, c in cells.items():
        net = c["long_usd"] + c["short_usd"]
        print(f"    {cell:>15s}: {c['long']}L {c['short']}S {c['flat']}F  "
              f"net=${net:>+10,.2f}  borrow=${c['borrow']:>6,.2f}")

    # ── Save CSV ──────────────────────────────────────────────────────────

    out_path = args.output or str(ROOT / f"paper_trade_{today}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "region", "horizon", "cell", "strategy", "ticker",
            "model", "last_price", "predicted_return", "dead_zone",
            "signal", "position_usd", "weight_pct", "borrow_cost",
        ])
        writer.writeheader()
        writer.writerows(positions)

    print(f"\nPositions saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
