"""
Paper Trade Simulator for TimesFM Sector Router.

Daily lifecycle:
  1. Load state (open positions, cash balance)
  2. Download today's prices for all open positions
  3. Mark-to-market: update unrealized P&L
  4. Enforce stop-losses: close shorts that hit -10%
  5. Close expired positions (horizon reached)
  6. Open new positions for cells that have no open positions
  7. Save state + append to journal

Commands:
  # Initialize a new paper trade with $1M
  python board_recommended_test/paper_trade.py init --notional 1000000

  # Daily run (MTM, stop-losses, expiry, rebalance)
  python board_recommended_test/paper_trade.py run

  # Show current portfolio status
  python board_recommended_test/paper_trade.py status

  # Show closed-trade journal summary
  python board_recommended_test/paper_trade.py history

  # Dry run (random prices, no model inference)
  python board_recommended_test/paper_trade.py run --dry-run
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TRADE_DIR = Path(__file__).resolve().parent
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
    {"region": "europe", "horizon": 120, "weight": 0.25, "strategy": "lsf"},
    {"region": "japan",  "horizon": 120, "weight": 0.20, "strategy": "lsf"},
    {"region": "europe", "horizon":  60, "weight": 0.15, "strategy": "lsf"},
    {"region": "china",  "horizon": 120, "weight": 0.10, "strategy": "lf"},
    {"region": "japan",  "horizon":  60, "weight": 0.10, "strategy": "lsf"},
    {"region": "us",     "horizon":  60, "weight": 0.10, "strategy": "lf"},
]
CASH_WEIGHT = 0.10

# ── Governance limits ─────────────────────────────────────────────────────

SINGLE_NAME_SHORT_CAP = 0.03      # 3% of notional per short position
STOP_LOSS_PCT = -0.10             # -10% on short position -> close
MAX_NET_SHORT_EXPOSURE = 0.40     # 40% of notional max net short

# Cell-level stop (board resolution 2026-09-08): if a cell's current cohort
# reaches this realized + unrealized P&L (net of borrow) as a fraction of cell
# capital, flatten the whole cell to cash until its next scheduled rebalance.
CELL_STOP_LOSS_PCT = {
    "europe_60": -0.075,          # -$11,250 on the $150k Europe 60d cell
}

DEFAULT_BORROW_RATE = 0.015       # 1.5% annualized

# ── File paths ────────────────────────────────────────────────────────────

STATE_FILE = TRADE_DIR / "state.json"
JOURNAL_FILE = TRADE_DIR / "journal.csv"

JOURNAL_FIELDS = [
    "trade_id", "open_date", "close_date", "region", "horizon", "cell",
    "strategy", "ticker", "model", "signal", "entry_price", "exit_price",
    "position_usd", "realized_pnl", "borrow_cost", "close_reason",
    "days_held",
]


# ── Helpers ───────────────────────────────────────────────────────────────

def cell_key(region, horizon):
    return f"{region}_{horizon}"


def cell_label(region, horizon):
    return f"{region.upper()} {horizon}d"


def routing_table_path(region):
    dirname = "finetune_usa" if region == "us" else f"finetune_{region}"
    return ROOT / dirname / "routing_table.json"


def get_dead_zone(region, horizon):
    table_path = routing_table_path(region)
    if table_path.exists():
        with open(table_path) as f:
            table = json.load(f)
        odz = table.get("optimal_dead_zones", {})
        h_key = str(horizon)
        if h_key in odz:
            return odz[h_key]["dead_zone"]
    return 0.0


def download_price(ticker):
    """Download latest close price for a single ticker."""
    end = datetime.datetime.now(datetime.timezone.utc).date()
    start = end - datetime.timedelta(days=10)
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                     progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel("Ticker", axis=1)
    close = df["Close"].dropna().values.astype(np.float32).flatten()
    if len(close) == 0:
        return None
    return float(close[-1])


def _extract_last_closes(df, tickers):
    """Pull the latest non-NaN close per ticker out of a yfinance frame.

    Handles both the multi-ticker MultiIndex layout (field, ticker) and the
    single-ticker flat layout. Tickers with no usable data are omitted so the
    caller can tell fresh prices from stale ones.
    """
    out = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return out
        close = df["Close"]
        for t in tickers:
            if t in close.columns:
                s = close[t].dropna()
                if len(s):
                    out[t] = float(s.iloc[-1])
    else:
        if "Close" in df.columns and len(tickers) == 1:
            s = df["Close"].dropna()
            if len(s):
                out[tickers[0]] = float(s.iloc[-1])
    return out


def download_prices_batch(all_tickers, retries=3, backoff=2.0):
    """Download latest close prices for many tickers in a single request.

    Uses one batched yf.download call instead of one-per-ticker (the sequential
    pattern was rate-limited into silent failure). Retries with backoff, and on
    partial success only re-requests the tickers still missing. Returns a dict
    ticker -> latest close; failed tickers are simply absent from the dict.
    """
    end = datetime.datetime.now(datetime.timezone.utc).date()
    start = end - datetime.timedelta(days=10)
    pending = list(dict.fromkeys(all_tickers))  # de-dup, preserve order
    prices = {}
    for attempt in range(retries):
        if not pending:
            break
        try:
            df = yf.download(pending, start=start.isoformat(),
                             end=end.isoformat(), progress=False,
                             group_by="column", threads=True)
        except Exception as e:  # network / parsing failure -> retry
            print(f"    (batch download attempt {attempt + 1} failed: {e})")
            df = None
        if df is not None:
            prices.update(_extract_last_closes(df, pending))
            pending = [t for t in pending if t not in prices]
        if pending and attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    return prices


def download_context(ticker, years=20):
    """Download full history for forecasting."""
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
    cp = str(checkpoint_path)
    if not os.path.isdir(cp):
        alt_path = cp.replace(r"workspace\timesfm\finetune_", r"workspace\timesfm_finetuning\finetune_").replace("workspace/timesfm/finetune_", "workspace/timesfm_finetuning/finetune_")
        if os.path.isdir(alt_path):
            cp = alt_path
        elif (ROOT / cp).is_dir():
            cp = str(ROOT / cp)
    if os.path.isdir(cp):
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            cp, local_files_only=True)
    else:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(cp)
    model.compile(timesfm.ForecastConfig(
        max_context=MAX_CONTEXT, max_horizon=max_horizon, **FORECAST_CONFIG))
    return model


def get_baseline_path():
    local = ROOT / "model"
    if (local / "model.safetensors").exists():
        return str(local)
    return "google/timesfm-2.5-200m-pytorch"


def load_state():
    if not STATE_FILE.exists():
        return None
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def append_journal(trades):
    """Append closed trades to the journal CSV."""
    write_header = not JOURNAL_FILE.exists()
    with open(JOURNAL_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if write_header:
            writer.writeheader()
        for t in trades:
            writer.writerow(t)


def cell_cohort_pnl(ck, cohort_open, open_cohort, closed_this_run):
    """Net P&L (after borrow) of a cell's current cohort.

    The cohort is every position in cell `ck` opened on or after `cohort_open`:
    trades already closed in the journal (e.g. earlier position stop-outs),
    trades closed earlier in this run, and the still-open positions.
    """
    def in_cohort(t):
        return (cell_key(t["region"], int(t["horizon"])) == ck
                and t["open_date"] >= cohort_open)

    closed = [t for t in closed_this_run if in_cohort(t)]
    if JOURNAL_FILE.exists():
        with open(JOURNAL_FILE, encoding="utf-8") as f:
            closed += [t for t in csv.DictReader(f) if in_cohort(t)]

    pnl = sum(float(t["realized_pnl"]) - float(t["borrow_cost"])
              for t in closed)
    pnl += sum(p.get("unrealized_pnl", 0) - p.get("accrued_borrow", 0)
               for p in open_cohort)
    return pnl


def next_trade_id(state):
    state["_next_id"] = state.get("_next_id", 0) + 1
    return state["_next_id"]


def trading_days_between(d1_str, d2_str):
    """Approximate trading days between two date strings."""
    d1 = datetime.date.fromisoformat(d1_str)
    d2 = datetime.date.fromisoformat(d2_str)
    # Count weekdays
    days = 0
    d = d1
    while d < d2:
        if d.weekday() < 5:
            days += 1
        d += datetime.timedelta(days=1)
    return days


# ── Forecast ──────────────────────────────────────────────────────────────

def forecast_cell(region, horizon, tickers, dry_run=False):
    """Run forecasts for a set of tickers.

    Returns dict: ticker -> {model, last_price, predicted_return}
    """
    if dry_run:
        rng = np.random.default_rng(int(datetime.date.today().toordinal()))
        results = {}
        for t in tickers:
            pred_ret = rng.normal(0, 0.05)
            results[t] = {
                "model": "dry-run",
                "last_price": 100.0 + rng.normal(0, 10),
                "predicted_return": pred_ret,
            }
        return results

    table_path = routing_table_path(region)
    with open(table_path) as f:
        table = json.load(f)

    policy = REGION_POLICY.get(region, "router")
    h_key = str(horizon)

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

    contexts = {}
    for t in tickers:
        s = download_context(t)
        ctx = s[-MAX_CONTEXT:] if len(s) >= MAX_CONTEXT else s
        contexts[t] = ctx

    results = {}

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
            }

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
            }

    return results


# ── Commands ──────────────────────────────────────────────────────────────

def cmd_init(args):
    """Initialize a fresh paper trade."""
    if STATE_FILE.exists() and not args.force:
        print(f"State file already exists: {STATE_FILE}")
        print("Use --force to overwrite.")
        sys.exit(1)

    today = str(datetime.date.today())
    notional = args.notional
    borrow_rate = args.borrow_rate

    state = {
        "created": today,
        "notional": notional,
        "borrow_rate": borrow_rate,
        "cash": notional * CASH_WEIGHT,
        "open_positions": [],
        "cells": {},
        "_next_id": 0,
    }

    # Initialize cell state: each cell is "ready" to open on first run
    for alloc in ALLOCATION:
        ck = cell_key(alloc["region"], alloc["horizon"])
        blocked = BLOCKED_CELLS.get(alloc["region"], set())
        state["cells"][ck] = {
            "region": alloc["region"],
            "horizon": alloc["horizon"],
            "weight": alloc["weight"],
            "strategy": alloc["strategy"],
            "blocked": alloc["horizon"] in blocked,
            "next_rebalance": today,
        }

    save_state(state)

    # Reset journal
    if JOURNAL_FILE.exists() and args.force:
        JOURNAL_FILE.unlink()

    print("=" * 70)
    print("PAPER TRADE INITIALIZED")
    print(f"  Date:     {today}")
    print(f"  Notional: ${notional:,.2f}")
    print(f"  Cash:     ${state['cash']:,.2f} ({CASH_WEIGHT*100:.0f}%)")
    print(f"  Borrow:   {borrow_rate*100:.2f}% annualized")
    print(f"  State:    {STATE_FILE}")
    print(f"  Journal:  {JOURNAL_FILE}")
    print("=" * 70)

    print("\n  Cells:")
    for ck, cell in state["cells"].items():
        status = "BLOCKED" if cell["blocked"] else "READY"
        print(f"    {cell_label(cell['region'], cell['horizon']):>15s}  "
              f"{cell['weight']*100:>5.1f}%  {cell['strategy'].upper():>3s}  "
              f"[{status}]")
    print(f"\n  Run 'paper_trade.py run' to open initial positions.")


def cmd_run(args):
    """Daily run: MTM, stop-losses, expiry, rebalance."""
    state = load_state()
    if state is None:
        print("No state file found. Run 'init' first.")
        sys.exit(1)

    today = str(datetime.date.today())
    notional = state["notional"]
    borrow_rate = state["borrow_rate"]
    dry_run = args.dry_run

    print("=" * 70)
    print(f"PAPER TRADE -- DAILY RUN ({today})")
    if dry_run:
        print("  MODE: DRY RUN (random prices)")
    print("=" * 70)

    closed_trades = []
    stale_tickers = []

    # ── Step 1: Mark-to-market open positions ─────────────────────────────

    open_pos = state["open_positions"]
    if open_pos:
        print(f"\n  Mark-to-Market ({len(open_pos)} open positions):")

        # Batch download prices
        all_tickers = list(set(p["ticker"] for p in open_pos))
        prices = {}
        if dry_run:
            rng = np.random.default_rng(int(datetime.date.today().toordinal()) + 1)
            for t in all_tickers:
                # Simulate price movement from entry
                prices[t] = 100.0 + rng.normal(0, 10)
        else:
            print("    Downloading prices...", flush=True)
            prices = download_prices_batch(all_tickers)
            missing = sorted(t for t in all_tickers if t not in prices)
            stale_tickers = missing
            if not prices:
                # Every download failed. Do NOT advance the lifecycle on stale
                # prices -- leave state untouched so the failure stays visible.
                print(f"    ERROR: all {len(all_tickers)} price downloads failed"
                      " -- aborting run, state unchanged (no stale MTM/expiry).")
                sys.exit(1)
            if missing:
                print(f"    WARNING: no fresh price for {len(missing)}/"
                      f"{len(all_tickers)} tickers: {', '.join(missing)}")
                print("    -> these positions keep last-known price; "
                      "MTM / borrow / expiry NOT advanced for them")

        for pos in open_pos:
            t = pos["ticker"]
            if t in prices:
                pos["current_price"] = prices[t]
                entry = pos["entry_price"]
                if pos["signal"] == "LONG":
                    pos["unrealized_pnl"] = (prices[t] - entry) / entry * pos["position_usd"]
                elif pos["signal"] == "SHORT":
                    # Short P&L: profit when price goes down
                    pos["unrealized_pnl"] = (entry - prices[t]) / entry * abs(pos["position_usd"])
                else:
                    pos["unrealized_pnl"] = 0.0

                days = trading_days_between(pos["open_date"], today)
                pos["days_held"] = days

                # Accrue borrow cost for shorts
                if pos["signal"] == "SHORT":
                    pos["accrued_borrow"] = (
                        abs(pos["position_usd"]) * borrow_rate * (days / 365))
                else:
                    pos["accrued_borrow"] = 0.0

    # ── Step 2: Enforce stop-losses on shorts ─────────────────────────────

    still_open = []
    stopped_count = 0
    for pos in open_pos:
        if pos["signal"] == "SHORT" and pos.get("unrealized_pnl", 0) != 0:
            pnl_pct = pos["unrealized_pnl"] / abs(pos["position_usd"])
            if pnl_pct <= STOP_LOSS_PCT:
                # Stop-loss triggered
                stopped_count += 1
                trade = {
                    "trade_id": pos["trade_id"],
                    "open_date": pos["open_date"],
                    "close_date": today,
                    "region": pos["region"],
                    "horizon": pos["horizon"],
                    "cell": pos["cell"],
                    "strategy": pos["strategy"],
                    "ticker": pos["ticker"],
                    "model": pos["model"],
                    "signal": pos["signal"],
                    "entry_price": pos["entry_price"],
                    "exit_price": pos.get("current_price", pos["entry_price"]),
                    "position_usd": pos["position_usd"],
                    "realized_pnl": round(pos["unrealized_pnl"], 2),
                    "borrow_cost": round(pos.get("accrued_borrow", 0), 2),
                    "close_reason": "STOP-LOSS",
                    "days_held": pos.get("days_held", 0),
                }
                closed_trades.append(trade)
                print(f"    STOP-LOSS: {pos['ticker']} "
                      f"P&L=${pos['unrealized_pnl']:+,.2f} "
                      f"({pnl_pct*100:+.1f}%)")
                continue
        still_open.append(pos)

    if stopped_count:
        print(f"    {stopped_count} position(s) stopped out")

    # ── Step 2b: Enforce cell-level stop-losses ───────────────────────────

    for ck, stop_pct in CELL_STOP_LOSS_PCT.items():
        cell = state["cells"].get(ck)
        if cell is None:
            continue
        cohort = [p for p in still_open
                  if cell_key(p["region"], p["horizon"]) == ck]
        if not cohort:
            continue
        cohort_open = min(p["open_date"] for p in cohort)
        cell_capital = notional * cell["weight"]
        limit = stop_pct * cell_capital
        cohort_pnl = cell_cohort_pnl(ck, cohort_open, cohort, closed_trades)
        label = cell_label(cell["region"], cell["horizon"])
        print(f"    Cell stop {label}: cohort P&L ${cohort_pnl:+,.2f} "
              f"(limit ${limit:,.2f})")
        if cohort_pnl > limit:
            continue

        # Cell stop triggered: flatten every remaining position in the cell
        print(f"    CELL STOP-LOSS: {label} cohort P&L ${cohort_pnl:+,.2f} "
              f"<= ${limit:,.2f} -- flattening {len(cohort)} position(s) "
              f"until {cell.get('next_rebalance', '?')}")
        for pos in cohort:
            closed_trades.append({
                "trade_id": pos["trade_id"],
                "open_date": pos["open_date"],
                "close_date": today,
                "region": pos["region"],
                "horizon": pos["horizon"],
                "cell": pos["cell"],
                "strategy": pos["strategy"],
                "ticker": pos["ticker"],
                "model": pos["model"],
                "signal": pos["signal"],
                "entry_price": pos["entry_price"],
                "exit_price": pos.get("current_price", pos["entry_price"]),
                "position_usd": pos["position_usd"],
                "realized_pnl": round(pos.get("unrealized_pnl", 0), 2),
                "borrow_cost": round(pos.get("accrued_borrow", 0), 2),
                "close_reason": "CELL-STOP",
                "days_held": pos.get("days_held", 0),
            })
            print(f"      closed {pos['ticker']} "
                  f"P&L=${pos.get('unrealized_pnl', 0):+,.2f}")
        cell["cell_stopped"] = today
        still_open = [p for p in still_open if p not in cohort]

    # ── Step 3: Close expired positions ───────────────────────────────────

    newly_open = []
    expired_count = 0
    for pos in still_open:
        days = pos.get("days_held", 0)
        if days >= pos["horizon"]:
            expired_count += 1
            trade = {
                "trade_id": pos["trade_id"],
                "open_date": pos["open_date"],
                "close_date": today,
                "region": pos["region"],
                "horizon": pos["horizon"],
                "cell": pos["cell"],
                "strategy": pos["strategy"],
                "ticker": pos["ticker"],
                "model": pos["model"],
                "signal": pos["signal"],
                "entry_price": pos["entry_price"],
                "exit_price": pos.get("current_price", pos["entry_price"]),
                "position_usd": pos["position_usd"],
                "realized_pnl": round(pos.get("unrealized_pnl", 0), 2),
                "borrow_cost": round(pos.get("accrued_borrow", 0), 2),
                "close_reason": "EXPIRED",
                "days_held": days,
            }
            closed_trades.append(trade)
            print(f"    EXPIRED: {pos['ticker']} ({pos['cell']}) "
                  f"{days}d held, P&L=${pos.get('unrealized_pnl', 0):+,.2f}")
        else:
            newly_open.append(pos)

    if expired_count:
        print(f"    {expired_count} position(s) expired")

    state["open_positions"] = newly_open

    # ── Step 4: Open new positions for empty cells ────────────────────────

    # Determine which cells have open positions
    active_cells = set()
    for pos in state["open_positions"]:
        active_cells.add(cell_key(pos["region"], pos["horizon"]))

    opened_count = 0
    for ck, cell in state["cells"].items():
        if cell["blocked"]:
            continue
        if ck in active_cells:
            continue  # Cell still has open positions

        # Check if it's time to rebalance
        next_reb = cell.get("next_rebalance", today)
        if next_reb > today:
            continue

        region = cell["region"]
        horizon = cell["horizon"]
        weight = cell["weight"]
        strategy = cell["strategy"]

        print(f"\n  Opening: {cell_label(region, horizon)} "
              f"({weight*100:.0f}%, {strategy.upper()}):")

        region_cfg = REGIONS[region]
        all_tickers = region_cfg["eval_tickers"] + region_cfg["train_tickers"]
        dz_threshold = get_dead_zone(region, horizon)

        print(f"    Dead zone: {dz_threshold*100:.1f}%")

        forecasts = forecast_cell(region, horizon, all_tickers, dry_run=dry_run)

        cell_notional = notional * weight
        n_tickers = len(all_tickers)
        per_ticker = cell_notional / n_tickers

        for t in all_tickers:
            fc = forecasts[t]
            pred_ret = fc["predicted_return"]

            # Signal determination
            if abs(pred_ret) < dz_threshold:
                signal = "FLAT"
            elif pred_ret > 0:
                signal = "LONG"
            else:
                signal = "SHORT"

            if signal == "SHORT" and t in NO_SHORT_BLOCKLIST:
                signal = "SHORT-BLOCKED"
            if strategy == "lf" and signal == "SHORT":
                signal = "FLAT"

            if signal in ("FLAT", "SHORT-BLOCKED"):
                continue  # No position

            # Size
            if signal == "LONG":
                position_usd = per_ticker
            else:  # SHORT
                cap = notional * SINGLE_NAME_SHORT_CAP
                position_usd = -min(per_ticker, cap)

            tid = next_trade_id(state)
            pos = {
                "trade_id": tid,
                "open_date": today,
                "region": region,
                "horizon": horizon,
                "cell": cell_label(region, horizon),
                "strategy": strategy.upper(),
                "ticker": t,
                "model": fc["model"],
                "signal": signal,
                "entry_price": round(fc["last_price"], 4),
                "position_usd": round(position_usd, 2),
                "predicted_return": round(pred_ret * 100, 3),
                "current_price": round(fc["last_price"], 4),
                "unrealized_pnl": 0.0,
                "accrued_borrow": 0.0,
                "days_held": 0,
            }
            state["open_positions"].append(pos)
            opened_count += 1

            print(f"    {t:>10s}  {signal:>6s}  pred={pred_ret*100:+6.2f}%  "
                  f"${position_usd:>+12,.2f}  @{fc['last_price']:.2f}")

        # Set next rebalance = today + horizon calendar days
        next_date = (datetime.date.fromisoformat(today)
                     + datetime.timedelta(days=int(horizon * 1.5)))
        cell["next_rebalance"] = str(next_date)
        cell.pop("cell_stopped", None)

    if opened_count:
        print(f"\n    {opened_count} new position(s) opened")

    # ── Step 5: Write journal and save state ──────────────────────────────

    if closed_trades:
        append_journal(closed_trades)
        print(f"\n  Journal: {len(closed_trades)} trade(s) closed, "
              f"appended to {JOURNAL_FILE}")

    state["last_run"] = today
    save_state(state)

    # ── Summary ───────────────────────────────────────────────────────────

    _print_summary(state)

    # ── Optional Markdown report ──────────────────────────────────────────

    if getattr(args, "report", False):
        path = generate_report(state, closed_trades, stale_tickers, today)
        print(f"\n  MTM report written to: {path}")


def cmd_status(args):
    """Show current portfolio status."""
    state = load_state()
    if state is None:
        print("No state file found. Run 'init' first.")
        sys.exit(1)

    print("=" * 70)
    print(f"PAPER TRADE STATUS (last run: {state.get('last_run', 'never')})")
    print("=" * 70)

    _print_summary(state)

    # Per-position detail
    open_pos = state["open_positions"]
    if open_pos:
        print("\n  Open Positions:")
        print(f"    {'Ticker':>10s} {'Cell':>15s} {'Signal':>6s} "
              f"{'Entry':>8s} {'Current':>8s} {'Unreal P&L':>12s} "
              f"{'Days':>5s} {'Borrow':>8s}")
        print("    " + "-" * 80)
        for pos in sorted(open_pos, key=lambda p: (p["cell"], p["ticker"])):
            print(f"    {pos['ticker']:>10s} {pos['cell']:>15s} "
                  f"{pos['signal']:>6s} "
                  f"{pos['entry_price']:>8.2f} "
                  f"{pos.get('current_price', 0):>8.2f} "
                  f"${pos.get('unrealized_pnl', 0):>+10,.2f} "
                  f"{pos.get('days_held', 0):>5d} "
                  f"${pos.get('accrued_borrow', 0):>7,.2f}")


def cmd_history(args):
    """Show closed-trade journal summary."""
    if not JOURNAL_FILE.exists():
        print("No journal file found. No trades closed yet.")
        return

    with open(JOURNAL_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        trades = list(reader)

    if not trades:
        print("Journal is empty.")
        return

    print("=" * 70)
    print(f"TRADE JOURNAL ({len(trades)} closed trades)")
    print("=" * 70)

    total_pnl = 0.0
    total_borrow = 0.0
    by_cell = {}
    by_reason = {}

    for t in trades:
        pnl = float(t["realized_pnl"])
        borrow = float(t["borrow_cost"])
        total_pnl += pnl
        total_borrow += borrow

        cell = t["cell"]
        if cell not in by_cell:
            by_cell[cell] = {"count": 0, "pnl": 0, "borrow": 0,
                             "wins": 0, "losses": 0}
        by_cell[cell]["count"] += 1
        by_cell[cell]["pnl"] += pnl
        by_cell[cell]["borrow"] += borrow
        if pnl > 0:
            by_cell[cell]["wins"] += 1
        elif pnl < 0:
            by_cell[cell]["losses"] += 1

        reason = t["close_reason"]
        by_reason[reason] = by_reason.get(reason, 0) + 1

    print(f"\n  Total realized P&L:  ${total_pnl:>+12,.2f}")
    print(f"  Total borrow cost:   ${total_borrow:>12,.2f}")
    print(f"  Net P&L:             ${total_pnl - total_borrow:>+12,.2f}")

    print(f"\n  Close reasons: ", end="")
    print(", ".join(f"{r}={n}" for r, n in sorted(by_reason.items())))

    print(f"\n  Per-Cell:")
    print(f"    {'Cell':>15s} {'Trades':>6s} {'W':>3s} {'L':>3s} "
          f"{'P&L':>12s} {'Borrow':>8s} {'Net':>12s}")
    print("    " + "-" * 65)
    for cell in sorted(by_cell.keys()):
        c = by_cell[cell]
        net = c["pnl"] - c["borrow"]
        print(f"    {cell:>15s} {c['count']:>6d} {c['wins']:>3d} "
              f"{c['losses']:>3d} ${c['pnl']:>+10,.2f} "
              f"${c['borrow']:>7,.2f} ${net:>+10,.2f}")

    # Last 10 trades
    print(f"\n  Last 10 trades:")
    print(f"    {'Date':>10s} {'Ticker':>10s} {'Signal':>6s} "
          f"{'P&L':>10s} {'Reason':>10s}")
    for t in trades[-10:]:
        print(f"    {t['close_date']:>10s} {t['ticker']:>10s} "
              f"{t['signal']:>6s} ${float(t['realized_pnl']):>+8,.2f} "
              f"{t['close_reason']:>10s}")


def _print_summary(state):
    """Print portfolio summary from state."""
    notional = state["notional"]
    open_pos = state["open_positions"]

    n_long = sum(1 for p in open_pos if p["signal"] == "LONG")
    n_short = sum(1 for p in open_pos if p["signal"] == "SHORT")
    total_long = sum(p["position_usd"] for p in open_pos if p["position_usd"] > 0)
    total_short = sum(p["position_usd"] for p in open_pos if p["position_usd"] < 0)
    total_unreal = sum(p.get("unrealized_pnl", 0) for p in open_pos)
    total_borrow = sum(p.get("accrued_borrow", 0) for p in open_pos)
    net_exposure = total_long + total_short
    net_short_pct = abs(total_short) / notional * 100 if notional else 0

    print(f"\n  Notional:        ${notional:>12,.2f}")
    print(f"  Open positions:  {n_long} long, {n_short} short")
    print(f"  Long exposure:   ${total_long:>12,.2f}  "
          f"({total_long/notional*100:.1f}%)" if notional else "")
    print(f"  Short exposure:  ${total_short:>12,.2f}  "
          f"({abs(total_short)/notional*100:.1f}%)" if notional else "")
    print(f"  Net exposure:    ${net_exposure:>12,.2f}  "
          f"({net_exposure/notional*100:.1f}%)" if notional else "")
    print(f"  Cash:            ${state['cash']:>12,.2f}")
    print(f"  Unrealized P&L:  ${total_unreal:>+12,.2f}")
    print(f"  Accrued borrow:  ${total_borrow:>12,.2f}")

    # Governance
    print("\n  Governance:")
    ok = True
    for p in open_pos:
        if p["position_usd"] < 0:
            pct = abs(p["position_usd"]) / notional
            if pct > SINGLE_NAME_SHORT_CAP + 0.001:
                print(f"    FAIL: {p['ticker']} short {pct*100:.1f}% "
                      f"> {SINGLE_NAME_SHORT_CAP*100:.0f}% cap")
                ok = False
    if net_short_pct > MAX_NET_SHORT_EXPOSURE * 100:
        print(f"    FAIL: Net short {net_short_pct:.1f}% "
              f"> {MAX_NET_SHORT_EXPOSURE*100:.0f}% limit")
        ok = False
    if ok:
        print(f"    PASS: Short cap ({SINGLE_NAME_SHORT_CAP*100:.0f}%), "
              f"net short ({net_short_pct:.1f}% <= "
              f"{MAX_NET_SHORT_EXPOSURE*100:.0f}%), blocklist")

    # Cell status
    print("\n  Cell Status:")
    active_cells = {}
    for p in open_pos:
        ck = cell_key(p["region"], p["horizon"])
        if ck not in active_cells:
            active_cells[ck] = 0
        active_cells[ck] += 1

    for ck, cell in state.get("cells", {}).items():
        n = active_cells.get(ck, 0)
        if cell["blocked"]:
            status = "BLOCKED"
        elif n > 0:
            status = f"{n} open"
        elif cell.get("cell_stopped"):
            status = (f"CELL-STOPPED {cell['cell_stopped']}, "
                      f"rebal {cell.get('next_rebalance', '?')}")
        else:
            status = f"empty, rebal {cell.get('next_rebalance', '?')}"
        print(f"    {cell_label(cell['region'], cell['horizon']):>15s}  "
              f"{cell['weight']*100:>5.1f}%  {cell['strategy'].upper():>3s}  "
              f"[{status}]")


# ── Report ────────────────────────────────────────────────────────────────

def _governance_ok(state):
    notional = state["notional"]
    open_pos = state["open_positions"]
    total_short = sum(p["position_usd"] for p in open_pos if p["position_usd"] < 0)
    net_short_pct = abs(total_short) / notional * 100 if notional else 0
    for p in open_pos:
        if p["position_usd"] < 0:
            if abs(p["position_usd"]) / notional > SINGLE_NAME_SHORT_CAP + 0.001:
                return False
    return net_short_pct <= MAX_NET_SHORT_EXPOSURE * 100


def generate_report(state, closed_this_run, stale_tickers, today):
    """Write mtm_{today}.md: factual tables auto-filled from state + journal,
    narrative sections stubbed as TODO for hand-editing. Returns the path.

    The dated Markdown reports were previously hand-written; this emits the
    same skeleton so a report exists for every run without manual bookkeeping.
    """
    notional = state["notional"]
    open_pos = state["open_positions"]

    # Realized P&L across the full journal history
    realized = 0.0
    n_closed_all = 0
    if JOURNAL_FILE.exists():
        with open(JOURNAL_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                realized += float(row["realized_pnl"])
                n_closed_all += 1

    unreal = sum(p.get("unrealized_pnl", 0) for p in open_pos)
    accrued = sum(p.get("accrued_borrow", 0) for p in open_pos)
    net = realized + unreal - accrued
    net_pct = net / notional * 100 if notional else 0

    total_long = sum(p["position_usd"] for p in open_pos if p["position_usd"] > 0)
    total_short = sum(p["position_usd"] for p in open_pos if p["position_usd"] < 0)
    net_exp = total_long + total_short
    days = max((p.get("days_held", 0) for p in open_pos), default=0)

    # Per-cell unrealized, sorted best-to-worst
    cells = {}
    for p in open_pos:
        c = cells.setdefault(p["cell"], {"pnl": 0.0, "n": 0, "L": 0, "S": 0})
        c["pnl"] += p.get("unrealized_pnl", 0)
        c["n"] += 1
        if p["position_usd"] > 0:
            c["L"] += 1
        elif p["position_usd"] < 0:
            c["S"] += 1

    ranked = sorted(open_pos, key=lambda p: p.get("unrealized_pnl", 0))

    def pos_row(p):
        return (f"| {p['ticker']} | {p['cell']} | {p['signal']} | "
                f"{p['entry_price']:g} → {p.get('current_price', 0):g} | "
                f"${p.get('unrealized_pnl', 0):+,.0f} |")

    L = []
    L.append(f"# Mark-to-Market Report — {today}")
    L.append("")
    L.append(f"**Notional:** ${notional:,.0f}. **Days held:** {days} trading days. "
             f"**Governance:** {'PASS' if _governance_ok(state) else 'FAIL'}.")
    L.append("")
    L.append("<!-- AUTO-GENERATED skeleton. Fill the TODO narrative sections by hand. -->")
    L.append("")
    L.append("## Headline")
    L.append("")
    L.append(f"**Net P&L to-date: ${net:+,.0f} ({net_pct:+.2f}% of notional)** "
             f"after {days} trading days.")
    L.append("")
    L.append("<!-- TODO: one paragraph on what drove the move this run. -->")
    L.append("")
    L.append("| | Amount |")
    L.append("|---|---|")
    L.append(f"| Realized P&L ({n_closed_all} closed) | ${realized:+,.0f} |")
    L.append(f"| Unrealized P&L ({len(open_pos)} open) | ${unreal:+,.0f} |")
    L.append(f"| Accrued borrow | ${-accrued:+,.0f} |")
    L.append(f"| **Net** | **${net:+,.0f} ({net_pct:+.2f}%)** |")
    L.append("")

    L.append("## Trades closed this run")
    L.append("")
    if closed_this_run:
        L.append("| Ticker | Cell | Signal | Entry | Exit | P&L | Reason |")
        L.append("|--------|------|--------|-------|------|-----|--------|")
        for t in closed_this_run:
            L.append(f"| {t['ticker']} | {t['cell']} | {t['signal']} | "
                     f"{t['entry_price']:g} | {t['exit_price']:g} | "
                     f"${t['realized_pnl']:+,.0f} | {t['close_reason']} |")
    else:
        L.append("None — no stop-losses or expiries this run.")
    L.append("")

    L.append("## Open P&L by cell")
    L.append("")
    L.append("| Cell | Unreal P&L | n | L / S |")
    L.append("|------|-----------:|---|-------|")
    for cell in sorted(cells, key=lambda x: -cells[x]["pnl"]):
        c = cells[cell]
        L.append(f"| {cell} | ${c['pnl']:+,.0f} | {c['n']} | "
                 f"{c['L']}L / {c['S']}S |")
    L.append("")
    L.append(f"Exposure: {total_long/notional*100:.1f}% long / "
             f"{abs(total_short)/notional*100:.1f}% short "
             f"(net {net_exp/notional*100:+.1f}%). "
             f"Cash ${state.get('cash', 0):,.0f}.")
    L.append("")

    L.append("## Winners and losers")
    L.append("")
    L.append("**Top 5 by unrealized P&L:**")
    L.append("")
    L.append("| Ticker | Cell | Signal | Entry → Current | Unreal P&L |")
    L.append("|--------|------|--------|-----------------|-----------:|")
    for p in reversed(ranked[-5:]):
        L.append(pos_row(p))
    L.append("")
    L.append("**Bottom 5 by unrealized P&L:**")
    L.append("")
    L.append("| Ticker | Cell | Signal | Entry → Current | Unreal P&L |")
    L.append("|--------|------|--------|-----------------|-----------:|")
    for p in ranked[:5]:
        L.append(pos_row(p))
    L.append("")

    if stale_tickers:
        L.append("## Data-pipeline note")
        L.append("")
        L.append(f"{len(stale_tickers)} ticker(s) had no fresh price this run and kept "
                 f"last-known marks (MTM/borrow/expiry not advanced for them): "
                 f"{', '.join(sorted(stale_tickers))}.")
        L.append("")

    L.append("## Reading it honestly")
    L.append("")
    L.append("<!-- TODO: is this signal or noise? What's realized vs unrealized, "
             "concentration, carried risks. -->")
    L.append("")

    L.append("## Rebalance / expiry schedule")
    L.append("")
    L.append("| Cell | Next rebalance | Status |")
    L.append("|------|----------------|--------|")
    active = {cell_key(p["region"], p["horizon"]) for p in open_pos}
    for ck, cell in state.get("cells", {}).items():
        cl = cell_label(cell["region"], cell["horizon"])
        if cell.get("blocked"):
            status = "BLOCKED"
        elif ck in active:
            status = "open"
        elif cell.get("cell_stopped"):
            status = f"cell-stopped {cell['cell_stopped']}"
        else:
            status = "empty"
        L.append(f"| {cl} | {cell.get('next_rebalance', '?')} | {status} |")
    L.append("")

    out_path = TRADE_DIR / f"mtm_{today}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paper Trade Simulator for TimesFM Sector Router")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Initialize a new paper trade")
    p_init.add_argument("--notional", type=float, default=1_000_000,
                        help="Portfolio notional in USD (default: 1,000,000)")
    p_init.add_argument("--borrow-rate", type=float, default=DEFAULT_BORROW_RATE,
                        help=f"Annualized borrow rate (default: {DEFAULT_BORROW_RATE})")
    p_init.add_argument("--force", action="store_true",
                        help="Overwrite existing state")

    # run
    p_run = subparsers.add_parser("run", help="Daily run")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Use random prices/signals")
    p_run.add_argument("--report", action="store_true",
                       help="Also write mtm_{date}.md report skeleton")

    # status
    subparsers.add_parser("status", help="Show current portfolio")

    # history
    subparsers.add_parser("history", help="Show trade journal")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "history":
        cmd_history(args)


if __name__ == "__main__":
    main()
