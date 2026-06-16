"""
Retroactive performance tracking for expanded global ticker universe.
Pretends all 25 tickers were included in the March 16 plan.

- Downloads actual prices March 13 (context/entry ref) through March 23
- Runs TimesFM forecasts from March 13 context for the NEW tickers
  (original 15 forecasts already exist in forecast_mar14.json)
- Shows per-ticker and per-day performance table

Usage:
  cd C:/Users/ylchen/workspace/timesfm
  python global_sim/track_expanded.py
"""
from __future__ import annotations

import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import timesfm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONTEXT_END  = datetime.date(2026, 3, 13)   # forecast context date
ENTRY_DATE   = datetime.date(2026, 3, 16)   # plan entry day
TRACK_END    = datetime.date(2026, 3, 23)   # today
HORIZON      = 10
MAX_CONTEXT  = 512

FORECAST_FILE = ROOT / "global_sim" / "forecast_mar14.json"

# Business days in the tracking window
BDAYS = [
    datetime.date(2026, 3, 16),
    datetime.date(2026, 3, 17),
    datetime.date(2026, 3, 18),
    datetime.date(2026, 3, 19),
    datetime.date(2026, 3, 20),
    datetime.date(2026, 3, 23),
]

# ---------------------------------------------------------------------------
# Original 15 tickers (forecasts already in forecast_mar14.json)
ORIGINAL = ["2382.HK", "UNH", "CVX", "VWAGY", "0388.HK",   # selected
            "NVDA", "GS", "KO", "BA", "1177.HK",             # tracked but not selected
            "UL", "EADSF", "8035.T", "6902.T", "4661.T"]

# New tickers to add retroactively
NEW_CHINA  = ["0700.HK", "0941.HK", "0005.HK", "1299.HK", "2318.HK"]
NEW_EUROPE = ["ASML", "SAP", "NVO", "SHEL", "TTE"]
NEW_TICKERS = NEW_CHINA + NEW_EUROPE

ALL_TICKERS = ORIGINAL + NEW_TICKERS

HKD_TICKERS = {t for t in ALL_TICKERS if t.endswith(".HK")}
JPY_TICKERS  = {t for t in ALL_TICKERS if t.endswith(".T")}

CHECKPOINT_MAP = {
    # USA
    "NVDA": ROOT / "checkpoints/run_glia_4b/best",
    "UNH":  ROOT / "checkpoints/best",
    "GS":   ROOT / "checkpoints/run_glia_4b/best",
    "CVX":  ROOT / "checkpoints/run_glia_3a/best",
    "KO":   ROOT / "checkpoints/run_glia_4b/best",
    "BA":   ROOT / "checkpoints/best",
    # China — Glia_China
    **{t: ROOT / "finetune_china/checkpoints/run_glia_2a/best"
       for t in ["0388.HK","2382.HK","1177.HK","0700.HK","0941.HK","0005.HK","1299.HK","2318.HK"]},
    # Europe — Run1_EU
    **{t: ROOT / "finetune_europe/checkpoints/run1/best"
       for t in ["UL","EADSF","VWAGY","ASML","SAP","NVO","SHEL","TTE"]},
    # Japan — pretrained
    **{t: "google/timesfm-2.5-200m-pytorch" for t in ["8035.T","6902.T","4661.T"]},
}

TICKER_NAMES = {
    "NVDA":"NVIDIA", "UNH":"UnitedHealth", "GS":"Goldman Sachs",
    "CVX":"Chevron", "KO":"Coca-Cola", "BA":"Boeing",
    "0388.HK":"HKEX", "2382.HK":"Sunny Optical", "1177.HK":"Sino Biopharma",
    "0700.HK":"Tencent", "0941.HK":"China Mobile", "0005.HK":"HSBC",
    "1299.HK":"AIA Group", "2318.HK":"Ping An",
    "UL":"Unilever", "EADSF":"Airbus", "VWAGY":"Volkswagen",
    "ASML":"ASML", "SAP":"SAP SE", "NVO":"Novo Nordisk",
    "SHEL":"Shell", "TTE":"TotalEnergies",
    "8035.T":"Tokyo Electron", "6902.T":"Denso", "4661.T":"Oriental Land",
}

def region(t):
    if t in ("NVDA","UNH","GS","CVX","KO","BA"): return "USA"
    if t.endswith(".HK"): return "China"
    if t.endswith(".T"):  return "Japan"
    return "Europe"

# ---------------------------------------------------------------------------
# Load existing forecasts for original tickers
# ---------------------------------------------------------------------------
with open(FORECAST_FILE) as f:
    fdata = json.load(f)

orig_forecasts   = fdata["forecasts"]    # ticker -> list[10 floats]
orig_last_prices = fdata["last_prices"]  # ticker -> Mar13 close

# ---------------------------------------------------------------------------
# Download actual prices (Mar 13 context through Mar 23)
# ---------------------------------------------------------------------------
print("Downloading actual prices...")
dl_start = (CONTEXT_END - datetime.timedelta(days=365 * 3)).isoformat()
dl_end   = (TRACK_END   + datetime.timedelta(days=3)).isoformat()

raw = yf.download(
    ALL_TICKERS,
    start=dl_start,
    end=dl_end,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

def get_price_series(ticker):
    """Return dict {date: price} for the full downloaded range."""
    cl = raw["Close"][ticker] if len(ALL_TICKERS) > 1 else raw["Close"]
    out = {}
    for ts, p in cl.items():
        if not (np.isnan(p) or p <= 0):
            d = ts.date() if hasattr(ts, "date") else datetime.date.fromisoformat(str(ts)[:10])
            out[d] = float(p)
    return out

price_series = {}   # ticker -> {date: price}
for t in ALL_TICKERS:
    try:
        price_series[t] = get_price_series(t)
    except Exception as e:
        print(f"  WARNING: {t}: {e}")
        price_series[t] = {}

# Entry price = March 16 open if available, else March 13 close
entry_prices = {}
context_prices = {}  # March 13 close
for t in ALL_TICKERS:
    ps = price_series[t]
    # March 13 close
    ctx = ps.get(CONTEXT_END)
    if ctx is None:
        ctx = orig_last_prices.get(t)
    context_prices[t] = ctx

    # March 16 open — use close as proxy (yfinance daily gives Open column)
    # Try to get March 16 open from raw["Open"]
    try:
        op_series = raw["Open"][t] if len(ALL_TICKERS) > 1 else raw["Open"]
        op16 = None
        for ts, p in op_series.items():
            d = ts.date() if hasattr(ts, "date") else datetime.date.fromisoformat(str(ts)[:10])
            if d == ENTRY_DATE and not np.isnan(p) and p > 0:
                op16 = float(p)
                break
        entry_prices[t] = op16 if op16 else ctx
    except Exception:
        entry_prices[t] = ctx

# ---------------------------------------------------------------------------
# Generate retroactive forecasts for NEW tickers (March 13 context)
# ---------------------------------------------------------------------------
print("\nRunning retroactive forecasts for new tickers from March 13 context...")

new_forecasts = {}

ckpt_groups: dict[str, list[str]] = defaultdict(list)
for t in NEW_TICKERS:
    if context_prices.get(t):
        ckpt_groups[str(CHECKPOINT_MAP[t])].append(t)

def ckpt_sort(k):
    return (1 if "google/" in k else 0, k)

for ckpt_path in sorted(ckpt_groups, key=ckpt_sort):
    tickers = ckpt_groups[ckpt_path]
    is_hf = "google/" in ckpt_path
    label = "pretrained" if is_hf else ckpt_path.replace(str(ROOT)+"\\","")
    print(f"  Loading {label}  →  {tickers}")

    try:
        if is_hf:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(ckpt_path)
        else:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                ckpt_path, local_files_only=True)
        model.compile(timesfm.ForecastConfig(max_context=MAX_CONTEXT, max_horizon=HORIZON))

        # Context = prices up to and including March 13
        inputs = []
        for t in tickers:
            ps = price_series[t]
            ctx_arr = np.array(
                [ps[d] for d in sorted(ps) if d <= CONTEXT_END],
                dtype=np.float32
            )[-MAX_CONTEXT:]
            inputs.append(ctx_arr.tolist())

        point_preds, _ = model.forecast(horizon=HORIZON, inputs=inputs)
        for t, preds in zip(tickers, point_preds):
            new_forecasts[t] = [float(p) for p in preds[:HORIZON]]
        del model

    except Exception as e:
        print(f"    ERROR: {e}")
        for t in tickers:
            new_forecasts[t] = [context_prices[t]] * HORIZON

# Merge forecasts
all_forecasts = {**orig_forecasts, **new_forecasts}

# ---------------------------------------------------------------------------
# Build per-ticker, per-day return table
# ---------------------------------------------------------------------------
# 10d forecast target (index 9 = day 10)
pred_return = {}
pred_target = {}
for t in ALL_TICKERS:
    base = context_prices.get(t)
    fc   = all_forecasts.get(t)
    if base and fc:
        pred_target[t] = fc[-1]
        pred_return[t] = (fc[-1] - base) / base * 100
    else:
        pred_target[t] = float("nan")
        pred_return[t] = float("nan")

# Actual returns per day from entry
daily_returns = {}   # ticker -> {date: return_from_entry %}
for t in ALL_TICKERS:
    ep = entry_prices.get(t)
    if not ep:
        continue
    daily_returns[t] = {}
    for d in BDAYS:
        ap = price_series[t].get(d)
        if ap:
            daily_returns[t][d] = (ap - ep) / ep * 100

# Rank all 25 tickers by predicted 10d return
ranked = sorted(
    [t for t in ALL_TICKERS if not np.isnan(pred_return.get(t, float("nan")))],
    key=lambda t: pred_return[t],
    reverse=True,
)

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
ccy = lambda t: "HKD" if t in HKD_TICKERS else ("JPY" if t in JPY_TICKERS else "USD")
flag = lambda t: " *" if t in NEW_TICKERS else "  "

print(f"\n{'='*110}")
print(f"Expanded Global Plan — Retroactive Performance (March 16–23, Day 1–6)")
print(f"* = new ticker (not in original plan)  |  Entry = March 16 open")
print(f"{'='*110}")
hdr = f"{'':2}{'Rank':<5}{'Ticker':<11}{'Rgn':<7}{'Name':<22}{'Entry':>10}  {'Pred Ret':>9}  "
hdr += "  ".join([d.strftime("%b%d") for d in BDAYS])
print(hdr)
print("-" * 110)

for i, t in enumerate(ranked, 1):
    ep = entry_prices.get(t, float("nan"))
    pr = pred_return.get(t, float("nan"))
    c  = ccy(t)
    row = f"{flag(t)}{i:<5}{t:<11}{region(t):<7}{TICKER_NAMES.get(t,t):<22}{c}{ep:>8.2f}  {pr:>+8.2f}%  "
    day_cols = []
    for d in BDAYS:
        ret = daily_returns.get(t, {}).get(d)
        day_cols.append(f"{ret:>+6.2f}%" if ret is not None else "   N/A ")
    row += "  ".join(day_cols)
    print(row)

print(f"\n* New tickers — retroactive forecast + actual prices pulled from yfinance")

# ---------------------------------------------------------------------------
# Summary: how would a Top-5 portfolio from the expanded universe have done?
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print("What if we had picked Top-5 from the expanded universe on March 13?")
print(f"{'='*80}")
top5_expanded = ranked[:5]
print(f"Top 5 (by predicted return): {top5_expanded}")
print()
print(f"{'Ticker':<12}{'Pred Ret':>10}  {'Day1':>7}  {'Day2':>7}  {'Day3':>7}  {'Day4':>7}  {'Day5':>7}  {'Day6':>7}")
print("-" * 70)
portf_by_day = {d: [] for d in BDAYS}
for t in top5_expanded:
    ep = entry_prices.get(t, float("nan"))
    pr = pred_return.get(t, float("nan"))
    row = f"{t:<12}{pr:>+9.2f}%"
    for d in BDAYS:
        ret = daily_returns.get(t, {}).get(d)
        row += f"  {ret:>+6.2f}%" if ret is not None else "     N/A"
        if ret is not None:
            portf_by_day[d].append(ret)
    print(row)

print("-" * 70)
row = f"{'Portfolio avg':<12}{'':>10}"
for d in BDAYS:
    rets = portf_by_day[d]
    avg = np.mean(rets) if rets else float("nan")
    row += f"  {avg:>+6.2f}%" if not np.isnan(avg) else "     N/A"
print(row)

# Compare to original 5-ticker plan
orig5 = ["2382.HK", "UNH", "CVX", "VWAGY", "0388.HK"]
print(f"\nOriginal 5-ticker plan: {orig5}")
portf_orig = {d: [] for d in BDAYS}
for t in orig5:
    for d in BDAYS:
        ret = daily_returns.get(t, {}).get(d)
        if ret is not None:
            portf_orig[d].append(ret)

row = f"{'Original avg':<12}{'':>10}"
for d in BDAYS:
    rets = portf_orig[d]
    avg = np.mean(rets) if rets else float("nan")
    row += f"  {avg:>+6.2f}%" if not np.isnan(avg) else "     N/A"
print(row)

# ---------------------------------------------------------------------------
# Save expanded results JSON
# ---------------------------------------------------------------------------
out = {
    "generated": TRACK_END.isoformat(),
    "context_date": CONTEXT_END.isoformat(),
    "entry_date": ENTRY_DATE.isoformat(),
    "biz_days": [d.isoformat() for d in BDAYS],
    "entry_prices": {t: entry_prices[t] for t in ALL_TICKERS if t in entry_prices},
    "context_prices": {t: context_prices[t] for t in ALL_TICKERS if context_prices.get(t)},
    "pred_returns": {t: pred_return[t] for t in ALL_TICKERS if not np.isnan(pred_return.get(t, float("nan")))},
    "pred_targets": {t: pred_target[t] for t in ALL_TICKERS if not np.isnan(pred_target.get(t, float("nan")))},
    "forecasts": all_forecasts,
    "daily_returns": {t: {d.isoformat(): r for d, r in daily_returns.get(t, {}).items()} for t in ALL_TICKERS},
    "ranking": ranked,
    "new_tickers": NEW_TICKERS,
    "original_tickers": ORIGINAL,
}
out_path = ROOT / "global_sim" / "track_expanded.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
