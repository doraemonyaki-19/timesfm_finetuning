"""
Expanded global forecast — context through March 23, 2026.

Universe: 25 tickers across 4 regions.
  USA    (6): NVDA, UNH, GS, CVX, KO, BA
  China  (8): 0388.HK, 2382.HK, 1177.HK  +  0700.HK, 0941.HK, 0005.HK, 1299.HK, 2318.HK
  Europe (8): UL, EADSF, VWAGY            +  ASML, SAP, NVO, SHEL, TTE
  Japan  (3): 8035.T, 6902.T, 4661.T

Checkpoint routing:
  USA    — selective (Run4 default, Glia3a for CVX, Glia4b for NVDA/GS/KO)
  China  — Glia_China (finetune_china/checkpoints/run_glia_2a/best) for all HK tickers
  Europe — Run1_EU (finetune_europe/checkpoints/run1/best) for all Europe tickers
  Japan  — pretrained (google/timesfm-2.5-200m-pytorch)

Output: global_sim/forecast_mar23.json

Usage:
  cd C:/Users/ylchen/workspace/timesfm
  python global_sim/forecast_mar23.py
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
CONTEXT_END  = datetime.date(2026, 3, 23)   # today's close
HORIZON      = 10                            # 10 trading-day steps
MAX_CONTEXT  = 512
OUT_FILE     = ROOT / "global_sim" / "forecast_mar23.json"

# Next 10 US trading days (skipping Good Friday March 28, Easter Monday March 31)
BIZ_DAYS = [
    datetime.date(2026, 3, 24),
    datetime.date(2026, 3, 25),
    datetime.date(2026, 3, 26),
    datetime.date(2026, 3, 27),
    datetime.date(2026, 3, 31),  # Good Friday skipped; Easter Monday open in US
    datetime.date(2026, 4,  1),
    datetime.date(2026, 4,  2),
    datetime.date(2026, 4,  3),
    datetime.date(2026, 4,  4),
    datetime.date(2026, 4,  7),
]
# Note: HK closes Good Friday (Mar 28) AND Easter Monday (Mar 31).
# Japan closes for Showa Day around April 29. No impact on this window.

# ---------------------------------------------------------------------------
# Tickers grouped by region and checkpoint
# ---------------------------------------------------------------------------
TICKERS_BY_REGION = {
    "USA":    ["NVDA", "UNH", "GS", "CVX", "KO", "BA"],
    "China":  ["0388.HK", "2382.HK", "1177.HK",
               "0700.HK", "0941.HK", "0005.HK", "1299.HK", "2318.HK"],
    "Europe": ["UL", "EADSF", "VWAGY",
               "ASML", "SAP", "NVO", "SHEL", "TTE"],
    "Japan":  ["8035.T", "6902.T", "4661.T"],
}
ALL_TICKERS = [t for tl in TICKERS_BY_REGION.values() for t in tl]

HKD_TICKERS = {t for t in ALL_TICKERS if t.endswith(".HK")}
JPY_TICKERS  = {t for t in ALL_TICKERS if t.endswith(".T")}

CHECKPOINT_MAP = {
    # USA — selective routing
    "NVDA": ROOT / "checkpoints/run_glia_4b/best",
    "UNH":  ROOT / "checkpoints/best",
    "GS":   ROOT / "checkpoints/run_glia_4b/best",
    "CVX":  ROOT / "checkpoints/run_glia_3a/best",
    "KO":   ROOT / "checkpoints/run_glia_4b/best",
    "BA":   ROOT / "checkpoints/best",
    # China — Glia_China for all HK tickers
    "0388.HK": ROOT / "finetune_china/checkpoints/run_glia_2a/best",
    "2382.HK": ROOT / "finetune_china/checkpoints/run_glia_2a/best",
    "1177.HK": ROOT / "finetune_china/checkpoints/run_glia_2a/best",
    "0700.HK": ROOT / "finetune_china/checkpoints/run_glia_2a/best",
    "0941.HK": ROOT / "finetune_china/checkpoints/run_glia_2a/best",
    "0005.HK": ROOT / "finetune_china/checkpoints/run_glia_2a/best",
    "1299.HK": ROOT / "finetune_china/checkpoints/run_glia_2a/best",
    "2318.HK": ROOT / "finetune_china/checkpoints/run_glia_2a/best",
    # Europe — Run1_EU for all Europe tickers (trained on ASML/SAP/NVO/SHEL/TTE)
    "UL":    ROOT / "finetune_europe/checkpoints/run1/best",
    "EADSF": ROOT / "finetune_europe/checkpoints/run1/best",
    "VWAGY": ROOT / "finetune_europe/checkpoints/run1/best",
    "ASML":  ROOT / "finetune_europe/checkpoints/run1/best",
    "SAP":   ROOT / "finetune_europe/checkpoints/run1/best",
    "NVO":   ROOT / "finetune_europe/checkpoints/run1/best",
    "SHEL":  ROOT / "finetune_europe/checkpoints/run1/best",
    "TTE":   ROOT / "finetune_europe/checkpoints/run1/best",
    # Japan — pretrained (finetuning hurt Japan uniformly)
    "8035.T": "google/timesfm-2.5-200m-pytorch",
    "6902.T": "google/timesfm-2.5-200m-pytorch",
    "4661.T": "google/timesfm-2.5-200m-pytorch",
}

TICKER_NAMES = {
    "NVDA": "NVIDIA", "UNH": "UnitedHealth", "GS": "Goldman Sachs",
    "CVX": "Chevron", "KO": "Coca-Cola", "BA": "Boeing",
    "0388.HK": "HKEX", "2382.HK": "Sunny Optical", "1177.HK": "Sino Biopharma",
    "0700.HK": "Tencent", "0941.HK": "China Mobile", "0005.HK": "HSBC",
    "1299.HK": "AIA Group", "2318.HK": "Ping An",
    "UL": "Unilever", "EADSF": "Airbus", "VWAGY": "Volkswagen ADR",
    "ASML": "ASML", "SAP": "SAP SE", "NVO": "Novo Nordisk",
    "SHEL": "Shell", "TTE": "TotalEnergies",
    "8035.T": "Tokyo Electron", "6902.T": "Denso", "4661.T": "Oriental Land",
}

# ---------------------------------------------------------------------------
# Download prices
# ---------------------------------------------------------------------------
print("Downloading price data...")
start = (CONTEXT_END - datetime.timedelta(days=365 * 3)).isoformat()
end   = (CONTEXT_END + datetime.timedelta(days=3)).isoformat()

raw = yf.download(
    ALL_TICKERS,
    start=start,
    end=end,
    interval="1d",
    auto_adjust=True,
    progress=False,
)

def get_context(ticker: str) -> tuple[np.ndarray, float]:
    cl = raw["Close"][ticker] if len(ALL_TICKERS) > 1 else raw["Close"]
    prices, dates = [], []
    for ts, p in cl.items():
        if not (np.isnan(p) or p <= 0):
            d = ts.date() if hasattr(ts, "date") else datetime.date.fromisoformat(str(ts)[:10])
            dates.append(d)
            prices.append(float(p))
    # Context: all prices up to and including CONTEXT_END
    ctx_prices = [p for p, d in zip(prices, dates) if d <= CONTEXT_END]
    if not ctx_prices:
        raise ValueError(f"No price data for {ticker} up to {CONTEXT_END}")
    ctx = np.array(ctx_prices, dtype=np.float32)[-MAX_CONTEXT:]
    last_price = float(ctx[-1])
    return ctx, last_price

# Gather contexts and last prices
contexts   = {}
last_prices = {}
print(f"\n{'Ticker':<12} {'Points':>7}  {'Mar23 Close':>12}")
print("-" * 35)
for t in ALL_TICKERS:
    try:
        ctx, lp = get_context(t)
        contexts[t]    = ctx
        last_prices[t] = lp
        ccy = "HKD" if t in HKD_TICKERS else ("JPY" if t in JPY_TICKERS else "USD")
        print(f"  {t:<10} {len(ctx):>7}  {ccy} {lp:>10.2f}")
    except Exception as e:
        print(f"  {t:<10}  ERROR: {e}")

# ---------------------------------------------------------------------------
# Run forecasts — group by checkpoint to load each model once
# ---------------------------------------------------------------------------
forecasts = {}   # ticker -> list of 10 predicted prices

ckpt_to_tickers: dict[str, list[str]] = defaultdict(list)
for t in ALL_TICKERS:
    if t in contexts:
        ckpt_to_tickers[str(CHECKPOINT_MAP[t])].append(t)

# Sort: local checkpoints first, pretrained last (needs network)
def ckpt_sort_key(k):
    return (0 if "google/" not in k else 1, k)

for ckpt_path in sorted(ckpt_to_tickers, key=ckpt_sort_key):
    tickers = ckpt_to_tickers[ckpt_path]
    is_pretrained = "google/" in ckpt_path
    label = "pretrained (HuggingFace)" if is_pretrained else ckpt_path.replace(str(ROOT) + "\\", "")
    print(f"\nLoading: {label}")
    print(f"  Tickers: {tickers}")

    try:
        if is_pretrained:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(ckpt_path)
        else:
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                ckpt_path, local_files_only=True
            )
        model.compile(timesfm.ForecastConfig(max_context=MAX_CONTEXT, max_horizon=HORIZON))

        inputs = [contexts[t].tolist() for t in tickers]
        point_preds, _ = model.forecast(horizon=HORIZON, inputs=inputs)

        for t, preds in zip(tickers, point_preds):
            base = last_prices[t]
            # preds are absolute price forecasts for steps 1..HORIZON
            forecasts[t] = [float(p) for p in preds[:HORIZON]]
            ret = (forecasts[t][-1] - base) / base * 100
            ccy = "HKD" if t in HKD_TICKERS else ("JPY" if t in JPY_TICKERS else "USD")
            print(f"  {t:<10}  10d target: {ccy} {forecasts[t][-1]:.2f}  ({ret:+.2f}%)")

        del model

    except Exception as e:
        print(f"  ERROR loading {label}: {e}")
        for t in tickers:
            forecasts[t] = [last_prices[t]] * HORIZON   # flat fallback

# ---------------------------------------------------------------------------
# Compute predicted returns and rank
# ---------------------------------------------------------------------------
pred_returns = {}
for t in ALL_TICKERS:
    if t in forecasts and t in last_prices:
        pred_returns[t] = (forecasts[t][-1] - last_prices[t]) / last_prices[t] * 100

ranked = sorted(pred_returns, key=lambda t: pred_returns[t], reverse=True)

# Determine region for each ticker
def get_region(t):
    for r, tl in TICKERS_BY_REGION.items():
        if t in tl:
            return r
    return "Unknown"

print(f"\n{'='*72}")
print(f"Global Ranking — 25 Tickers — Context: March 23, 2026")
print(f"{'='*72}")
print(f"{'Rank':<5} {'Ticker':<12} {'Region':<8} {'Name':<22} {'Mar23 Close':>12} {'10d Target':>12} {'Pred Ret':>9}")
print("-" * 85)
for i, t in enumerate(ranked, 1):
    ccy = "HKD" if t in HKD_TICKERS else ("JPY" if t in JPY_TICKERS else "USD")
    region = get_region(t)
    name = TICKER_NAMES.get(t, t)
    lp = last_prices.get(t, float("nan"))
    tgt = forecasts[t][-1] if t in forecasts else float("nan")
    ret = pred_returns.get(t, float("nan"))
    print(f"{i:<5} {t:<12} {region:<8} {name:<22} {ccy} {lp:>9.2f}  {ccy} {tgt:>9.2f}  {ret:>+8.2f}%")

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
out = {
    "generated":    CONTEXT_END.isoformat(),
    "context_date": CONTEXT_END.isoformat(),
    "biz_days":     [d.isoformat() for d in BIZ_DAYS],
    "tickers_by_region": TICKERS_BY_REGION,
    "last_prices":  {t: last_prices[t] for t in ALL_TICKERS if t in last_prices},
    "forecasts":    {t: forecasts[t]   for t in ALL_TICKERS if t in forecasts},
    "pred_returns": {t: pred_returns[t] for t in ALL_TICKERS if t in pred_returns},
    "ranking":      ranked,
    "checkpoint_map": {t: str(CHECKPOINT_MAP[t]) for t in ALL_TICKERS},
}

with open(OUT_FILE, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {OUT_FILE}")
