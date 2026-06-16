"""
Prepare EXPANDED training data for international market finetuning (v2).

15-20 training tickers per region (up from 10). Eval tickers unchanged.
Saves to finetune_{region}/data/train_v2/ to avoid overwriting originals.

Usage:
  cd timesfm
  python scripts/prepare_intl_data_v2.py                     # all regions
  python scripts/prepare_intl_data_v2.py --region japan
  python scripts/prepare_intl_data_v2.py --region china
  python scripts/prepare_intl_data_v2.py --region europe
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
  import yfinance as yf
except ImportError:
  sys.exit("yfinance required: pip install yfinance")

# ---------------------------------------------------------------------------
# Expanded ticker definitions: 15-20 per region
# ---------------------------------------------------------------------------
# Selection criteria:
#   - Long history (>10 years preferred, >5 years minimum)
#   - High liquidity
#   - Sector diversity within region
#   - No overlap with eval tickers

REGIONS = {
  "japan": {
    "train": [
      # Original 10
      "^N225",    # Nikkei 225 index
      "7203.T",   # Toyota Motor (auto)
      "6758.T",   # Sony Group (tech/entertainment)
      "9984.T",   # SoftBank Group (telecom/investment)
      "8306.T",   # Mitsubishi UFJ Financial (MUFG, banking)
      "6501.T",   # Hitachi (conglomerate)
      "7974.T",   # Nintendo (gaming)
      "9432.T",   # NTT (telecom)
      "4519.T",   # Chugai Pharmaceutical (healthcare)
      "3382.T",   # Seven & i Holdings (retail)
      # New 8 — sector gap fills + long-history large caps
      "4502.T",   # Takeda Pharmaceutical (healthcare)
      "6367.T",   # Daikin Industries (HVAC/industrials)
      "8316.T",   # Sumitomo Mitsui Financial (banking)
      "6723.T",   # Renesas Electronics (semiconductors — analogue to eval 8035.T)
      "7267.T",   # Honda Motor (auto — analogue to eval 6902.T Denso)
      "9433.T",   # KDDI (telecom)
      "4063.T",   # Shin-Etsu Chemical (materials/chemicals)
      "8801.T",   # Mitsui Fudosan (real estate)
    ],
    "eval": [
      "8035.T",   # Tokyo Electron (semiconductor)
      "6902.T",   # Denso (auto parts)
      "4661.T",   # Oriental Land (Disney Japan)
    ],
    "eval_training": [
      # Subset of training tickers for held-out window evaluation
      # Tests whether finetuning learned useful patterns at all
      "7203.T",   # Toyota Motor (auto)
      "6758.T",   # Sony Group (tech/entertainment)
      "9984.T",   # SoftBank Group (telecom/investment)
      "8306.T",   # Mitsubishi UFJ Financial (banking)
      "4502.T",   # Takeda Pharmaceutical (healthcare)
      "7974.T",   # Nintendo (gaming)
    ],
    "train_dir": "finetune_japan/data/train_v2",
    "eval_dir": "finetune_japan/data/eval",
  },
  "china": {
    "train": [
      # Original 10 (keep ALL including short-history — Glia proved they're essential)
      "^HSI",     # Hang Seng Index
      "0700.HK",  # Tencent Holdings (tech)
      "9988.HK",  # Alibaba Group (tech, post-2019)
      "1299.HK",  # AIA Group (insurance)
      "0941.HK",  # China Mobile (telecom)
      "3690.HK",  # Meituan (tech/food delivery, post-2018)
      "9999.HK",  # NetEase (gaming, post-2020)
      "1810.HK",  # Xiaomi (tech, post-2018)
      "2318.HK",  # Ping An Insurance
      "0005.HK",  # HSBC Holdings (HK)
      # New 8 — fill sector gaps, prioritize long history
      "0001.HK",  # CK Hutchison (conglomerate, 20+ years)
      "0016.HK",  # Sun Hung Kai Properties (real estate, 20+ years)
      "0002.HK",  # CLP Holdings (utilities/power, 20+ years)
      "0003.HK",  # HK & China Gas (utilities, 20+ years)
      "0011.HK",  # Hang Seng Bank (banking, 20+ years)
      "1398.HK",  # ICBC (banking, 15+ years)
      "0883.HK",  # CNOOC (energy/oil, 15+ years)
      "0027.HK",  # Galaxy Entertainment (casino/leisure, 15+ years)
    ],
    "eval": [
      "0388.HK",  # Hong Kong Exchanges (HKEX)
      "2382.HK",  # Sunny Optical
      "1177.HK",  # Sino Biopharmaceutical
    ],
    "eval_training": [
      # Subset of training tickers for held-out window evaluation
      "0700.HK",  # Tencent Holdings (tech)
      "0941.HK",  # China Mobile (telecom)
      "2318.HK",  # Ping An Insurance
      "1299.HK",  # AIA Group (insurance)
      "0005.HK",  # HSBC Holdings (HK)
      "0001.HK",  # CK Hutchison (conglomerate)
    ],
    "train_dir": "finetune_china/data/train_v2",
    "eval_dir": "finetune_china/data/eval",
  },
  "europe": {
    "train": [
      # Original 10
      "^FTSE",    # FTSE 100 (London, GBP)
      "^GDAXI",   # DAX (Frankfurt, EUR)
      "^FCHI",    # CAC 40 (Paris, EUR)
      "ASML",     # ASML Holding (NASDAQ-listed, USD)
      "SAP",      # SAP SE (NYSE-listed, USD)
      "NVO",      # Novo Nordisk (NYSE-listed, USD)
      "AZN",      # AstraZeneca (NASDAQ-listed, USD)
      "SHEL",     # Shell plc (NYSE-listed, USD)
      "TTE",      # TotalEnergies (NYSE-listed, USD)
      "RIO",      # Rio Tinto (NYSE-listed, USD)
      # New 8 — fill auto/aero gaps + more EU large caps with US listing
      "DEO",      # Diageo (NYSE, consumer staples — close to eval UL)
      "GSK",      # GSK plc (NYSE, healthcare)
      "LIN",      # Linde plc (NYSE, industrial gases — German HQ)
      "SAN",      # Banco Santander (NYSE ADR, banking)
      "ING",      # ING Group (NYSE ADR, banking)
      "STLA",     # Stellantis (NYSE, auto — analogue to eval VWAGY)
      "PHG",      # Philips (NYSE ADR, healthcare tech)
      "TEF",      # Telefonica (NYSE ADR, telecom)
    ],
    "eval": [
      "UL",       # Unilever (NYSE ADR)
      "EADSF",    # Airbus (OTC)
      "VWAGY",    # Volkswagen (OTC ADR)
    ],
    "eval_training": [
      # Subset of training tickers for held-out window evaluation
      "ASML",     # ASML Holding (semiconductor)
      "SAP",      # SAP SE (enterprise software)
      "NVO",      # Novo Nordisk (pharma)
      "AZN",      # AstraZeneca (pharma)
      "SHEL",     # Shell plc (energy)
      "DEO",      # Diageo (consumer staples)
    ],
    "train_dir": "finetune_europe/data/train_v2",
    "eval_dir": "finetune_europe/data/eval",
  },
}


def download_ticker(ticker: str, start: str, end: str) -> np.ndarray | None:
  try:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
      return None
    if isinstance(df.columns, pd.MultiIndex):
      df = df.droplevel("Ticker", axis=1)
    close = df["Close"].values.astype(np.float32).flatten()
    close = close[~np.isnan(close) & (close > 0)]
    return close if len(close) >= 252 else None
  except Exception as e:
    print(f"    ERROR: {e}")
    return None


def safe_filename(ticker: str) -> str:
  return ticker.replace("^", "IDX_").replace(".", "_") + ".npy"


def prepare_region(name: str, cfg: dict, years: int, root: Path):
  print(f"\n{'='*60}")
  print(f"Region: {name.upper()}")
  print(f"{'='*60}")

  end = datetime.date.today().isoformat()
  start = (datetime.date.today() - datetime.timedelta(days=365 * years)).isoformat()

  for split in ["train", "eval"]:
    tickers = cfg[split]
    out_dir = root / cfg[f"{split}_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{split.upper()}] → {out_dir}  ({len(tickers)} tickers)")

    success, skipped = [], []
    for ticker in tickers:
      fname = safe_filename(ticker)
      print(f"    {ticker:<15}", end=" ", flush=True)
      data = download_ticker(ticker, start, end)
      if data is None:
        print(f"SKIPPED (no data or < 252 pts)")
        skipped.append(ticker)
        continue
      np.save(out_dir / fname, data)
      print(f"{len(data):>5} pts  ({len(data)/252:.1f} yrs)  →  {fname}")
      success.append(ticker)

    print(f"\n  Saved {len(success)}/{len(tickers)} tickers to {out_dir}")
    if skipped:
      print(f"  Skipped: {skipped}")

    # Summary stats
    print(f"\n  {'File':<25} {'Points':>7}  {'Min':>9}  {'Max':>9}  {'Yrs':>5}")
    for f in sorted(out_dir.glob("*.npy")):
      arr = np.load(f)
      yrs = len(arr) / 252
      print(f"  {f.name:<25} {len(arr):>7}  {arr.min():>9.2f}  {arr.max():>9.2f}  {yrs:>5.1f}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--region", choices=["japan", "china", "europe", "all"],
                      default="all")
  parser.add_argument("--years", type=int, default=20,
                      help="Years of history to download (default: 20)")
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  regions = list(REGIONS.keys()) if args.region == "all" else [args.region]

  print(f"Downloading {args.years} years of data for: {', '.join(regions)}")
  print(f"Root: {root}")

  for name in regions:
    prepare_region(name, REGIONS[name], args.years, root)

  print(f"\n{'='*60}")
  print("Done! New training data in finetune_*/data/train_v2/")
  for name in regions:
    cfg = REGIONS[name]
    n = len(cfg["train"])
    print(f"  {name}: {n} tickers → {cfg['train_dir']}")


if __name__ == "__main__":
  main()
