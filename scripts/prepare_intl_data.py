"""
Prepare training and evaluation data for international market finetuning.

Downloads daily close prices for three regions:
  - Japan   (Tokyo Stock Exchange, prices in JPY)
  - China   (Hong Kong Stock Exchange, prices in HKD)
  - Europe  (mix of local exchanges, prices in local currency)

All series are saved as .npy float32 arrays. Log-transform in the training
loop makes the pipeline currency-agnostic (only log-returns matter).

Usage:
  cd timesfm
  python scripts/prepare_intl_data.py                     # all regions
  python scripts/prepare_intl_data.py --region japan
  python scripts/prepare_intl_data.py --region china
  python scripts/prepare_intl_data.py --region europe
  python scripts/prepare_intl_data.py --years 15          # shorter history
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
# Ticker definitions
# ---------------------------------------------------------------------------
# Training tickers: 10 per region (mirrors US 10-ticker regime that worked)
# Evaluation tickers: 3-4 held-out per region (never seen during training)
# Chosen for liquidity, long history, and sector diversity.

REGIONS = {
    "japan": {
        "train": [
            "^N225",    # Nikkei 225 index
            "7203.T",   # Toyota Motor
            "6758.T",   # Sony Group
            "9984.T",   # SoftBank Group
            "8306.T",   # Mitsubishi UFJ Financial (MUFG)
            "6501.T",   # Hitachi
            "7974.T",   # Nintendo
            "9432.T",   # NTT (Nippon Telegraph)
            "4519.T",   # Chugai Pharmaceutical
            "3382.T",   # Seven & i Holdings
        ],
        "eval": [
            "8035.T",   # Tokyo Electron (semiconductor)
            "6902.T",   # Denso (auto parts)
            "4661.T",   # Oriental Land (Disney Japan)
        ],
        "currency": "JPY",
        "data_dir": "data/npy_japan",
        "eval_dir": "data/npy_japan_eval",
    },
    "china": {
        # Using Hong Kong-listed stocks (HKD) — more liquid, longer yfinance history
        # than China A-shares (which have strict data access limitations)
        "train": [
            "^HSI",     # Hang Seng Index
            "0700.HK",  # Tencent Holdings
            "9988.HK",  # Alibaba Group (HK listing)
            "1299.HK",  # AIA Group (insurance)
            "0941.HK",  # China Mobile
            "3690.HK",  # Meituan (food delivery)
            "9999.HK",  # NetEase
            "1810.HK",  # Xiaomi
            "2318.HK",  # Ping An Insurance
            "0005.HK",  # HSBC Holdings (HK)
        ],
        "eval": [
            "0388.HK",  # Hong Kong Exchanges (HKEX)
            "2382.HK",  # Sunny Optical
            "1177.HK",  # Sino Biopharmaceutical
        ],
        "currency": "HKD",
        "data_dir": "data/npy_china",
        "eval_dir": "data/npy_china_eval",
    },
    "europe": {
        # Mix: major indices + large-caps listed on home exchanges or US-ADR
        # Using home-exchange tickers where yfinance has good coverage
        "train": [
            "^FTSE",    # FTSE 100 (London, GBP)
            "^GDAXI",   # DAX (Frankfurt, EUR)
            "^FCHI",    # CAC 40 (Paris, EUR)
            "ASML",     # ASML Holding (NASDAQ-listed, USD — best yfinance data)
            "SAP",      # SAP SE (NYSE-listed, USD)
            "NVO",      # Novo Nordisk (NYSE-listed, USD)
            "AZN",      # AstraZeneca (NASDAQ-listed, USD)
            "SHEL",     # Shell plc (NYSE-listed, USD)
            "TTE",      # TotalEnergies (NYSE-listed, USD)
            "RIO",      # Rio Tinto (NYSE-listed, USD)
        ],
        "eval": [
            "UL",       # Unilever (NYSE ADR)
            "EADSF",    # Airbus (OTC)
            "VWAGY",    # Volkswagen (OTC ADR)
        ],
        "currency": "mixed (EUR/GBP/USD)",
        "data_dir": "data/npy_europe",
        "eval_dir": "data/npy_europe_eval",
    },
}

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
def download_ticker(ticker: str, start: str, end: str) -> np.ndarray | None:
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel("Ticker", axis=1)
        close = df["Close"].values.astype(np.float32).flatten()
        close = close[~np.isnan(close) & (close > 0)]
        return close if len(close) >= 252 else None   # require at least 1 year
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def safe_filename(ticker: str) -> str:
    """Convert ticker to a valid filename."""
    return ticker.replace("^", "IDX_").replace(".", "_") + ".npy"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def prepare_region(name: str, cfg: dict, years: int, root: Path):
    print(f"\n{'='*60}")
    print(f"Region: {name.upper()}  ({cfg['currency']})")
    print(f"{'='*60}")

    end   = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=365 * years)).isoformat()

    for split, tickers in [("train", cfg["train"]), ("eval", cfg["eval"])]:
        out_dir = root / cfg[f"{split}_dir" if split == "eval" else "data_dir"]
        # re-map key names
        out_dir = root / (cfg["eval_dir"] if split == "eval" else cfg["data_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [{split.upper()}] → {out_dir}")

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
            print(f"{len(data):>5} pts  →  {fname}")
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
    print("Done. Training commands:")
    for name in regions:
        cfg = REGIONS[name]
        train_tickers = ",".join(cfg["train"])
        ckpt = f"checkpoints/intl_{name}"
        print(f"""
  # {name.upper()}
  python src/timesfm/finetune_timesfm.py \\
    --data-dir {cfg['data_dir']} \\
    --epochs 50 --batch-size 8 --lr 1e-4 \\
    --optimizer adamw --weight-decay 0.01 \\
    --freeze-layers 17 --accumulation-steps 16 \\
    --warmup-epochs 5 --early-stopping 15 \\
    --save-dir {ckpt} \\
    --max-context 512 --horizon 128 \\
    --no-epoch-ckpts""")

    print(f"\n  Evaluation tickers per region:")
    for name in regions:
        cfg = REGIONS[name]
        print(f"    {name}: {','.join(cfg['eval'])}")


if __name__ == "__main__":
    main()
