"""
Rolling-window evaluation for finetuned TimesFM 2.5 checkpoints.

Fixed from previous version:
- masks bug: was torch.ones_like (zeroed all context); now torch.zeros bool
- log transform: model was trained on log(price); inputs must be log-transformed
  before get_predictions and predictions exp-transformed back before MAPE
- cache format: reads .npy files (not CSV) to match finetune_timesfm.py cache
- model loading: uses from_pretrained + load_state_dict (not bare constructor)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Make finetune_timesfm importable
_FT_DIR = Path(__file__).resolve().parents[1]
if str(_FT_DIR) not in sys.path:
    sys.path.insert(0, str(_FT_DIR))

from finetune_timesfm import TimesFM_2p5_200M_torch, get_predictions

_EPS = 1e-9
_BASE_MODEL = "C:/Users/ylchen/workspace/timesfm_finetuning/model"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _npy_cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = (ticker.replace("^", "_").replace("/", "_")
                  .replace("\\", "_").replace(".", "_"))
    return cache_dir / f"{safe}.npy"


def load_series(tickers: list[str], cache_dir: str) -> dict[str, np.ndarray]:
    """Load close-price arrays from .npy cache. Downloads via yfinance if missing."""
    cd = Path(cache_dir)
    cd.mkdir(parents=True, exist_ok=True)
    result = {}
    for ticker in tickers:
        p = _npy_cache_path(cd, ticker)
        if p.exists():
            arr = np.load(p).astype(np.float32)
            arr = arr[~np.isnan(arr) & (arr > 0)]
            print(f"[cache] {ticker}: {len(arr)} pts")
            result[ticker] = arr
        else:
            try:
                import yfinance as yf
                import datetime
                end = datetime.datetime.now(datetime.timezone.utc).date()
                start = end - datetime.timedelta(days=365 * 20)
                print(f"Downloading {ticker}...")
                df = yf.download(ticker, start=str(start), end=str(end),
                                 progress=False, auto_adjust=True)
                if df is None or df.empty:
                    print(f"  WARNING: no data for {ticker}", file=sys.stderr)
                    continue
                arr = df["Close"].values.astype(np.float32).flatten()
                arr = arr[~np.isnan(arr) & (arr > 0)]
                np.save(p, arr)
                result[ticker] = arr
            except Exception as e:
                print(f"  ERROR loading {ticker}: {e}", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, _EPS))) * 100)


def dir_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.mean(
        np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])
    ) * 100)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_series(
    model: torch.nn.Module,
    series: np.ndarray,
    max_context: int,
    horizons: list[int],
    n_windows: int,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Rolling-window evaluation for one series. Returns MAPE and DirAcc per horizon."""
    max_h = max(horizons)
    results: dict[int, dict[str, list]] = {h: {"mape": [], "dir_acc": []} for h in horizons}

    if len(series) < max_context + max_h:
        print(f"  Skipping: only {len(series)} pts (need {max_context + max_h})", file=sys.stderr)
        return {h: {"mape": float("nan"), "dir_acc": float("nan")} for h in horizons}

    # Anchor i=0 at len(series)-max_h so every window has room for a full max_h forecast.
    # Step backwards by max_h//n_windows each window so we cover a spread of dates.
    step = max(1, max_h // n_windows)
    for i in range(n_windows):
        end_idx = (len(series) - max_h) - i * step
        if end_idx < max_context:
            continue

        ctx_arr = series[max(0, end_idx - max_context):end_idx].astype(np.float32)
        if len(ctx_arr) < 2:
            continue

        # Pad context to max_context if series is short
        if len(ctx_arr) < max_context:
            pad = np.zeros(max_context - len(ctx_arr), dtype=np.float32)
            ctx_arr = np.concatenate([pad, ctx_arr])
            n_valid = max_context - len(pad)
        else:
            n_valid = max_context

        # Log-transform context (model was trained in log-space)
        log_ctx = np.log(np.maximum(ctx_arr, _EPS))

        inputs = torch.from_numpy(log_ctx).unsqueeze(0).to(device)
        # mask=True means padding; only prefix is padding when series is short
        masks = torch.zeros(1, max_context, dtype=torch.bool, device=device)
        if n_valid < max_context:
            masks[0, :max_context - n_valid] = True

        with torch.no_grad():
            log_preds = get_predictions(model, inputs, masks, max_h)[0].cpu().numpy()

        # Convert log-space predictions back to price space
        preds_price = np.exp(log_preds)

        for h in horizons:
            if end_idx + h > len(series):
                continue
            target = series[end_idx:end_idx + h]
            pred = preds_price[:h]
            results[h]["mape"].append(mape(target, pred))
            results[h]["dir_acc"].append(dir_acc(target, pred))

    return {
        h: {
            "mape": float(np.mean(v["mape"])) if v["mape"] else float("nan"),
            "dir_acc": float(np.mean(v["dir_acc"])) if v["dir_acc"] else float("nan"),
        }
        for h, v in results.items()
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pa = argparse.ArgumentParser(description="Rolling-window eval for finetuned TimesFM checkpoint")
    pa.add_argument("--checkpoint", required=True,
                    help="Path to checkpoint directory containing model.safetensors")
    pa.add_argument("--tickers", required=True, help="Comma-separated eval tickers")
    pa.add_argument("--horizons", default="5,14,30,60,90,120",
                    help="Comma-separated forecast horizons")
    pa.add_argument("--cache-dir", required=True, help="Directory with .npy ticker files")
    pa.add_argument("--max-context", type=int, default=512)
    pa.add_argument("--n-windows", type=int, default=15)
    pa.add_argument("--base-model", type=str, default=_BASE_MODEL,
                    help="Path to pre-trained TimesFM checkpoint (for model architecture).")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--strict-tickers", action="store_true")
    pa.add_argument("--results-file", type=str, default=None,
                    help="Override output JSON path (default: <checkpoint>/rolling_eval.json)")
    args = pa.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt_path = Path(args.checkpoint) / "model.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {args.checkpoint}")

    wrapper = TimesFM_2p5_200M_torch.from_pretrained(args.base_model)
    model = wrapper.model
    state = load_file(str(ckpt_path), device=str(device))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Model loaded from {ckpt_path}")

    tickers = [t.strip() for t in args.tickers.split(",")]
    horizons = sorted(int(h) for h in args.horizons.split(","))

    data = load_series(tickers, args.cache_dir)
    missing = [t for t in tickers if t not in data]
    if missing:
        print(f"WARNING: could not load {missing}", file=sys.stderr)
        if args.strict_tickers:
            sys.exit(1)

    all_results: dict[str, dict] = {}
    for ticker in tickers:
        if ticker not in data:
            continue
        print(f"\nEvaluating {ticker}...")
        r = evaluate_series(model, data[ticker], args.max_context, horizons, args.n_windows, device)
        all_results[ticker] = r
        for h in horizons:
            print(f"  h={h:3d}: MAPE={r[h]['mape']:6.3f}%  DirAcc={r[h]['dir_acc']:5.1f}%")

    # Aggregate
    print("\nAverage across tickers:")
    avg: dict[int, dict[str, float]] = {}
    for h in horizons:
        mapes = [all_results[t][h]["mape"] for t in all_results if not np.isnan(all_results[t][h]["mape"])]
        daccs = [all_results[t][h]["dir_acc"] for t in all_results if not np.isnan(all_results[t][h]["dir_acc"])]
        m = float(np.mean(mapes)) if mapes else float("nan")
        d = float(np.mean(daccs)) if daccs else float("nan")
        avg[h] = {"mape": m, "dir_acc": d}
        print(f"  h={h:3d}: MAPE={m:6.3f}%  DirAcc={d:5.1f}%")

    output = {
        "args": vars(args),
        "per_ticker": all_results,
        "average": {str(k): v for k, v in avg.items()},
        "primary_120d_mape": avg.get(120, {}).get("mape", float("nan")),
    }
    result_path = Path(args.results_file) if args.results_file else Path(args.checkpoint) / "rolling_eval.json"
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {result_path}")
    print(f"PRIMARY METRIC — 120d MAPE: {output['primary_120d_mape']:.3f}%")


if __name__ == "__main__":
    main()
