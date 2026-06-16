"""
Train TimesFM on international markets (Japan, China, Europe).

For each region:
  1. Train with --no-epoch-ckpts (only best/ and last/ saved)
  2. Evaluate best/ vs pretrained baseline on held-out eval tickers
  3. If finetuned MAPE >= baseline MAPE at 120d → regression → remove best/

Usage:
  cd C:/Users/ylchen/workspace/timesfm
  python scripts/train_intl.py
  python scripts/train_intl.py --regions japan europe
  python scripts/train_intl.py --regions china --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yfinance as yf
import timesfm

# ---------------------------------------------------------------------------
# Region config
# ---------------------------------------------------------------------------
REGIONS = {
    "usa": {
        "train_dir":  ROOT / "finetune_usa/data/train",
        "eval_dir":   ROOT / "finetune_usa/data/eval",
        "ckpt_dir":   ROOT / "finetune_usa/checkpoints/run1",
        "results_dir": ROOT / "finetune_usa/finetune_expts",
        "eval_tickers": ["AAPL", "GOOG", "MSFT"],
    },
    "japan": {
        "train_dir":  ROOT / "finetune_japan/data/train",
        "eval_dir":   ROOT / "finetune_japan/data/eval",
        "ckpt_dir":   ROOT / "finetune_japan/checkpoints/run1",
        "results_dir": ROOT / "finetune_japan/finetune_expts",
        "eval_tickers": ["8035.T", "6902.T", "4661.T"],
    },
    "china": {
        "train_dir":  ROOT / "finetune_china/data/train",
        "eval_dir":   ROOT / "finetune_china/data/eval",
        "ckpt_dir":   ROOT / "finetune_china/checkpoints/run1",
        "results_dir": ROOT / "finetune_china/finetune_expts",
        "eval_tickers": ["0388.HK", "2382.HK", "1177.HK"],
    },
    "europe": {
        "train_dir":  ROOT / "finetune_europe/data/train",
        "eval_dir":   ROOT / "finetune_europe/data/eval",
        "ckpt_dir":   ROOT / "finetune_europe/checkpoints/run1",
        "results_dir": ROOT / "finetune_europe/finetune_expts",
        "eval_tickers": ["UL", "EADSF", "VWAGY"],
    },
}

BASELINE_MODEL = "google/timesfm-2.5-200m-pytorch"
MAX_CONTEXT    = 512
EVAL_HORIZON   = 120   # primary metric: 120-day MAPE
TRAIN_ARGS = [
    "--epochs", "5",
    "--batch-size", "8",
    "--lr", "1e-5",
    "--optimizer", "adamw",
    "--weight-decay", "0.01",
    "--freeze-layers", "22",
    "--accumulation-steps", "16",
    "--warmup-epochs", "0",
    "--early-stopping", "5",
    "--max-context", "512",
    "--min-context", "256",
    "--horizon", "128",
    "--stride", "32",
    "--horizon-loss-decay", "0.1",
    "--no-epoch-ckpts",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str, file=None):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if file:
        file.write(line + "\n")
        file.flush()


def download_eval_data(tickers: list[str], years: int = 5) -> dict[str, np.ndarray]:
    """Download price series for eval tickers."""
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=365 * years)
    result = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start.isoformat(), end=end.isoformat(),
                             progress=False, auto_adjust=True)
            if df is None or df.empty:
                print(f"  WARNING: no data for {t}")
                continue
            import pandas as pd
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)
            close = df["Close"].values.astype(np.float32).flatten()
            close = close[~np.isnan(close) & (close > 0)]
            result[t] = close
            print(f"  {t}: {len(close)} pts")
        except Exception as e:
            print(f"  ERROR {t}: {e}")
    return result


def mape(pred: np.ndarray, actual: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])) * 100)


def build_contexts_and_actuals(
    series_dict: dict[str, np.ndarray],
    max_context: int,
    horizon: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Split each series into (context, actuals) using most recent data."""
    contexts, actuals, tickers = [], [], []
    for t, series in series_dict.items():
        if len(series) < max_context + horizon:
            print(f"  {t}: too short ({len(series)} pts), skipping eval")
            continue
        context = series[-(max_context + horizon):-horizon]
        actual  = series[-horizon:]
        contexts.append(context)
        actuals.append(actual)
        tickers.append(t)
    return contexts, actuals, tickers


def eval_model(
    model_id: str,
    local: bool,
    contexts: list[np.ndarray],
    actuals: list[np.ndarray],
    tickers: list[str],
    horizon: int,
) -> dict:
    """Load model, forecast, compute per-ticker and overall MAPE."""
    if local:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_id, local_files_only=True)
    else:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)

    model.compile(timesfm.ForecastConfig(
        max_context=MAX_CONTEXT,
        max_horizon=horizon,
        normalize_inputs=True,
        force_flip_invariance=True,
        infer_is_positive=True,
    ))

    inputs = [c.tolist() for c in contexts]
    preds, _ = model.forecast(horizon=horizon, inputs=inputs)
    del model

    results = {}
    mapes = []
    for t, pred_row, actual_row in zip(tickers, preds, actuals):
        h = min(horizon, len(actual_row))
        m = mape(np.array(pred_row[:h]), actual_row[:h])
        results[t] = round(m, 3)
        mapes.append(m)
    results["overall"] = round(float(np.mean(mapes)), 3)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_region(name: str, cfg: dict, dry_run: bool):
    cfg["results_dir"].mkdir(parents=True, exist_ok=True)
    log_path = cfg["results_dir"] / "train_log.txt"

    with open(log_path, "w") as logf:
        log(f"{'='*60}", logf)
        log(f"REGION: {name.upper()}", logf)
        log(f"{'='*60}", logf)

        # ── Step 1: Train ────────────────────────────────────────
        log(f"Step 1: Training...", logf)
        cfg["ckpt_dir"].mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(ROOT / "src/timesfm/finetune_timesfm.py"),
            "--data-dir", str(cfg["train_dir"]),
            "--save-dir", str(cfg["ckpt_dir"]),
        ] + TRAIN_ARGS

        log(f"Command: {' '.join(cmd)}", logf)

        if dry_run:
            log("DRY RUN: skipping training.", logf)
            best_exists = (cfg["ckpt_dir"] / "best").exists()
        else:
            proc = subprocess.run(cmd, capture_output=False, text=True)
            if proc.returncode != 0:
                log(f"ERROR: training failed (exit {proc.returncode})", logf)
                return
            log("Training complete.", logf)
            best_exists = (cfg["ckpt_dir"] / "best").exists()

        if not best_exists:
            log("WARNING: no best/ checkpoint saved — val_loss never improved.", logf)
            return

        # ── Step 2: Download eval data ───────────────────────────
        log(f"\nStep 2: Downloading eval data for {cfg['eval_tickers']}...", logf)
        series = download_eval_data(cfg["eval_tickers"])
        if not series:
            log("ERROR: no eval data downloaded.", logf)
            return

        contexts, actuals, tickers = build_contexts_and_actuals(
            series, MAX_CONTEXT, EVAL_HORIZON)
        if not tickers:
            log("ERROR: all eval tickers too short.", logf)
            return

        log(f"Eval tickers: {tickers}", logf)

        # ── Step 3: Evaluate baseline ────────────────────────────
        log(f"\nStep 3: Evaluating pretrained baseline ({BASELINE_MODEL})...", logf)
        baseline_results = eval_model(
            BASELINE_MODEL, local=False, contexts=contexts,
            actuals=actuals, tickers=tickers, horizon=EVAL_HORIZON)
        log(f"Baseline MAPE: {baseline_results}", logf)

        # ── Step 4: Evaluate finetuned best/ ────────────────────
        best_dir = str(cfg["ckpt_dir"] / "best")
        log(f"\nStep 4: Evaluating finetuned best/ ({best_dir})...", logf)
        ft_results = eval_model(
            best_dir, local=True, contexts=contexts,
            actuals=actuals, tickers=tickers, horizon=EVAL_HORIZON)
        log(f"Finetuned MAPE: {ft_results}", logf)

        # ── Step 5: Compare and decide ───────────────────────────
        baseline_mape = baseline_results["overall"]
        ft_mape       = ft_results["overall"]
        improvement   = baseline_mape - ft_mape   # positive = better

        log(f"\nStep 5: Decision", logf)
        log(f"  Baseline MAPE: {baseline_mape:.3f}%", logf)
        log(f"  Finetuned MAPE: {ft_mape:.3f}%", logf)
        log(f"  Improvement: {improvement:+.3f}pp", logf)

        if improvement <= 0:
            log(f"  REGRESSION detected ({improvement:+.3f}pp). Removing best/ checkpoint.", logf)
            if not dry_run:
                shutil.rmtree(cfg["ckpt_dir"] / "best")
                log(f"  Removed: {cfg['ckpt_dir'] / 'best'}", logf)
            else:
                log(f"  DRY RUN: would remove {cfg['ckpt_dir'] / 'best'}", logf)
            verdict = "REGRESSED"
        else:
            log(f"  IMPROVEMENT: {improvement:+.3f}pp. Keeping best/ checkpoint.", logf)
            verdict = "IMPROVED"

        # ── Save results JSON ────────────────────────────────────
        results = {
            "region": name,
            "generated": datetime.date.today().isoformat(),
            "eval_horizon": EVAL_HORIZON,
            "eval_tickers": tickers,
            "baseline_mape": baseline_results,
            "finetuned_mape": ft_results,
            "improvement_pp": round(improvement, 3),
            "verdict": verdict,
            "best_checkpoint_kept": verdict == "IMPROVED",
        }
        out_path = cfg["results_dir"] / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        log(f"\nResults saved: {out_path}", logf)

        # ── Disk usage ───────────────────────────────────────────
        log(f"\nCheckpoint disk usage:", logf)
        for subdir in ["best", "last"]:
            p = cfg["ckpt_dir"] / subdir
            if p.exists():
                size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
                log(f"  {subdir}/  {size_mb:.0f} MB", logf)
            else:
                log(f"  {subdir}/  (removed or not saved)", logf)

        log(f"\nDone: {name.upper()} — {verdict}", logf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", nargs="+",
                        choices=list(REGIONS.keys()),
                        default=list(REGIONS.keys()),
                        help="Which regions to train (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip training; only evaluate existing checkpoints")
    args = parser.parse_args()

    print(f"Training regions: {args.regions}")
    print(f"Dry run: {args.dry_run}")
    print(f"Storage policy: --no-epoch-ckpts | remove best/ if MAPE regresses\n")

    for name in args.regions:
        run_region(name, REGIONS[name], dry_run=args.dry_run)

    print("\nAll regions complete.")


if __name__ == "__main__":
    main()
