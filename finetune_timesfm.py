"""
Finetune TimesFM 2.5 (PyTorch) on financial time series.

Based on Fu, Hirano & Imajo (2024) "Financial Fine-tuning a Large Time
Series Model" (arXiv:2412.09880):
  1. Log-transform: z = log(y + eps)
  2. Random masking: each sample picks a random context window in
     [min_context, max_context] so the model trains on variable-length contexts.
  3. MSE loss in log-space.

Restored from 2026-02-22 backup (confirmed 8.31% 120d MAPE with fixed eval).
Added: --cache-dir, --no-epoch-ckpts, --seed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# Prevent local timesfm/ directory from shadowing stdlib torch.
_script_dir = str(Path(__file__).resolve().parent)
_removed_sys0 = None
if sys.path and sys.path[0] == _script_dir:
    _removed_sys0 = sys.path.pop(0)
torch = importlib.import_module("torch")
from torch.utils.data import Dataset, DataLoader
if _removed_sys0 is not None:
    sys.path.insert(0, _removed_sys0)

try:
    import yfinance as yf
except ImportError:
    yf = None

# Ensure timesfm src is importable from timesfm_finetuning/ location.
_ROOT = Path(__file__).resolve().parent.parent  # workspace root
_SRC = _ROOT / "timesfm" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.torch.util import revin, update_running_stats

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Dataset — log-transform + random context masking (Fu et al. Sec III-B)
# ---------------------------------------------------------------------------

class FinancialTimeSeriesDataset(Dataset):
    """Sliding-window dataset.

    Each item is (context, mask, target) in log-price space.
    Random masking samples a variable-length context window aligned to the
    right edge of the context buffer, so the model trains on context lengths
    uniformly drawn from [min_context, max_context].
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        max_context: int,
        min_context: int,
        horizon: int,
        patch_len: int = 32,
        stride: int | None = None,
    ):
        self.max_context = max_context
        self.min_context = min_context
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride or horizon

        self.windows: list[tuple[int, int]] = []
        total_needed = max_context + horizon
        for s_idx, series in enumerate(series_list):
            if len(series) < min_context + horizon:
                continue
            if len(series) < total_needed:
                self.windows.append((s_idx, 0))
                continue
            for start in range(0, len(series) - total_needed + 1, self.stride):
                self.windows.append((s_idx, start))

        self.series_list = series_list

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        s_idx, start = self.windows[idx]
        series = self.series_list[s_idx]
        total_needed = self.max_context + self.horizon

        if len(series) < total_needed:
            avail = len(series)
            window = series.copy()
        else:
            window = series[start: start + total_needed]
            avail = len(window)

        # Log-transform (paper Eq. 2)
        window = np.log(np.maximum(window, _EPS)).astype(np.float32)

        if avail >= total_needed:
            context_full = window[: self.max_context]
            target = window[self.max_context:]
        else:
            ctx_len = max(min(avail - self.horizon, self.max_context), self.min_context)
            context_full_raw = window[:ctx_len]
            target = window[ctx_len: ctx_len + self.horizon]
            if len(target) < self.horizon:
                target = np.pad(target, (0, self.horizon - len(target)))
            pad_len = self.max_context - len(context_full_raw)
            context_full = np.pad(context_full_raw, (pad_len, 0))

        # Random masking (paper Sec III-B): sample t_end in [min_ctx, max_ctx],
        # t_start in [0, t_end - min_ctx].  Valid region = [t_start, t_end],
        # right-aligned in the context buffer.
        t_end = random.randint(self.min_context, self.max_context)
        t_start = random.randint(0, t_end - self.min_context)
        mask = np.ones(self.max_context, dtype=bool)
        valid_start = self.max_context - t_end + t_start
        valid_end = self.max_context  # always right-aligned
        mask[valid_start:valid_end] = False

        context = context_full.copy()
        context[mask] = 0.0

        # Pad context to multiple of patch_len if needed.
        rem = self.max_context % self.patch_len
        if rem != 0:
            add = self.patch_len - rem
            context = np.pad(context, (add, 0))
            mask = np.concatenate([np.ones(add, dtype=bool), mask])

        return context.astype(np.float32), mask, target.astype(np.float32)


def collate_fn(batch: List):
    contexts, masks, targets = zip(*batch)
    return (
        torch.from_numpy(np.stack(contexts)).float(),
        torch.from_numpy(np.stack(masks)).bool(),
        torch.from_numpy(np.stack(targets)).float(),
    )


# ---------------------------------------------------------------------------
# Training step — patching + RevIN + MSE in log-space (paper Eq. 3)
# ---------------------------------------------------------------------------

def training_step(model_module, inputs, masks, targets, horizon: int):
    device = inputs.device
    B = inputs.shape[0]
    p = model_module.p   # input patch len (32)
    o = model_module.o   # output patch len (128)
    q = model_module.q   # num quantiles + 1

    patched_inputs = inputs.reshape(B, -1, p)
    patched_masks = masks.reshape(B, -1, p)
    num_input_patches = patched_inputs.shape[1]

    n = torch.zeros(B, device=device)
    mu = torch.zeros(B, device=device)
    sigma = torch.zeros(B, device=device)
    patch_mu, patch_sigma = [], []
    for i in range(num_input_patches):
        (n, mu, sigma), _ = update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)

    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

    (_, _, normed_outputs, _), _ = model_module(normed_inputs, patched_masks)

    renormed_outputs = revin(normed_outputs, context_mu, context_sigma, reverse=True)
    renormed_outputs = renormed_outputs.reshape(B, num_input_patches, o, q)

    pred = renormed_outputs[:, -1, :horizon, model_module.aridx]
    loss = torch.nn.functional.mse_loss(pred, targets.to(device))
    return loss


# ---------------------------------------------------------------------------
# Inference helper (used by rolling_eval.py)
# ---------------------------------------------------------------------------

def get_predictions(
    model_module,
    inputs: "torch.Tensor",
    masks: "torch.Tensor",
    horizon: int,
) -> "torch.Tensor":
    """RevIN-normalize, run model, denormalize. Returns (B, horizon) in input space."""
    device = inputs.device
    B = inputs.shape[0]
    p = model_module.p
    o = model_module.o
    q = model_module.q

    patched = inputs.reshape(B, -1, p)
    pmasks = masks.reshape(B, -1, p)

    n = torch.zeros(B, device=device)
    mu = torch.zeros(B, device=device)
    sigma = torch.zeros(B, device=device)
    patch_mu, patch_sigma = [], []
    for i in range(patched.shape[1]):
        (n, mu, sigma), _ = update_running_stats(n, mu, sigma, patched[:, i], pmasks[:, i])
        patch_mu.append(mu)
        patch_sigma.append(sigma)

    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    normed = revin(patched, context_mu, context_sigma, reverse=False)
    normed = torch.where(pmasks, 0.0, normed)

    with torch.no_grad():
        (_, _, normed_out, _), _ = model_module(normed, pmasks)
    if torch.any(torch.isnan(normed_out)):
        normed_out = torch.nan_to_num(normed_out, nan=0.0)

    out = revin(normed_out, context_mu, context_sigma, reverse=True)
    out = out.reshape(B, patched.shape[1], o, q)
    return out[:, -1, :horizon, model_module.aridx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _npy_cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = (ticker.replace("^", "_").replace("/", "_")
                  .replace("\\", "_").replace(".", "_"))
    return cache_dir / f"{safe}.npy"


def load_series_from_cache(tickers: list[str], cache_dir: str) -> list[np.ndarray]:
    """Load pre-cached .npy files by ticker name. Does not download."""
    cd = Path(cache_dir)
    result = []
    for ticker in tickers:
        p = _npy_cache_path(cd, ticker)
        if not p.exists():
            raise FileNotFoundError(
                f"Cache file not found: {p}. "
                f"Provide --cache-dir with pre-downloaded .npy files."
            )
        arr = np.load(p).astype(np.float32).flatten()
        arr = arr[~np.isnan(arr) & (arr > 0)]
        print(f"[cache] {ticker}: {len(arr)} pts")
        result.append(arr)
    return result


def download_tickers(tickers: list[str], years: int = 20) -> list[np.ndarray]:
    if yf is None:
        raise RuntimeError("yfinance required. Install: pip install yfinance")
    import datetime
    end = datetime.datetime.now(datetime.timezone.utc).date()
    start = end - datetime.timedelta(days=365 * years)
    result = []
    for ticker in tickers:
        print(f"Downloading {ticker} {start}→{end}...")
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"  WARNING: no data for {ticker}, skipping")
            continue
        arr = df["Close"].values.astype(np.float32).flatten()
        arr = arr[~np.isnan(arr) & (arr > 0)]
        print(f"  {len(arr)} pts")
        result.append(arr)
    if not result:
        raise RuntimeError("No data downloaded.")
    return result


def load_npy_dir(data_dir: str) -> list[np.ndarray]:
    files = sorted(Path(data_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {data_dir}")
    result = []
    for f in files:
        arr = np.load(f).astype(np.float32).flatten()
        arr = arr[~np.isnan(arr) & (arr > 0)]
        print(f"[data] {f.name}: {len(arr)} pts")
        result.append(arr)
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def freeze_backbone(model_module):
    for param in model_module.tokenizer.parameters():
        param.requires_grad = False
    for layer in model_module.stacked_xf:
        for param in layer.parameters():
            param.requires_grad = False


def freeze_layers(model_module, num_layers: int):
    for i in range(min(num_layers, len(model_module.stacked_xf))):
        for param in model_module.stacked_xf[i].parameters():
            param.requires_grad = False


def count_params(model_module) -> tuple[int, int]:
    total = sum(p.numel() for p in model_module.parameters())
    trainable = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    return total, trainable


def _copy_config(model_id: str, dest_dir: Path):
    src = Path(model_id) / "config.json"
    if src.is_file():
        shutil.copy2(src, dest_dir / "config.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Finetune TimesFM 2.5 (Fu et al. 2024) with random masking")

    # Data — three mutually exclusive loading modes:
    #   1. --tickers + --cache-dir : load .npy by ticker name (preferred)
    #   2. --tickers only           : download via yfinance
    #   3. --data-dir               : load all .npy from directory
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers (e.g. '^GSPC,AAPL')")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory with pre-cached .npy files named by ticker")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing .npy time series files")
    parser.add_argument("--val-ratio", type=float, default=0.25)

    # Model
    parser.add_argument("--model-id", type=str,
                        default="C:/Users/ylchen/workspace/timesfm_finetuning/model",
                        help="Local path or HuggingFace model ID")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-layers", type=int, default=17,
                        help="Freeze first N of 20 transformer layers (Run 4: 17)")

    # Training — paper Table II defaults
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate (Run 4 best: 1e-4 AdamW)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (only used with --optimizer sgd)")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["sgd", "adam", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear warmup epochs (Run 4: 3)")
    parser.add_argument("--accumulation-steps", type=int, default=16,
                        help="Gradient accumulation steps (Run 4: 16, eff. batch=128)")
    parser.add_argument("--early-stopping", type=int, default=15,
                        help="Stop after N epochs with no val improvement")

    # Context / horizon
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--min-context", type=int, default=128,
                        help="Min context length for random masking (paper: 128)")
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding-window stride (default: horizon)")

    # Output
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--no-epoch-ckpts", action="store_true",
                        help="Skip per-epoch checkpoints; only save best/")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.horizon > 128:
        print(f"WARNING: horizon={args.horizon} > 128. Clamping to 128.")
        args.horizon = 128

    patch_len = 32
    if args.max_context % patch_len != 0:
        args.max_context = math.ceil(args.max_context / patch_len) * patch_len
        print(f"Rounded max_context to {args.max_context}")
    if args.min_context % patch_len != 0:
        args.min_context = math.ceil(args.min_context / patch_len) * patch_len
        print(f"Rounded min_context to {args.min_context}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Load data ───────────────────────────────────────────────────────────
    if args.tickers and args.cache_dir:
        tickers = [t.strip() for t in args.tickers.split(",")]
        series_list = load_series_from_cache(tickers, args.cache_dir)
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
        series_list = download_tickers(tickers)
    elif args.data_dir:
        series_list = load_npy_dir(args.data_dir)
    else:
        parser.error("Provide --tickers or --data-dir")

    # ── Dataset ─────────────────────────────────────────────────────────────
    full_ds = FinancialTimeSeriesDataset(
        series_list,
        max_context=args.max_context,
        min_context=args.min_context,
        horizon=args.horizon,
        patch_len=patch_len,
        stride=args.stride,
    )
    print(f"Total windows: {len(full_ds)}")

    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn)
    print(f"Train: {n_train}  Val: {n_val}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading model from: {args.model_id}")
    if os.path.isdir(args.model_id):
        model_wrapper = TimesFM_2p5_200M_torch.from_pretrained(
            args.model_id, local_files_only=True
        )
    else:
        model_wrapper = TimesFM_2p5_200M_torch.from_pretrained(args.model_id)

    model = model_wrapper.model
    model.to(device)

    if args.freeze_backbone:
        freeze_backbone(model)
    elif args.freeze_layers > 0:
        freeze_layers(model, args.freeze_layers)
        print(f"Froze first {args.freeze_layers}/{len(model.stacked_xf)} layers")

    total_p, trainable_p = count_params(model)
    print(f"Params: {total_p:,} total | {trainable_p:,} trainable "
          f"({100*trainable_p/total_p:.1f}%)")

    # ── Optimizer ───────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"AdamW lr={args.lr} wd={args.weight_decay}")
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        print(f"Adam lr={args.lr}")
    else:
        optimizer = torch.optim.SGD(
            trainable_params, lr=args.lr, momentum=args.momentum)
        print(f"SGD lr={args.lr} momentum={args.momentum}")

    accum = args.accumulation_steps
    steps_per_epoch = max(1, len(train_dl) // accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    if accum > 1:
        print(f"Grad accumulation: {accum} "
              f"(effective batch={args.batch_size * accum})")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_epoch = 0
    global_step = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (inputs, masks, targets) in enumerate(train_dl, start=1):
            inputs = inputs.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            loss = training_step(model, inputs, masks, targets, horizon=args.horizon)
            (loss / accum).backward()
            epoch_loss += loss.item()

            if step % accum == 0 or step == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} step {step}/{len(train_dl)} | "
                      f"loss={loss.item():.6f} lr={lr:.2e}")

        avg_train = epoch_loss / max(1, len(train_dl))
        print(f"Epoch {epoch:3d} train | loss={avg_train:.6f}")

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks, targets in val_dl:
                val_loss += training_step(
                    model, inputs.to(device), masks.to(device),
                    targets.to(device), horizon=args.horizon
                ).item()

        avg_val = val_loss / max(1, len(val_dl))
        print(f"Epoch {epoch:3d} val   | loss={avg_val:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train,
            "val_loss": avg_val,
            "lr": scheduler.get_last_lr()[0],
        })
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpointing ─────────────────────────────────────────────────────
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_val_epoch = epoch

        if not args.no_epoch_ckpts:
            out_dir = save_dir / f"epoch_{epoch}"
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                model_wrapper._save_pretrained(str(out_dir))
                _copy_config(args.model_id, out_dir)
            except Exception:
                torch.save(model.state_dict(), out_dir / "model.pt")
            tag = " (best)" if is_best else ""
            print(f"Saved epoch {epoch}{tag} → {out_dir}")

        if is_best:
            best_dir = save_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            try:
                model_wrapper._save_pretrained(str(best_dir))
                _copy_config(args.model_id, best_dir)
            except Exception:
                torch.save(model.state_dict(), best_dir / "model.pt")
            print(f"  ✓ New best val_loss={best_val_loss:.6f} → {best_dir}")

        with open(save_dir / "final_val_metric.json", "w") as f:
            json.dump({"best_val_loss": best_val_loss,
                       "best_epoch": best_val_epoch}, f, indent=2)

        if args.early_stopping > 0 and (epoch - best_val_epoch) >= args.early_stopping:
            print(f"\nEarly stopping: no val improvement for "
                  f"{args.early_stopping} epochs (best epoch {best_val_epoch})")
            break

    # ── Save last checkpoint ─────────────────────────────────────────────────
    last_dir = save_dir / "last"
    last_dir.mkdir(parents=True, exist_ok=True)
    try:
        model_wrapper._save_pretrained(str(last_dir))
        _copy_config(args.model_id, last_dir)
    except Exception:
        torch.save(model.state_dict(), last_dir / "model.pt")
    print(f"Saved last checkpoint → {last_dir}")

    if best_val_loss > 0.015:
        print(f"WARNING: best val_loss={best_val_loss:.6f} > 0.015 "
              f"— may produce NaN at inference")

    print(f"\nDone. Best val_loss={best_val_loss:.6f} (epoch {best_val_epoch})")


if __name__ == "__main__":
    main()
