"""
Finetune TimesFM 2.5 (PyTorch) on financial time series.

Based on Fu, Hirano & Imajo (2024) "Financial Fine-tuning a Large Time
Series Model" (arXiv:2412.09880). Key ideas from the paper:

  1. Log-transform: z = log(y) is fed as input; MSE loss computed in log-space.
  2. Random masking: each sample randomly picks a context length in
     [min_context, max_context] to prevent overfitting.
  3. Full-model continual pre-training with SGD + momentum.

Usage examples:
  # Finetune on S&P500 (downloads via yfinance):
  python src/timesfm/finetune_timesfm.py --tickers "^GSPC" --epochs 100

  # Finetune on multiple tickers:
  python src/timesfm/finetune_timesfm.py --tickers "^GSPC,AAPL,MSFT" --epochs 100

  # Finetune on pre-saved .npy files:
  python src/timesfm/finetune_timesfm.py --data-dir data/npy_series --epochs 100

  # Last-layer only (freeze backbone):
  python src/timesfm/finetune_timesfm.py --tickers "^GSPC" --freeze-backbone
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np

# Handle torch import to avoid local `timesfm/torch` shadowing.
import importlib

script_dir = str(Path(__file__).resolve().parent)
_removed_sys0 = None
if sys.path and sys.path[0] == script_dir:
    _removed_sys0 = sys.path.pop(0)
torch = importlib.import_module("torch")
from torch.utils.data import Dataset, DataLoader
if _removed_sys0 is not None:
    sys.path.insert(0, _removed_sys0)

try:
    import yfinance as yf
except ImportError:
    yf = None

# Ensure repository `src` is importable when running this script directly.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.torch.util import revin, update_running_stats

_EPS = 1e-9  # small constant to avoid log(0)


# ---------------------------------------------------------------------------
# Dataset — log-transformed with random context masking (per paper Sec III)
# ---------------------------------------------------------------------------

class FinancialTimeSeriesDataset(Dataset):
    """Sliding-window dataset with log-transform and random context masking.

    Per the paper:
      - Input data is log-transformed: z = log(y + eps)
      - Each sample randomly picks t_end in [min_context, max_context],
        then t_start in [0, t_end - min_context].  The segment
        [t_start, t_end] becomes the context; the next output_len points
        are the target.  Points before t_start are masked.
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

        # Build window index: each window is max_context + horizon long.
        # Random masking is applied at __getitem__ time.
        self.windows: list[tuple[int, int]] = []
        total_needed = max_context + horizon
        for s_idx, series in enumerate(series_list):
            if len(series) < min_context + horizon:
                continue  # series too short
            if len(series) < total_needed:
                self.windows.append((s_idx, 0))
                continue
            for start in range(0, len(series) - total_needed + 1, self.stride):
                self.windows.append((s_idx, start))

        self.series_list = series_list

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        s_idx, start = self.windows[idx]
        series = self.series_list[s_idx]
        total_needed = self.max_context + self.horizon

        # Extract the full window (may need padding for short series)
        if len(series) < total_needed:
            avail = len(series)
            window = series.copy()
        else:
            window = series[start : start + total_needed]
            avail = len(window)

        # Log-transform: z = log(y + eps)  (paper Eq. 2)
        window = np.log(np.maximum(window, _EPS)).astype(np.float32)

        # Split context vs target
        if avail >= total_needed:
            context_full = window[:self.max_context]
            target = window[self.max_context:]
        else:
            # Short series: use what we have
            ctx_len = min(avail - self.horizon, self.max_context)
            ctx_len = max(ctx_len, self.min_context)
            context_full_raw = window[:ctx_len]
            target = window[ctx_len : ctx_len + self.horizon]
            # Pad target if needed
            if len(target) < self.horizon:
                target = np.pad(target, (0, self.horizon - len(target)),
                                constant_values=0.0)
            # Front-pad context to max_context
            pad_len = self.max_context - len(context_full_raw)
            context_full = np.pad(context_full_raw, (pad_len, 0),
                                  constant_values=0.0)

        # Random masking (paper Sec III-B):
        # Sample t_end in [min_context, max_context], then
        # t_start in [0, t_end - min_context].
        # The valid region is [t_start, t_end] within the context.
        # Everything else is masked (True = masked).
        t_end = random.randint(self.min_context, self.max_context)
        t_start = random.randint(0, t_end - self.min_context)
        # Build mask: True = masked/padded, False = valid
        mask = np.ones(self.max_context, dtype=bool)
        # The valid region is the last `valid_len` points before position t_end
        # In the context array, valid region is [max_context - t_end + t_start, max_context - t_end + t_end)
        # = [max_context - valid_len - (t_end - t_end), ...] — simpler:
        # Place valid data right-aligned at position (max_context - (max_context - t_end)) = t_end
        valid_start_in_ctx = self.max_context - t_end + t_start
        valid_end_in_ctx = self.max_context - t_end + t_end
        mask[valid_start_in_ctx:valid_end_in_ctx] = False

        # Zero out masked positions in context
        context = context_full.copy()
        context[mask] = 0.0

        # Ensure context is multiple of patch_len by front-padding
        rem = self.max_context % self.patch_len
        if rem != 0:
            add = self.patch_len - rem
            context = np.pad(context, (add, 0), constant_values=0.0)
            mask = np.concatenate([np.ones(add, dtype=bool), mask])

        return (
            context.astype(np.float32),
            mask,
            target.astype(np.float32),
        )


def collate_fn(batch: List):
    contexts, masks, targets = zip(*batch)
    return (
        torch.from_numpy(np.stack(contexts)).float(),
        torch.from_numpy(np.stack(masks)).bool(),
        torch.from_numpy(np.stack(targets)).float(),
    )


# ---------------------------------------------------------------------------
# Training step (proper forward with patching + ReVIN)
# ---------------------------------------------------------------------------

def training_step(
    model_module,
    inputs,
    masks,
    targets,
    horizon: int,
):
    """One forward pass with patching and ReVIN; MSE loss in log-space.

    Inputs and targets are already log-transformed by the dataset.

    Args:
        model_module: The inner nn.Module (TimesFM_2p5_200M_torch_module).
        inputs: (B, context_len) log-transformed time series context.
        masks: (B, context_len) bool masks (True=masked).
        targets: (B, horizon) log-transformed ground truth future values.
        horizon: Forecast horizon length.
    """
    device = inputs.device
    B = inputs.shape[0]
    p = model_module.p   # input patch len (32)
    o = model_module.o   # output patch len (128)
    q = model_module.q   # num quantiles + 1 (10)

    # 1. Patch the inputs: (B, context) -> (B, num_patches, p)
    patched_inputs = inputs.reshape(B, -1, p)
    patched_masks = masks.reshape(B, -1, p)
    num_input_patches = patched_inputs.shape[1]

    # 2. Compute running stats per patch for ReVIN
    n = torch.zeros(B, device=device)
    mu = torch.zeros(B, device=device)
    sigma = torch.zeros(B, device=device)
    patch_mu = []
    patch_sigma = []
    for i in range(num_input_patches):
        (n, mu, sigma), _ = update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)

    context_mu = torch.stack(patch_mu, dim=1)    # (B, num_patches)
    context_sigma = torch.stack(patch_sigma, dim=1)  # (B, num_patches)

    # 3. ReVIN normalize
    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

    # 4. Forward pass (with gradients)
    (_, _, normed_outputs, _), _ = model_module(normed_inputs, patched_masks)

    # 5. Reverse ReVIN and reshape
    renormed_outputs = revin(normed_outputs, context_mu, context_sigma, reverse=True)
    renormed_outputs = renormed_outputs.reshape(B, num_input_patches, o, q)

    # 6. Extract point forecast from last patch (median at index aridx=5)
    aridx = model_module.aridx
    pred = renormed_outputs[:, -1, :horizon, aridx]  # (B, horizon)

    # 7. MSE loss — both pred and targets are already in log-space (paper Eq. 3)
    targets = targets.to(device)
    loss = torch.nn.functional.mse_loss(pred, targets)

    return loss


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def download_tickers(
    tickers: list[str],
    years: int = 20,
) -> list[np.ndarray]:
    """Download daily Close prices for given tickers via yfinance."""
    if yf is None:
        raise RuntimeError("yfinance is required. Install with: pip install yfinance")

    import datetime
    end = datetime.datetime.now(datetime.timezone.utc).date()
    start = end - datetime.timedelta(days=365 * years)

    series_list = []
    for ticker in tickers:
        print(f"Downloading {ticker} from {start} to {end}...")
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
        if df is None or df.empty:
            print(f"  WARNING: Failed to download {ticker}, skipping.")
            continue
        close = df["Close"].values.astype(np.float32).flatten()
        close = close[~np.isnan(close)]
        # Remove zeros/negatives that would break log transform
        close = close[close > 0]
        series_list.append(close)
        print(f"  Got {len(close)} data points for {ticker}")

    if not series_list:
        raise RuntimeError("No data downloaded for any ticker.")
    return series_list


def load_npy_dir(data_dir: str) -> list[np.ndarray]:
    """Load all .npy files from a directory as a list of 1-D arrays."""
    files = sorted(Path(data_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
    series_list = []
    for f in files:
        arr = np.load(f).astype(np.float32).flatten()
        arr = arr[~np.isnan(arr)]
        arr = arr[arr > 0]  # must be positive for log transform
        series_list.append(arr)
        print(f"Loaded {f.name}: {len(arr)} points")
    return series_list


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_backbone(model_module):
    """Freeze tokenizer and transformer layers, leave output heads trainable."""
    for param in model_module.tokenizer.parameters():
        param.requires_grad = False
    for layer in model_module.stacked_xf:
        for param in layer.parameters():
            param.requires_grad = False


def freeze_layers(model_module, num_layers: int):
    """Freeze the first `num_layers` transformer layers (of 20 total)."""
    for i in range(min(num_layers, len(model_module.stacked_xf))):
        for param in model_module.stacked_xf[i].parameters():
            param.requires_grad = False


def count_params(model_module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model_module.parameters())
    trainable = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    return total, trainable


def _copy_config_json(model_id: str, dest_dir: Path):
    """Copy config.json from the source model directory to a checkpoint dir."""
    src = Path(model_id) / "config.json"
    if src.is_file():
        shutil.copy2(src, dest_dir / "config.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Finetune TimesFM 2.5 on financial time series "
                    "(Fu, Hirano & Imajo 2024)")
    # Data
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated tickers to download via yfinance")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing .npy time series files")
    parser.add_argument("--val-ratio", type=float, default=0.25,
                        help="Fraction of windows for validation (paper: 0.25)")
    # Model
    parser.add_argument("--model-id", type=str, default="google/timesfm-2.5-200m-pytorch",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze transformer backbone (last-layer only finetune)")
    # Training — defaults from paper Table II
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size (paper: 1024 on 8xV100)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate (paper: 5e-4)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum (paper: 0.9)")
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adam", "adamw"],
                        help="Optimizer: sgd (paper), adam, or adamw (adam + weight decay)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW (default: 0.01)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=25,
                        help="Linear warmup epochs (paper: 25)")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--freeze-layers", type=int, default=0,
                        help="Freeze first N transformer layers (0-20, default: 0)")
    parser.add_argument("--early-stopping", type=int, default=0,
                        help="Stop after N epochs with no val improvement (0=disabled)")
    # Architecture — from paper Table II
    parser.add_argument("--max-context", type=int, default=512,
                        help="Max context length (paper: 512)")
    parser.add_argument("--min-context", type=int, default=128,
                        help="Min context length for random masking (paper: 128)")
    parser.add_argument("--horizon", type=int, default=128,
                        help="Forecast horizon / output length (paper: 128)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride (default: horizon)")
    # Output
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Validate horizon
    if args.horizon > 128:
        print(f"WARNING: horizon={args.horizon} > output_patch_len=128. Clamping to 128.")
        args.horizon = 128

    # Round context lengths to multiples of patch_len=32
    patch_len = 32
    if args.max_context % patch_len != 0:
        args.max_context = math.ceil(args.max_context / patch_len) * patch_len
        print(f"Rounded max_context to {args.max_context}")
    if args.min_context % patch_len != 0:
        args.min_context = math.ceil(args.min_context / patch_len) * patch_len
        print(f"Rounded min_context to {args.min_context}")

    # ---- Load data ----
    if args.tickers:
        ticker_list = [t.strip() for t in args.tickers.split(",")]
        series_list = download_tickers(ticker_list)
    elif args.data_dir:
        series_list = load_npy_dir(args.data_dir)
    else:
        parser.error("Provide either --tickers or --data-dir")

    # ---- Create datasets ----
    full_ds = FinancialTimeSeriesDataset(
        series_list,
        max_context=args.max_context,
        min_context=args.min_context,
        horizon=args.horizon,
        patch_len=patch_len,
        stride=args.stride,
    )
    print(f"Total sliding windows: {len(full_ds)}")

    # Train/val split (paper: 75/25)
    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_fn)
    print(f"Train: {n_train} samples, Val: {n_val} samples")

    # ---- Load model ----
    print(f"Loading model from: {args.model_id}")
    if os.path.isdir(args.model_id):
        model_wrapper = TimesFM_2p5_200M_torch.from_pretrained(
            args.model_id, local_files_only=True
        )
    else:
        model_wrapper = TimesFM_2p5_200M_torch.from_pretrained(args.model_id)

    model = model_wrapper.model
    model.to(device)

    # ---- Freeze layers (optional) ----
    if args.freeze_backbone:
        freeze_backbone(model)
    elif args.freeze_layers > 0:
        freeze_layers(model, args.freeze_layers)
        print(f"Froze first {args.freeze_layers} of {len(model.stacked_xf)} transformer layers")

    total_p, trainable_p = count_params(model)
    print(f"Total params: {total_p:,} | Trainable params: {trainable_p:,} "
          f"({100*trainable_p/total_p:.1f}%)")

    # ---- Optimizer ----
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"Using AdamW optimizer (lr={args.lr}, weight_decay={args.weight_decay})")
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
        print(f"Using Adam optimizer (lr={args.lr})")
    else:
        optimizer = torch.optim.SGD(
            trainable_params, lr=args.lr, momentum=args.momentum,
        )
        print(f"Using SGD optimizer (lr={args.lr}, momentum={args.momentum})")

    # LR schedule: linear warmup then cosine decay (paper Sec III)
    # With gradient accumulation, optimizer steps = batch steps / accum_steps
    accum = args.accumulation_steps
    steps_per_epoch = len(train_dl) // accum  # optimizer steps per epoch
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    if accum > 1:
        print(f"Gradient accumulation: {accum} steps "
              f"(effective batch = {args.batch_size * accum})")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Training ----
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_val_epoch = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (inputs, masks, targets) in enumerate(train_dl, start=1):
            inputs = inputs.to(device)
            masks = masks.to(device)
            targets = targets.to(device)

            loss = training_step(model, inputs, masks, targets,
                                 horizon=args.horizon)

            # Scale loss for gradient accumulation
            scaled_loss = loss / accum
            scaled_loss.backward()

            epoch_loss += loss.item()

            # Update weights every `accum` steps (or at end of epoch)
            if step % accum == 0 or step == len(train_dl):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch} step {step}/{len(train_dl)} | "
                      f"loss={loss.item():.6f} lr={lr:.2e}")

        avg_loss = epoch_loss / max(1, len(train_dl))
        print(f"Epoch {epoch} train | loss={avg_loss:.6f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks, targets in val_dl:
                inputs = inputs.to(device)
                masks = masks.to(device)
                targets = targets.to(device)
                loss = training_step(model, inputs, masks, targets,
                                     horizon=args.horizon)
                val_loss += loss.item()

        avg_val = val_loss / max(1, len(val_dl))
        print(f"Epoch {epoch} val   | loss={avg_val:.6f}")

        # ---- Save checkpoint ----
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_val_epoch = epoch

        out_dir = Path(args.save_dir) / f"epoch_{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            model_wrapper._save_pretrained(str(out_dir))
            _copy_config_json(args.model_id, out_dir)
            tag = " (best)" if is_best else ""
            print(f"Saved checkpoint to {out_dir}{tag}")
        except Exception:
            torch.save(model.state_dict(), out_dir / "model.pt")
            print(f"Saved PyTorch checkpoint to {out_dir / 'model.pt'} (fallback)")

        if is_best:
            best_dir = Path(args.save_dir) / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            try:
                model_wrapper._save_pretrained(str(best_dir))
                _copy_config_json(args.model_id, best_dir)
            except Exception:
                torch.save(model.state_dict(), best_dir / "model.pt")
            print(f"Updated best checkpoint at {best_dir}")

        # Early stopping
        if args.early_stopping > 0 and (epoch - best_val_epoch) >= args.early_stopping:
            print(f"\nEarly stopping: no val improvement for {args.early_stopping} epochs "
                  f"(best was epoch {best_val_epoch})")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f} (epoch {best_val_epoch})")


if __name__ == "__main__":
    main()
