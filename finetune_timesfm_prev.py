"""
Finetune TimesFM 2.5 (PyTorch) on financial time series.
"""
from __future__ import annotations
import argparse
import math
import os
import random
import shutil
import sys
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

# Handle torch import to avoid local `timesfm/torch` shadowing.
script_dir = str(Path(__file__).resolve().parent)
_removed_sys0 = None
if sys.path and sys.path[0] == script_dir:
    _removed_sys0 = sys.path.pop(0)
if _removed_sys0 is not None:
    sys.path.insert(0, _removed_sys0)

try:
    import yfinance as yf
except ImportError:
    yf = None

# Ensure repository `src` is importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "timesfm" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.torch.util import revin, update_running_stats

_EPS = 1e-9

class FinancialTimeSeriesDataset(Dataset):
    def __init__(
        self,
        series_list: list[np.ndarray],
        windows_list: list[tuple[int, int]],
        max_context: int,
        min_context: int,
        horizon: int,
    ):
        self.series_list = series_list
        self.windows = windows_list
        self.max_context = max_context
        self.min_context = min_context
        self.horizon = horizon

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        s_idx, start = self.windows[idx]
        series = self.series_list[s_idx]
        total_needed = self.max_context + self.horizon

        window = np.log(np.maximum(series[start : start + total_needed], _EPS)).astype(np.float32)

        context = window[:self.max_context]
        mask = np.zeros(self.max_context, dtype=bool)  # full context, no masking
        target = window[self.max_context:]
        return (context, mask, target)

def create_windows(
    series_list: list[np.ndarray], max_context: int, horizon: int, stride: int
) -> list[tuple[int, int]]:
    windows = []
    total_needed = max_context + horizon
    for s_idx, series in enumerate(series_list):
        if len(series) >= total_needed:
            for start in range(0, len(series) - total_needed + 1, stride):
                windows.append((s_idx, start))
    return windows

def collate_fn(batch: List):
    contexts, masks, targets = zip(*batch)
    return (
        torch.from_numpy(np.stack(contexts)).float(),
        torch.from_numpy(np.stack(masks)).bool(),
        torch.from_numpy(np.stack(targets)).float(),
    )

def get_predictions(model_module, inputs, masks, horizon: int) -> torch.Tensor:
    """Helper function to get model predictions for a given batch."""
    device = inputs.device
    B = inputs.shape[0]
    p, o, q = model_module.p, model_module.o, model_module.q
    
    patched_inputs = inputs.reshape(B, -1, p)
    patched_masks = masks.reshape(B, -1, p)
    num_input_patches = patched_inputs.shape[1]
    
    n = torch.zeros(B, device=device)
    mu = torch.zeros(B, device=device)
    sigma = torch.zeros(B, device=device)
    
    patch_mu, patch_sigma = [], []
    for i in range(num_input_patches):
        (n, mu, sigma), _ = update_running_stats(n, mu, sigma, patched_inputs[:, i], patched_masks[:, i])
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)
    
    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
    
    (_, _, normed_outputs, _), _ = model_module(normed_inputs, patched_masks)
    
    if torch.any(torch.isnan(normed_outputs)):
        normed_outputs = torch.nan_to_num(normed_outputs, nan=0.0)
        
    renormed_outputs = revin(normed_outputs, context_mu, context_sigma, reverse=True)
    renormed_outputs = renormed_outputs.reshape(B, num_input_patches, o, q)
    
    return renormed_outputs[:, -1, :horizon, model_module.aridx]

def training_loss_fn(pred, targets, horizon_decay: float = 0.0) -> torch.Tensor:
    """Calculates training loss. Can apply exponential decay to loss across horizon."""
    if horizon_decay > 0.0:
        horizon = targets.shape[1]
        # Create position-based weights
        positions = torch.arange(horizon, device=targets.device, dtype=torch.float32)
        # Exponential decay: weight decreases for points further in the future
        weights = torch.exp(-horizon_decay * positions / horizon)
        # Normalize weights to have the same sum as an unweighted loss
        weights = weights / weights.sum() * horizon
        # Apply weights to the squared error
        return ((pred - targets) ** 2 * weights.unsqueeze(0)).mean()
    else:
        # Standard MSE loss
        return torch.nn.functional.mse_loss(pred, targets)

def validation_metrics_fn(pred, targets) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates validation metrics (MSE and MAE)."""
    mse = torch.nn.functional.mse_loss(pred, targets)
    mae = torch.nn.functional.l1_loss(pred, targets)
    return mse, mae

def _ticker_cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe}.npy"


def download_tickers(
    tickers: list[str],
    years: int = 20,
    cache_dir: str | Path | None = None,
    end_date: str | None = None,
) -> list[np.ndarray]:
    """Return close-price arrays for each ticker.

    When cache_dir is provided and end_date is None:
    - Tickers already cached are loaded from disk (no network call).
    - Newly downloaded tickers are saved to the cache for next time.
    When end_date is set, always downloads fresh (bypasses cache) so the
    cutoff is applied precisely. Data is not written back to cache.
    """
    import datetime
    cd = Path(cache_dir) if cache_dir else None
    if cd:
        cd.mkdir(parents=True, exist_ok=True)

    series_list = []
    for ticker in tickers:
        ticker = ticker.strip()
        # Use cache only when no date cutoff is requested
        if cd and end_date is None:
            cached = _ticker_cache_path(cd, ticker)
            if cached.exists():
                data = np.load(cached).astype(np.float32)
                data = data[~np.isnan(data) & (data > 0)]
                print(f"[cache] {ticker}: {len(data)} pts from {cached.name}")
                series_list.append(data)
                continue

        if yf is None:
            raise RuntimeError(
                f"yfinance is required to download {ticker}. "
                "Install with: pip install yfinance, or pre-cache with "
                "scripts/cache_tickers.py and pass --cache-dir."
            )
        if end_date is not None:
            end = datetime.date.fromisoformat(end_date)
        else:
            end = datetime.datetime.now(datetime.timezone.utc).date()
        start = end - datetime.timedelta(days=365 * years)
        print(f"Downloading {ticker} from {start} to {end}...")
        df = yf.download(
            ticker, start=start.isoformat(), end=end.isoformat(),
            progress=False, auto_adjust=True,
        )
        if df is None or df.empty:
            print(f"  WARNING: Failed to download {ticker}, skipping.")
            continue
        close = df["Close"].values.astype(np.float32).flatten()
        close = close[~np.isnan(close) & (close > 0)]
        print(f"  Got {len(close)} data points for {ticker}")
        # Only cache when no cutoff (cache holds full-history data)
        if cd and end_date is None:
            np.save(_ticker_cache_path(cd, ticker), close)
            print(f"  Cached to {_ticker_cache_path(cd, ticker).name}")
        series_list.append(close)

    if not series_list:
        raise RuntimeError("No data loaded for any ticker.")
    return series_list

def load_from_data_dir(data_dir: str) -> list[np.ndarray]:
    """Loads time series data from .npy files in a directory."""
    p = Path(data_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    series_list = []
    for f in sorted(p.glob("*.npy")):
        print(f"Loading data from {f}...")
        data = np.load(f).astype(np.float32)
        data = data[~np.isnan(data) & (data > 0)]
        series_list.append(data)
        print(f"  Got {len(data)} data points.")
    if not series_list:
        raise RuntimeError(f"No .npy files found in {data_dir}")
    return series_list


def freeze_layers(model_module, num_layers: int):
    for i in range(min(num_layers, len(model_module.stacked_xf))):
        for param in model_module.stacked_xf[i].parameters():
            param.requires_grad = False

def count_params(model_module) -> tuple[int, int]:
    total = sum(p.numel() for p in model_module.parameters())
    trainable = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    return total, trainable

def main():
    parser = argparse.ArgumentParser(description="Finetune TimesFM 2.5 on financial time series")
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--model-id", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--freeze-layers", type=int, default=17)
    parser.add_argument("--early-stopping", type=int, default=15)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--min-context", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--save-dir", type=str, default="finetune_checkpoints")
    parser.add_argument("--save-all-epochs", action="store_true", help="Save a checkpoint for every epoch.")
    parser.add_argument("--val-metric", type=str, default="mse", choices=["mse", "mae"])
    parser.add_argument("--no-epoch-ckpts", action="store_true", help="Do not save epoch checkpoints.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--horizon-loss-decay", type=float, default=0.0, help="Decay factor for horizon-weighted loss. 0.0 = standard MSE.")
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for cached .npy ticker files. When set, tickers are loaded "
             "from cache instead of downloading via yfinance; missing tickers are "
             "downloaded once and saved. Pre-populate with scripts/cache_tickers.py.",
    )
    parser.add_argument(
        "--train-end-date", type=str, default=None,
        help="ISO date (YYYY-MM-DD). When set, training data is downloaded up to "
             "this date (inclusive) instead of today. Cache is bypassed so the "
             "cutoff is applied precisely.",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_dir = save_dir / "best"
    
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.tickers:
        series_list = download_tickers(
            args.tickers.split(","),
            cache_dir=args.cache_dir,
            end_date=args.train_end_date,
        )
    elif args.data_dir:
        series_list = load_from_data_dir(args.data_dir)
    else:
        raise ValueError("Either --tickers or --data-dir must be provided.")

    windows = create_windows(series_list, args.max_context, args.horizon, args.stride)
    random.shuffle(windows)
    val_size = int(len(windows) * args.val_ratio)
    train_windows, val_windows = windows[val_size:], windows[:val_size]

    train_dataset = FinancialTimeSeriesDataset(series_list, train_windows, args.max_context, args.min_context, args.horizon)
    val_dataset = FinancialTimeSeriesDataset(series_list, val_windows, args.max_context, args.min_context, args.horizon)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2)

    wrapper = TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
    model = wrapper.model.to(device)

    if args.freeze_layers > 0:
        freeze_layers(model, args.freeze_layers)
    total_params, trainable_params = count_params(model)
    print(f"Model loaded. Total params: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = len(train_loader) * args.warmup_epochs // args.accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    scaler = torch.cuda.amp.GradScaler()
    best_val_metric = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for i, (inputs, masks, targets) in enumerate(train_loader):
            step = epoch * len(train_loader) + i
            if step < warmup_steps:
                lr = args.lr * (step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            elif step == warmup_steps:
                print("Warmup complete, starting cosine decay.")

            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            
            with torch.cuda.amp.autocast():
                preds = get_predictions(model, inputs, masks, args.horizon)
                loss = training_loss_fn(preds, targets, args.horizon_loss_decay)
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.accumulation_steps

            if (i + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if step >= warmup_steps:
                    scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        total_val_mse, total_val_mae = 0, 0
        with torch.no_grad():
            for inputs, masks, targets in val_loader:
                inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
                with torch.cuda.amp.autocast():
                    preds = get_predictions(model, inputs, masks, args.horizon)
                    mse, mae = validation_metrics_fn(preds, targets.to(device))
                total_val_mse += mse.item()
                total_val_mae += mae.item()

        avg_val_mse = total_val_mse / len(val_loader)
        avg_val_mae = total_val_mae / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val MSE: {avg_val_mse:.6f} | Val MAE: {avg_val_mae:.6f} | LR: {current_lr:.2e}")
        
        history.append({
            "epoch": epoch + 1, "train_loss": avg_train_loss, "val_mse": avg_val_mse, "val_mae": avg_val_mae, "lr": current_lr
        })
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        current_val_metric = avg_val_mae if args.val_metric == 'mae' else avg_val_mse
        
        if args.save_all_epochs:
            epoch_dir = save_dir / f"epoch_{epoch+1:02d}"
            epoch_dir.mkdir(exist_ok=True)
            checkpoint_path = epoch_dir / "model.safetensors"
            save_file(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint for epoch {epoch+1}")
        
        if current_val_metric < best_val_metric:
            best_val_metric = current_val_metric
            epochs_no_improve = 0
            best_dir.mkdir(exist_ok=True)
            checkpoint_path = best_dir / "model.safetensors"
            save_file(model.state_dict(), checkpoint_path)
            print(f"  New best val_{args.val_metric}: {best_val_metric:.6f}. Checkpoint saved to {best_dir}")
        else:
            epochs_no_improve += 1
            
        if args.early_stopping > 0 and epochs_no_improve >= args.early_stopping:
            print(f"Early stopping triggered after {args.early_stopping} epochs with no improvement.")
            break
    
    # Save the final validation metric to a file for easy access
    with open(save_dir / "final_val_metric.json", "w") as f:
        json.dump({"best_val_metric": best_val_metric, "metric": args.val_metric}, f)

    if best_val_metric > 0.015:
        print(f"WARNING: Best val_loss {best_val_metric:.6f} > 0.015. Model may produce NaNs at inference.")

    # Remove any per-epoch checkpoint directories, keeping only best/
    import shutil
    for epoch_dir in sorted(save_dir.glob("epoch_*")):
        if epoch_dir.is_dir():
            shutil.rmtree(epoch_dir)
            print(f"  Removed epoch checkpoint: {epoch_dir.name}")

    print(f"Training complete. Best model saved to {best_dir} with val_{args.val_metric}: {best_val_metric:.6f}")

if __name__ == "__main__":
    main()
