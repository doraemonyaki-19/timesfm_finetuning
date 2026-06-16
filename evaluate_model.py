
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yfinance as yf
from safetensors.torch import load_file
import pandas as pd
import sys
import time

# Add this to fix stdout/stderr encoding issues
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from finetune_timesfm import TimesFM_2p5_200M_torch, get_predictions

def download_data(tickers, start_date="2010-01-01"):
    """Downloads historical stock data for the given tickers."""
    print(f"[{time.time()}] Downloading data for: {tickers}")
    # When downloading multiple tickers, yfinance returns a DataFrame with multi-level columns
    # (e.g., ('Adj Close', 'AAPL'), ('Volume', 'AAPL')).
    data = yf.download(tickers, start=start_date, progress=False)
    print(f"[{time.time()}] Data download complete.")
    # We only need the 'Adj Close' price.
    return data['Adj Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Adj Close']]


def directional_accuracy(y_true, y_pred):
    """Calculates directional accuracy."""
    return np.mean(np.sign(y_pred[1:] - y_pred[:-1]) == np.sign(y_true[1:] - y_true[:-1])) * 100

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(model, series, max_context, horizons, n_windows, device):
    """Performs a rolling-window evaluation of the model on the given series."""
    results = {h: {"mape": [], "dir_acc": []} for h in horizons}

    for i in range(n_windows):
        end_idx = len(series) - (i * (max(horizons) // n_windows))
        start_idx = max(0, end_idx - max_context)
        context = series.iloc[start_idx:end_idx].values.astype(np.float32)

        if len(context) < 2:
            continue

        inputs = torch.from_numpy(context).unsqueeze(0).to(device)
        masks = torch.ones_like(inputs).to(device) 
        
        preds_all_horizons = get_predictions(model, inputs, masks, max(horizons))
        
        for h in horizons:
            if end_idx + h > len(series):
                continue

            preds = preds_all_horizons[0, :h].cpu().numpy()
            target = series.iloc[end_idx : end_idx + h].values

            if len(preds) != len(target):
                continue
                
            results[h]["mape"].append(mean_absolute_percentage_error(target, preds))
            if len(target) > 1:
                results[h]["dir_acc"].append(directional_accuracy(target, preds))

    return {h: {"mape": np.mean(v["mape"]), "dir_acc": np.mean(v["dir_acc"]) if v["dir_acc"] else 0.0} for h, v in results.items()}

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned TimesFM model with rolling windows.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the fine-tuned model checkpoint directory (e.g., .../exp_11/best).")
    parser.add_argument("--tickers", type=str, required=True, help="Comma-separated list of tickers to evaluate on.")
    parser.add_argument("--horizons", type=str, required=True, help="Comma-separated list of forecast horizons (e.g., '5,14,30,60,90,120').")
    parser.add_argument("--max-context", type=int, default=512, help="Maximum context length for the model.")
    parser.add_argument("--n-windows", type=int, default=15, help="Number of rolling windows for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_path = Path(args.checkpoint) / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {args.checkpoint}")
        
    config_path = "C:/Users/ylchen/workspace/timesfm/model"
    wrapper = TimesFM_2p5_200M_torch(config_path)
    model = wrapper.model
    model.load_state_dict(load_file(model_path, device=str(device)))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")

    # Parse and clean tickers
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]

    # Download data
    data = download_data(tickers)
    
    all_results = {}
    for ticker in tickers:
        print(f"Evaluating {ticker}...")
        if ticker not in data.columns:
            print(f"Skipping {ticker} as data could not be downloaded.")
            continue
            
        series = data[ticker].dropna()
        if len(series) < args.max_context + max(horizons):
            print(f"Skipping {ticker} due to insufficient data.")
            continue
        
        ticker_results = evaluate_model(model, series, args.max_context, horizons, args.n_windows, device)
        all_results[ticker] = ticker_results
        
        print(f"Results for {ticker}:")
        for h in horizons:
            print(f"  Horizon {h}d: MAPE={ticker_results[h]['mape']:.2f}%, DirAcc={ticker_results[h]['dir_acc']:.2f}%")

    # Calculate and print average results
    avg_results = {h: {"mape": [], "dir_acc": []} for h in horizons}
    for ticker_res in all_results.values():
        for h in horizons:
            avg_results[h]['mape'].append(ticker_res[h]['mape'])
            avg_results[h]['dir_acc'].append(ticker_res[h]['dir_acc'])

    print("\nAverage Results Across All Tickers:")
    for h in horizons:
        if avg_results[h]['mape']:
            avg_mape = np.mean(avg_results[h]['mape'])
            avg_dir_acc = np.mean(avg_results[h]['dir_acc'])
            print(f"  Horizon {h}d: MAPE={avg_mape:.2f}%, DirAcc={avg_dir_acc:.2f}%")

    # Save the result
    result_path = Path(args.checkpoint) / "evaluation_results_rolling.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEvaluation results saved to {result_path}")


if __name__ == "__main__":
    main()
