# Glia Run Log - Japan (Expanded Tickers v2)

Started: 2026-03-25
Output dir: finetune_japan/finetune_expts/glia_run_20260325

---

## Experiment: Japan v2 — 18 tickers (expanded from 10), stride=32

### Hypothesis

Expanding from 10 to 18 training tickers with stride=32 will break through Japan's
val_loss floor of 0.027 (which prevented all prior Japan finetuning attempts).

**New tickers added:** Takeda (4502.T), Daikin (6367.T), SMFG (8316.T), Renesas (6723.T),
Honda (7267.T), KDDI (9433.T), Shin-Etsu Chemical (4063.T), Mitsui Fudosan (8801.T)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Tickers | 18 (all loaded successfully) |
| Sliding windows | 2411 |
| Train/Val split | 1809 / 602 |
| Steps per epoch | 227 |
| Best val_loss | **0.012469** (epoch 47) |

All other hyperparameters identical to Run4 recipe.

### Val Loss Trajectory (breakthrough)

| Epoch | Val Loss | Notes |
|-------|----------|-------|
| 1 | 0.026747 | Similar to prior 10-ticker runs |
| 9 | 0.018302 | Already below prior floor of 0.027 |
| 15 | 0.015488 | Approaching 0.015 threshold |
| 17 | 0.014669 | **Crossed 0.015 threshold** |
| 20 | 0.013522 | |
| 29 | 0.012779 | |
| 32 | 0.012711 | |
| 44 | 0.012728 | |
| **47** | **0.012469** | **Best** |
| 50 | 0.013249 | Training complete (no early stopping triggered) |

**Prior Japan runs (10 tickers):** val_loss floor ~0.027, never crossed 0.015.
**This run (18 tickers):** val_loss 0.012469 — a 54% reduction. The additional 8 tickers
provided sufficient training diversity for convergence.

### Rolling Evaluation (15 windows, 120d horizon)

| Model | Mean MAPE | Std | vs Baseline |
|-------|-----------|-----|-------------|
| Baseline | 18.10% | ±1.15% | — |
| Japan_v2_18tk | 19.59% | ±0.83% | **-1.49pp (WORSE)** |

**Per-Ticker:**

| Model | 8035.T (Tokyo Electron) | 6902.T (Denso) | 4661.T (Oriental Land) |
|-------|------------------------|----------------|----------------------|
| Baseline | 33.2% | 3.1% | 18.0% |
| Japan_v2_18tk | 38.1% | 4.3% | 16.4% |

### Analysis

1. **Val loss breakthrough is real but misleading.** The 18-ticker training set provides
   enough diversity for the model to fit Japanese stock patterns well in training. But
   the eval tickers (semiconductor, auto parts, theme park) are sufficiently different
   from the training set that the learned patterns don't transfer.

2. **8035.T (Tokyo Electron) dominates the error.** Semiconductor stocks have extreme
   volatility (baseline MAPE 33.2%). Finetuning makes this worse (+4.9pp) — the model
   learns patterns from the training tickers (banks, telecom, chemicals) that actively
   misguide semiconductor forecasting.

3. **Only 4661.T improved** (18.0% → 16.4%, -1.6pp). Oriental Land (Disney Japan) is the
   most "consumer staple"-like ticker, closest to the training set's retail/entertainment mix.

4. **Japan vs China comparison:**
   - China: 10 tickers work, 17 tickers dilute → prior Glia2a remains best
   - Japan: 10 tickers can't converge (val_loss 0.027), 18 tickers converge (0.012) but don't help eval
   - Root cause differs: China's eval tickers match training sector. Japan's eval tickers
     (semiconductor, auto parts) are structurally different from any training ticker.

### Conclusion

**Japan finetuning remains non-viable.** Despite breaking the val_loss floor, the checkpoint
regresses on eval tickers. The baseline is recommended for Japan.

The fundamental issue is eval ticker composition: 8035.T (semiconductor, 33% MAPE) is
inherently difficult and no amount of Japanese equity training data helps — the training
tickers don't contain semiconductor-like dynamics.

---
