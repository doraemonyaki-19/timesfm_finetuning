# TimesFM Finetuning Experiments

## Goal
Finetune TimesFM 2.5 (200M params, PyTorch) on financial time series data and evaluate whether finetuning improves forecast accuracy on held-out tickers.

## Summary

Seven finetuning runs were conducted. SGD (Runs 1-2) failed to move the 231M-param model's weights meaningfully with batch_size=8. Adam (Run 3) succeeded in producing different predictions but showed mixed results due to overfitting. **Run 4 (AdamW + layer freezing + gradient accumulation) achieved the best results: overall MAPE improved from 8.1% to 7.0% (-1.0pp), with 5/6 held-out tickers improving.** The key was reducing overfitting through freezing 17/20 transformer layers, weight decay regularization, and 16x gradient accumulation for an effective batch size of 128. Run 5 attempted to scale up with 100 tickers and fewer frozen layers but regressed. Run 6 isolated "more data" as the only variable (100 tickers with Run 4's exact hyperparameters) — it also regressed (+0.5pp at 120d), confirming that 10 focused tickers outperform 100 diverse tickers for this eval set. Run 6b continued from Run 6's best checkpoint but early-stopped at epoch 22 with similar results.

### Multi-Horizon Summary (Run 4 — Best Model)

| Horizon | Baseline MAPE | Finetuned MAPE | Delta |
|---------|--------------|----------------|-------|
| 7 days | 2.5% | 2.0% | **-0.5pp** |
| 30 days | 4.7% | 4.5% | **-0.2pp** |
| 60 days | 5.7% | 5.5% | **-0.2pp** |
| 120 days | 8.1% | 7.0% | **-1.0pp** |

Run 4 improves MAPE at every forecast horizon, with the largest improvement at 120 days.

## Method
Based on Fu, Hirano & Imajo (2024) "Financial Fine-tuning a Large Time Series Model" (arXiv:2412.09880):
- Log-transform inputs: z = log(y + eps)
- Random context masking to prevent overfitting
- Linear warmup + cosine decay LR schedule
- MSE loss in log-space
- Optimizers tested: SGD + momentum (paper), Adam, AdamW (best for small batches)
- Layer freezing: freeze early transformer layers to reduce trainable params and prevent overfitting
- Gradient accumulation: simulate larger effective batch sizes

## Training Tickers (10, diverse sectors)
| Ticker | Sector |
|--------|--------|
| ^GSPC | S&P 500 Index (broad market) |
| AAPL | Technology |
| JNJ | Healthcare |
| JPM | Financials |
| XOM | Energy |
| PG | Consumer Staples |
| CAT | Industrials |
| NEE | Utilities |
| AMT | Real Estate |
| WMT | Consumer Discretionary/Retail |

## Evaluation Tickers (6, different from training)
| Ticker | Sector |
|--------|--------|
| NVDA | Technology (Semiconductors) |
| UNH | Healthcare (Insurance) |
| GS | Financials (Investment Banking) |
| CVX | Energy |
| KO | Consumer Staples |
| BA | Industrials (Aerospace) |

## Evaluation Metrics
- **MSE** — Mean Squared Error (raw price space)
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error
- **Directional Accuracy** — % of steps where predicted direction matches actual

## Results

### Run 1: SGD lr=5e-4, 10 epochs (2026-02-19)

**Outcome: No measurable improvement.** Weights moved by < 1e-6 from pretrained baseline.

**Training:** 350 sliding windows (263 train / 87 val). Loss went from 0.018 to 0.013 (best val at epoch 7), but this was too small to meaningfully shift the 231M parameter model.

**Root cause:** The paper used batch_size=1024 on 8xV100. Our batch_size=8 gives ~128x weaker gradient signal per step. Combined with only 330 total gradient steps (33 steps/epoch * 10 epochs) and a conservative lr=5e-4, the model barely moved.

**Baseline evaluation on held-out tickers (pretrained model):**

| Ticker | MSE | MAE | RMSE | MAPE | DirAcc |
|--------|-----|-----|------|------|--------|
| NVDA | 133.17 | 9.63 | 11.54 | 5.1% | 52.3% |
| UNH | 1174.96 | 25.10 | 34.28 | 7.9% | 52.3% |
| GS | 15607.13 | 100.66 | 124.93 | 11.4% | 45.3% |
| CVX | 258.06 | 13.33 | 16.06 | 8.2% | 56.2% |
| KO | 23.82 | 3.62 | 4.88 | 5.0% | 49.2% |
| BA | 543.01 | 19.31 | 23.30 | 9.3% | 45.3% |
| **Overall** | **2956.69** | **28.61** | **54.38** | **7.8%** | **50.1%** |

### Run 2: SGD lr=5e-3, 50 epochs (2026-02-19)

**Outcome: No measurable improvement.** Weights moved by ~2e-6 average — still too small for inference.

10x higher LR and 5x more epochs to compensate for small batch size. Train loss went from 0.0168 to 0.0083, best val 0.0138 at epoch 4. However, SGD with small batch_size=8 still could not generate large enough per-parameter updates for this 231M model.

### Run 3: Adam lr=1e-4, 50 epochs (2026-02-20)

**Outcome: Mixed results.** Weights changed meaningfully (mean relative diff ~20-33%). Finetuned model improved on some tickers but degraded on others.

**Training:** Adam optimizer replaced SGD for better per-parameter updates with small batches. Train loss dropped dramatically from 0.0139 to 0.0003 (significant overfitting). Best val loss 0.0161 at epoch 40.

**Bug fix:** Discovered that `TimesFM_2p5_200M_torch` uses a class-level `model` attribute shared across all instances. The evaluation script was loading both models simultaneously, causing the second to overwrite the first. Fixed by loading + evaluating sequentially.

**Evaluation on held-out tickers (Baseline vs Finetuned):**

| Ticker | Model | MSE | MAE | RMSE | MAPE | DirAcc |
|--------|-------|-----|-----|------|------|--------|
| NVDA | Baseline | 137.02 | 9.79 | 11.71 | 5.2% | 43.8% |
| NVDA | **Finetuned** | **111.41** | **8.36** | **10.56** | **4.5%** | **53.1%** |
| UNH | **Baseline** | **1622.85** | **31.86** | **40.28** | **10.2%** | 48.4% |
| UNH | Finetuned | 2704.91 | 45.72 | 52.01 | 13.6% | 46.1% |
| GS | **Baseline** | **15090.51** | **99.71** | **122.84** | **11.3%** | 54.7% |
| GS | Finetuned | 20135.75 | 121.92 | 141.90 | 14.0% | 55.5% |
| CVX | Baseline | 260.70 | 13.11 | 16.15 | 8.0% | **57.0%** |
| CVX | **Finetuned** | **219.00** | **8.57** | **14.80** | **5.0%** | 49.2% |
| KO | Baseline | 29.54 | 3.99 | 5.44 | 5.5% | 49.2% |
| KO | **Finetuned** | **21.54** | **3.16** | **4.64** | **4.3%** | **50.0%** |
| BA | **Baseline** | **417.49** | **17.13** | **20.43** | **8.2%** | 49.2% |
| BA | Finetuned | 1023.04 | 25.65 | 31.99 | 12.5% | 50.8% |
| **Overall** | **Baseline** | **2926.35** | **29.26** | **54.10** | **8.1%** | 50.4% |
| **Overall** | Finetuned | 4035.94 | 35.56 | 63.53 | 9.0% | 50.8% |

**Per-ticker analysis:**
- **NVDA** — Finetuned wins: MAPE -0.7pp, DirAcc +9.4pp
- **CVX** — Finetuned wins: MAPE -3.0pp (but DirAcc -7.8pp)
- **KO** — Finetuned wins: MAPE -1.2pp, DirAcc +0.8pp
- **UNH** — Baseline wins: MAPE +3.5pp
- **GS** — Baseline wins: MAPE +2.7pp
- **BA** — Baseline wins: MAPE +4.3pp

**Conclusion:** Finetuning with Adam improved MAPE on 3/6 held-out tickers (NVDA, CVX, KO) but degraded on the other 3 (UNH, GS, BA). Overall MAPE slightly worse (+0.9pp). The model shows signs of overfitting to training distribution — it improved on stocks with moderate price levels similar to training data but degraded on high-price/volatile stocks (UNH, GS, BA). More data, regularization, or selective layer freezing would likely help.

**Plot:** See [evaluation_results.png](evaluation_results.png) for predicted vs actual curves.

### Run 4: AdamW + Layer Freezing + Gradient Accumulation (2026-02-20)

**Outcome: Best results yet.** Overall MAPE improved from 8.1% to 7.0% (-1.0pp). 5/6 held-out tickers improved.

**Key changes from Run 3:**
- **Layer freezing:** Froze first 17/20 transformer layers → 64M trainable params (27.7%) vs 231M (100%)
- **AdamW:** Weight decay = 0.01 for regularization
- **Gradient accumulation:** 16 steps → effective batch size 128 (vs 8)

**Training:** Train loss decreased gradually from 0.018 to 0.003 (vs Run 3's dramatic 0.014→0.0003 overfitting). Best val loss **0.0128** at epoch 4 (vs Run 3's 0.0161). Much healthier training dynamics with significantly less overfitting.

**Evaluation on held-out tickers (Baseline vs Finetuned):**

| Ticker | Model | MSE | MAE | RMSE | MAPE | DirAcc |
|--------|-------|-----|-----|------|------|--------|
| NVDA | Baseline | 137.02 | 9.79 | 11.71 | 5.2% | 43.8% |
| NVDA | **Finetuned** | **71.87** | **6.81** | **8.48** | **3.6%** | **46.9%** |
| UNH | Baseline | 1622.85 | 31.86 | 40.28 | 10.2% | 48.4% |
| UNH | **Finetuned** | **953.75** | **23.70** | **30.88** | **7.5%** | 46.1% |
| GS | Baseline | 15090.45 | 99.71 | 122.84 | 11.3% | 54.7% |
| GS | **Finetuned** | **11948.89** | **88.52** | **109.31** | **10.0%** | 54.7% |
| CVX | Baseline | 260.70 | 13.11 | 16.15 | 8.0% | 57.0% |
| CVX | **Finetuned** | **209.98** | **11.14** | **14.49** | **6.7%** | **60.2%** |
| KO | Baseline | 29.54 | 3.99 | 5.44 | 5.5% | 49.2% |
| KO | **Finetuned** | **27.21** | **3.81** | **5.22** | **5.2%** | **50.0%** |
| BA | **Baseline** | **417.49** | **17.13** | **20.43** | **8.2%** | 49.2% |
| BA | Finetuned | 541.33 | 18.55 | 23.27 | 9.0% | **50.8%** |
| **Overall** | **Baseline** | 2926.34 | 29.26 | 54.10 | 8.1% | 50.4% |
| **Overall** | **Finetuned** | **2292.17** | **25.42** | **47.88** | **7.0%** | **51.4%** |

**Per-ticker analysis:**
- **NVDA** — Finetuned wins: MAPE -1.6pp, DirAcc +3.1pp, MSE nearly halved
- **UNH** — Finetuned wins: MAPE -2.6pp (was +3.5pp in Run 3 — biggest turnaround)
- **GS** — Finetuned wins: MAPE -1.3pp (was +2.7pp in Run 3)
- **CVX** — Finetuned wins: MAPE -1.3pp, DirAcc +3.1pp
- **KO** — Finetuned wins: MAPE -0.3pp, DirAcc +0.8pp
- **BA** — Baseline wins: MAPE +0.8pp (much smaller degradation than Run 3's +4.3pp)

**Conclusion:** Layer freezing + weight decay + gradient accumulation dramatically reduced overfitting. The finetuned model now improves on 5/6 held-out tickers, and the single degraded ticker (BA) shows much less damage than before. Overall MAPE improved by 1.0pp and all MSE/MAE/RMSE metrics improved. This validates that the Run 3 degradation was primarily due to overfitting, not a fundamental issue with finetuning.

**Multi-horizon evaluation (Run 4):**

| Horizon | Baseline MAPE | Finetuned MAPE | Delta | DirAcc Delta |
|---------|--------------|----------------|-------|-------------|
| 7 days | 2.5% | 2.0% | **-0.5pp** | -7.1pp |
| 30 days | 4.7% | 4.5% | **-0.2pp** | -2.2pp |
| 60 days | 5.7% | 5.5% | **-0.2pp** | +2.5pp |
| 120 days | 8.1% | 7.0% | **-1.0pp** | +1.9pp |

Run 4 improves MAPE at every horizon. The improvement is largest at 120 days, suggesting finetuning helps most with longer-range forecasts where the pretrained model has more room for error.

**Plots:** See [evaluation_results_run4_h7.png](evaluation_results_run4_h7.png), [evaluation_results_run4_h30.png](evaluation_results_run4_h30.png), [evaluation_results_run4_h60.png](evaluation_results_run4_h60.png), [evaluation_results_run4_h120.png](evaluation_results_run4_h120.png).

### Run 5: 100 Tickers + AdamW + Freeze 14 + 64x Accumulation + Early Stopping (2026-02-21)

**Outcome: Regression from Run 4.** Overall MAPE worse than baseline across all horizons. Continued finetuning from Run 4 checkpoint with more data and less freezing led to overfitting.

**Key changes from Run 4:**
- **100 training tickers** (vs 10) across all sectors + international indices
- **Freeze 14 layers** (vs 17) → 93.6M trainable params (40.5%) vs 64M (27.7%)
- **64x gradient accumulation** → effective batch size 512 (vs 128)
- **Lower LR:** 5e-5 (vs 1e-4) with 5-epoch warmup
- **Early stopping:** patience=15 epochs
- **Continued from Run 4 checkpoint** (not pretrained baseline)

**Training:** 100 tickers downloaded, 3445 sliding windows (10x more than Run 4). Train loss: 0.018 → 0.006. Best val loss **0.016711** at epoch 20. Early stopping would have triggered at epoch 35 (training crashed on checkpoint save due to disk full at epoch 35, but best checkpoint was already saved at epoch 20).

**Multi-horizon evaluation (Run 5):**

| Horizon | Baseline MAPE | Finetuned MAPE | Delta | DirAcc Delta |
|---------|--------------|----------------|-------|-------------|
| 7 days | 2.5% | 2.7% | +0.2pp | -7.1pp |
| 30 days | 4.7% | 5.5% | +0.7pp | +0.6pp |
| 60 days | 5.7% | 6.7% | +0.9pp | +4.4pp |
| 120 days | 8.1% | 8.4% | +0.4pp | +3.6pp |

**Per-ticker analysis (120-day horizon):**
- **NVDA** — Finetuned wins: MAPE -2.6pp, DirAcc +12.5pp (strong improvement)
- **UNH** — Finetuned wins: MAPE -1.4pp
- **CVX** — Finetuned wins: MAPE -2.9pp, DirAcc +2.5pp
- **GS** — Baseline wins: MAPE +1.3pp
- **KO** — Baseline wins: MAPE +0.8pp
- **BA** — Baseline wins: MAPE +6.9pp (severe degradation)

**Conclusion:** Continuing finetuning from Run 4's checkpoint with fewer frozen layers (14 vs 17) accumulated too much domain-specific adaptation, causing the model to lose generalization. BA was particularly degraded (MAPE 16.3% vs 9.4% baseline). The directional accuracy did improve at longer horizons, suggesting the model learned some useful patterns but over-corrected on magnitude. **Run 4 remains the best model.**

**Plots:** See [evaluation_results_run5_h7.png](evaluation_results_run5_h7.png), [evaluation_results_run5_h30.png](evaluation_results_run5_h30.png), [evaluation_results_run5_h60.png](evaluation_results_run5_h60.png), [evaluation_results_run5_h120.png](evaluation_results_run5_h120.png).

### Run 6: 100 Tickers + Run 4 Hyperparameters (2026-02-22)

**Outcome: Regression from Run 4.** Isolating "more data" as the only variable — 100 tickers with identical hyperparameters to Run 4 — still regressed. More data did not help.

**Key changes from Run 4:**
- **100 training tickers** (vs 10) across all sectors + international indices
- **All other hyperparameters identical** to Run 4: freeze 17 layers, lr=1e-4, AdamW, weight_decay=0.01, 16x accumulation, 25-epoch warmup
- **Started from pretrained** (not from Run 4 checkpoint)
- **Early stopping:** patience=15 epochs

**Training:** 3445 sliding windows (2584 train / 861 val). Best val loss **0.015296** at epoch 40. Training completed all 50 epochs.

**Multi-horizon evaluation (Run 6):**

| Horizon | Baseline MAPE | Finetuned MAPE | Delta | DirAcc Delta |
|---------|--------------|----------------|-------|-------------|
| 7 days | 2.5% | 2.8% | +0.3pp | -19.0pp |
| 30 days | 4.7% | 5.1% | +0.4pp | +0.0pp |
| 60 days | 5.7% | 6.3% | +0.5pp | +6.9pp |
| 120 days | 8.1% | 8.5% | +0.5pp | +4.0pp |

**Conclusion:** With every hyperparameter held constant, the only difference from Run 4 was 100 tickers vs 10. The regression confirms that diverse training data dilutes the signal for these specific eval tickers. The model learns a broader but less precise representation. CVX still benefits (-2.0pp at 120d), but UNH (+0.5pp) and BA (+2.1pp) degrade.

### Run 6b: Continued from Run 6 Best, 100 Epochs (2026-02-22)

**Outcome: No further improvement.** Continuing from Run 6's best checkpoint with the same data and hyperparameters early-stopped at epoch 22 with marginal differences.

**Training:** Started from Run 6 best checkpoint (epoch 40, val loss 0.015296). Val loss immediately low (0.007011 at epoch 1). Best val loss **0.006729** at epoch 7. Early stopped at epoch 22 — model had already converged.

**Multi-horizon evaluation (Run 6b):**

| Horizon | Baseline MAPE | Finetuned MAPE | Delta | DirAcc Delta |
|---------|--------------|----------------|-------|-------------|
| 7 days | 2.5% | 2.8% | +0.3pp | -16.7pp |
| 30 days | 4.7% | 5.1% | +0.3pp | +0.0pp |
| 60 days | 5.7% | 6.1% | +0.4pp | +7.8pp |
| 120 days | 8.1% | 8.2% | +0.1pp | +4.0pp |

**Conclusion:** Very similar to Run 6. The slight improvement at 120d (+0.1pp vs +0.5pp) suggests continued training helped marginally, but the model was already near convergence. **Run 4 remains the best model.**

**Plots:** See [evaluation_results_run6_h7.png](evaluation_results_run6_h7.png) through [evaluation_results_run6b_h120.png](evaluation_results_run6b_h120.png).

## Broad Evaluation: 50 Held-Out Tickers

To test generalization beyond the original 6 eval tickers, all models were evaluated on 50 tickers not in the Run 4 training set (^GSPC, AAPL, JNJ, JPM, XOM, PG, CAT, NEE, AMT, WMT). The 50 tickers span technology, healthcare, financials, energy, consumer staples, industrials, utilities, consumer discretionary, and communication sectors.

### Overall MAPE Comparison (50 tickers)

| Horizon | Baseline | Run 4 (10 tickers) | Run 5 (100 tickers) | Run 6 (100 tickers) | Run 6b (continued) |
|---------|----------|-------------------|--------------------|--------------------|-------------------|
| 7 days | 2.1% | **2.1% (-0.0pp)** | 2.2% (+0.1pp) | 2.2% (+0.1pp) | 2.3% (+0.1pp) |
| 30 days | 4.9% | **4.9% (-0.0pp)** | 5.0% (+0.1pp) | 4.9% (+0.0pp) | 4.9% (+0.0pp) |
| 60 days | 7.0% | **6.9% (-0.1pp)** | 7.2% (+0.2pp) | 7.2% (+0.2pp) | 7.2% (+0.2pp) |
| 120 days | 10.4% | **10.2% (-0.2pp)** | 10.5% (+0.1pp) | 11.0% (+0.6pp) | 10.9% (+0.5pp) |

**Key findings:**
- **Run 4 is the only model that improves (or ties) at every horizon** on 50 unseen tickers
- Runs 5, 6, 6b all regress — confirming that 100-ticker training hurts generalization
- Run 4's improvement is modest at scale (-0.2pp at 120d vs -1.0pp on 6-ticker eval) but consistent
- At 120d, Run 4 improved 29/50 tickers (58% win rate)

### Run 4 — Top Improvers at 120 Days (50 tickers)

| Ticker | Baseline MAPE | Finetuned MAPE | Delta |
|--------|--------------|----------------|-------|
| ADBE | 16.3% | 11.9% | **-4.3pp** |
| UNH | 11.6% | 8.0% | **-3.5pp** |
| AVGO | 14.6% | 11.2% | **-3.4pp** |
| IBM | 18.1% | 15.0% | **-3.1pp** |
| NKE | 21.9% | 19.5% | **-2.4pp** |
| NVDA | 6.9% | 4.8% | **-2.1pp** |
| CVX | 7.0% | 5.2% | **-1.8pp** |

### Run 4 — Top Degradations at 120 Days (50 tickers)

| Ticker | Baseline MAPE | Finetuned MAPE | Delta |
|--------|--------------|----------------|-------|
| ORCL | 18.4% | 21.7% | +3.3pp |
| T | 11.2% | 14.4% | +3.2pp |
| BLK | 2.4% | 5.2% | +2.8pp |

**Plots:** See `evaluation_results_run4_50tickers_h*.png`, `evaluation_results_run5_50tickers_h*.png`, `evaluation_results_run6_50tickers_h*.png`, `evaluation_results_run6b_50tickers_h*.png`.

## Feature Engineering: External Regressors (xreg)

TimesFM supports external regressors (xreg) — a post-hoc ridge regression layer that corrects forecasts using covariates at inference time (no retraining needed). We implemented this in NumPy (the upstream `xreg_lib.py` uses JAX) and tested with calendar + statistical covariates.

**Covariates used:**
- **day_of_week** — dynamic categorical (7 values), known for future dates
- **month** — dynamic categorical (12 values), known for future dates
- **historical_volatility** — static numerical (std of log returns in context)
- **avg_log_return** — static numerical (mean log return in context)

**Approach:** "timesfm + xreg" — TimesFM provides the base forecast, ridge regression on covariates corrects the residuals.

### 4-Way Comparison: Baseline vs Baseline+xreg vs Finetuned (Run 4) vs Finetuned+xreg

| Horizon | Baseline | Baseline+xreg | Finetuned | Finetuned+xreg |
|---------|----------|---------------|-----------|----------------|
| 7 days | 2.5% | 2.5% (-0.0pp) | **2.0% (-0.5pp)** | 2.0% (-0.5pp) |
| 30 days | 4.7% | 4.6% (-0.1pp) | 4.5% (-0.2pp) | **4.4% (-0.3pp)** |
| 60 days | 5.7% | 11.0% (+5.2pp) | **5.5% (-0.2pp)** | 10.1% (+4.3pp) |
| 120 days | 8.1% | 16.6% (+8.6pp) | **7.0% (-1.0pp)** | 14.7% (+6.7pp) |

### Per-Ticker Highlights

**At 30 days, xreg helps specific tickers dramatically:**
- **UNH**: Finetuned+xreg 3.4% vs Finetuned 9.4% (**-6.0pp**) — ridge regression captures a correction the model missed
- **BA**: Finetuned+xreg 3.3% vs Finetuned 7.0% (**-3.6pp**)
- **GS**: Finetuned+xreg 9.7% vs Finetuned 4.2% (+5.5pp) — xreg overcorrects

**At 7 days, xreg is neutral to slightly helpful** across all tickers.

### Conclusion

**xreg corrections are harmful at longer horizons (60d, 120d).** The ridge regression is fitted on context-period patterns, but the corrections grow proportionally over the horizon without any dampening, leading to large divergences. At 30 days, the effect is mixed — spectacular per-ticker improvements (UNH, BA) alongside harmful overcorrections (GS, NVDA). At 7 days, the effect is negligible.

**The pure finetuned model (Run 4) remains the best overall approach.** xreg with simple calendar/statistical features adds noise rather than signal for multi-horizon forecasting. Future work could explore:
1. Decaying the xreg correction over the horizon to prevent divergence
2. More informative covariates (sector-specific features, macro indicators)
3. Ensemble approaches that weight xreg corrections by horizon

**Plots:** See `evaluation_results_xreg_h*.png`.

## Experiment Log
See [experiment_log.md](experiment_log.md) for full hyperparameters and per-epoch losses.

## Scripts
- `src/timesfm/finetune_timesfm.py` — Finetuning script (supports `--optimizer sgd|adam|adamw`, `--freeze-layers`, `--accumulation-steps`)
- `scripts/evaluate_forecast.py` — Evaluation script (baseline vs finetuned, with optional `--use-covariates` for xreg)
- `scripts/sector_router.py` — Bootstrap-based tier classification, policy-enforced routing, and forecasting (see below)
- `scripts/xreg_numpy.py` — NumPy ridge regression for xreg (replaces JAX-based `xreg_lib.py`)
- `scripts/build_covariates.py` — Covariate construction helpers (calendar + statistical features)

---

## Multi-Region Finetuning & Evaluation (2026-03 — 2026-04)

After the US experiments above, the finetuning recipe (Run 4 config: AdamW lr=1e-4, wd=0.01, freeze 17/20, 16x grad accum, cosine + 5-epoch warmup) was applied to Japan, Europe, and China/HK. Full results are in [international_finetuning_summary.md](international_finetuning_summary.md) and the top-level [sector_routing_results.md](../sector_routing_results.md).

### Regional Checkpoints

| Region | Checkpoint | Training tickers | Eval tickers (held-out) |
|--------|-----------|-----------------|------------------------|
| US | `finetune_usa/checkpoints/best` | AAPL, MSFT, AMZN, GOOGL, META, JPM | NVDA, UNH, GS, CVX, KO, BA |
| Japan | `finetune_japan/checkpoints/run_glia_t2_r2a/best` | 7203.T, 6758.T, 9984.T, 8306.T, 4502.T, 7974.T | 8035.T, 6902.T, 4661.T |
| Europe | `finetune_europe/checkpoints/run_glia_v2_1a/best` | ASML, SAP, NVO, AZN, SHEL, DEO | UL, EADSF, VWAGY |
| China | `finetune_china/checkpoints/run_glia_2a/best` | 0700.HK, 0941.HK, 2318.HK, 1299.HK, 0005.HK, 0001.HK | 0388.HK, 2382.HK, 1177.HK |

### Key Cross-Region Finding

Finetuning is a **ticker-level** improvement, not a market-level one. Each region has tickers that significantly regress under finetuning. The primary predictor of success is **sector alignment** between training and eval tickers (more important than data volume, val_loss, or hyperparameters). See `international_finetuning_summary.md` for the full sector-matching analysis.

---

## Finetuned Model Evaluation Procedure

Adopted by board review 2026-04-08, implemented 2026-04-09. This procedure replaces ad-hoc per-run evaluation with a standardized, auditable pipeline.

### Step 1 — Bootstrap Build (per region)

```bash
cd timesfm
source .venv/bin/activate  # or .venv/Scripts/activate on Windows

python scripts/sector_router.py build \
  --region us \
  --honest-split \
  --n-windows 30 \
  --n-boot 10000 \
  --seed 42
```

This runs per-ticker, per-horizon bootstrap evaluation (10,000 resamples over 30 rolling windows) and classifies each ticker into:

- **Tier 1:** finetuned model significantly better (95% CI entirely > 0)
- **Tier 2:** finetuned model significantly worse (95% CI entirely < 0)
- **Tier 3:** indeterminate (95% CI spans 0) — defaults to baseline

The `--honest-split` flag enables temporal out-of-sample validation: tiers are classified on the older 15 windows, then evaluated on the newer 15 windows. Without it, classification and evaluation use the same windows (in-sample only).

**Outputs per region:**

| File | Purpose |
|------|---------|
| `routing_table.json` | Machine-readable tier labels, policy, seed, CIs |
| `routing_table.md` | Human-readable report with governance annotations |
| `tier_labels_{date}_seed{N}.csv` | Frozen audit artifact — one row per ticker per horizon |

### Step 2 — Read the Report

The markdown report includes the following governance features:

- **Policy field:** Each region is assigned either `simple` (horizon-only rule) or `router` (per-ticker tier routing). Policy is recorded in the JSON and printed in the report header.
- **Noise flag:** Horizons at or below 14d are flagged `[NOISE]`. Do not interpret short-horizon deltas as signal — pretrained TimesFM is already well-calibrated for daily.
- **Median MAPE:** Reported alongside the mean to expose outlier dependence.
- **Worst-ticker disclosure:** The single ticker with the largest regression (delta vs baseline) is named in every summary row.

### Step 3 — Forecast with Policy Enforcement

```bash
# US at 120d: simple policy applies FT-only (horizon >= 60d)
python scripts/sector_router.py forecast \
  --region us \
  --tickers "NVDA,MSFT,GS" \
  --horizon 120

# China at 120d: router policy applies per-ticker tier labels
python scripts/sector_router.py forecast \
  --region china \
  --tickers "0388.HK,2382.HK,0700.HK" \
  --horizon 120

# Override policy (e.g., force per-ticker routing for US)
python scripts/sector_router.py forecast \
  --region us \
  --tickers "NVDA,MSFT,GS" \
  --horizon 120 \
  --policy router
```

A **sizing governor warning** is printed for all forecasts at horizon >= 60d, requiring a second orthogonal signal before any capital allocation. A **noise warning** is printed for horizon <= 14d.

### Deployment Policies (Board-Adopted 2026-04-08)

| Region | Policy | Rule | Rationale |
|--------|--------|------|-----------|
| US | `simple` | FT at 60d+, baseline below | Per-ticker routing adds ~0pp vs FT-only OOS |
| Japan | `simple` | FT at 60d+, baseline below | Routing hurts at 120d OOS, helps mid-horizon but not enough to justify complexity |
| Europe | `simple` | FT at 60d+, baseline below | Routing hurts vs FT-only at every horizon OOS |
| China | `router` | Per-ticker tier labels | Routing adds +2.47pp at 120d OOS — only region where it survives temporal validation |

### Known Degradation Tickers (120d)

These tickers consistently regress under finetuning. Under the `simple` policy they are included in FT-only at 60d+ (accepted tradeoff). Under `router` they are routed to baseline.

| Ticker | Region | Delta vs BL (120d OOS) | Sector |
|--------|--------|----------------------|--------|
| MSFT | US | -0.83pp | Tech |
| 8306.T (MUFG) | Japan | -5.75pp | Banking |
| 6902.T (Denso) | Japan | -1.01pp | Auto parts |
| VWAGY | Europe | -1.60pp | Auto |
| DEO (Diageo) | Europe | -2.78pp | Consumer staples |
| 0005.HK (HSBC) | China | -3.25pp | Banking |
| 0001.HK (CKH) | China | -0.36pp | Conglomerate |

### OOS Performance Summary (2026-04-09 rebuild)

Policy vs Baseline, pp MAPE improvement on temporal OOS windows:

| Region | Policy | 60d | 120d | Worst ticker (120d) |
|--------|--------|-----|------|---------------------|
| US | simple | +0.41 | +0.96 | MSFT (-0.83pp) |
| Japan | simple | +0.60 | +1.60 | 8306.T (-5.75pp) |
| Europe | simple | +2.52 | +2.95 | VWAGY (-1.60pp) |
| China | router | +1.01 | +2.47 | 0700.HK (-1.29pp) |

### Rebuild Cadence

- **China router:** quarterly rebuild required. Alert on any ticker's tier flip between rebuilds.
- **US/Japan/Europe:** rebuild on checkpoint change or after new finetuning runs.
- **Seed:** always record via `--seed` for reproducibility. Default is 42.

### Audit Requirements

Before any capital deployment:

1. **Frozen CSV** (`tier_labels_{date}_seed{N}.csv`) must exist for the active routing table.
2. **P&L-weighted evaluation** (not just MAPE) must be conducted — MAPE is a diagnostic, not a decision metric.
3. **Orthogonal second signal** required for position sizing at 60d+ horizons.
4. **Adversarial review** by someone other than the model builder.

See `personas/board_meetings/timesfm_regional_finetuning_apr2026.md` for the full board review.
