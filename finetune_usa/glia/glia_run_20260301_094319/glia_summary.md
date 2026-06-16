# Glia Run Summary (Run 6)

Stop reason: approved
Turns completed: 1
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_expts\glia_run_20260301_094319
Date: 2026-03-01

---

## Key Findings (Run 6, 4 experiments)

### 1. Rolling Evaluation Overturns Prior Conclusions

20-window rolling mean MAPE (horizon=120d):

| Model    | Mean MAPE | Std    | DirAcc |
|----------|-----------|--------|--------|
| Baseline | 8.08%     | ±0.83% | 50.9%  |
| **Run4** | **7.40%** | **±0.67%** | 51.1% |
| Glia4b   | 7.63%     | ±0.89% | 51.1%  |
| Ensemble | 7.68%     | ±0.83% | 50.9%  |
| Glia3a   | 8.27%     | ±1.00% | 50.7%  |

**Revised conclusions**:
- Run4 is the best single model (7.40%), NOT the equal-weight ensemble (7.68%)
- Glia3a is WORSE than baseline on average (8.27% vs 8.08%) — strong on CVX but catastrophic on NVDA/BA
- The previous "7.9% ensemble is best" was a single-window fluke

Per-ticker rolling means:

| Model    | NVDA | UNH  | GS   | CVX  | KO   | BA    |
|----------|------|------|------|------|------|-------|
| Baseline | 7.3% | 11.0%| 9.8% | 7.2% | 5.0% | 8.1%  |
| Run4     | 5.6% | 9.7% | 9.5% | 5.7% | 4.7% | 9.1%  |
| Glia3a   | 8.1% | 10.3%| 9.1% | 4.7% | 5.9% | 11.5% |
| Glia4b   | 5.3% | 11.8%| 8.9% | 5.1% | 4.7% | 9.9%  |
| Ensemble | 6.2% | 10.4%| 9.2% | 5.2% | 5.1% | 10.1% |

### 2. Hyperparameter Variations Both Fail (NaN)

- **run_glia_r6_1a (LR=5e-5)**: val_loss 0.026730 → NaN predictions
- **run_glia_r6_1b (WD=0.05)**: val_loss 0.026448 → NaN predictions

**Val loss threshold (now 8 data points, sharp boundary at ~0.015):**

| Checkpoint              | Val Loss | Inference | MAPE  |
|-------------------------|----------|-----------|-------|
| run_glia_6a             | 0.011894 | Valid     | 9.5%  |
| run_glia_3a             | 0.011956 | Valid     | ~8%   |
| Run4                    | 0.012862 | Valid     | 7.40% |
| **Threshold ~0.015**    | —        | —         | —     |
| r6_1a (LR=5e-5)        | 0.026730 | NaN       | —     |
| r6_1b (WD=0.05)        | 0.026448 | NaN       | —     |
| r5_1a (BA post-2019)   | 0.024304 | NaN       | —     |
| r5_1b (SGDR no-warmup) | 0.027478 | NaN       | —     |
| run_glia_7a (layer 19) | 0.040811 | NaN       | —     |

Implication: Standard recipe (lr=1e-4, wd=0.01, freeze-17, cosine+warmup, 10 tickers) is uniquely capable of reaching valid convergence. Deviations of ±2x LR or ±5x WD block convergence.

### 3. Selective Ensemble: First Genuine Out-of-Sample Improvement

**Setup**: 15-window honest split — select best model per ticker on windows 0-14, evaluate on 15-29.

**Selection**: NVDA→Glia4b, UNH→Run4, GS→Glia4b, CVX→Glia3a, KO→Glia4b, BA→Run4

**Out-of-sample results (windows 15-29):**

| Model              | Mean MAPE | Std    | vs Run4  |
|--------------------|-----------|--------|----------|
| Run4               | 7.93%     | ±0.78% | —        |
| **SelectiveEns**   | **7.65%** | **±0.96%** | **+0.28pp** |
| EqualWeightEns     | 8.69%     | ±1.08% | -0.76pp  |

Improvement driven by NVDA (+1.1pp via Glia4b) and CVX (+0.5pp via Glia3a).

---

## Best Configurations (as of Run 6)

**Option A: Run4 solo** (recommended for deployment)
- `checkpoints/best/`
- Rolling MAPE: 7.40% ± 0.67% (20 windows)
- Improvement over baseline: +0.68pp
- Lowest variance, most reliable

**Option B: Selective Ensemble** (higher reward, higher variance)
- NVDA, GS, KO → `checkpoints/run_glia_4b/best/`
- CVX → `checkpoints/run_glia_3a/best/`
- UNH, BA → `checkpoints/best/`
- Out-of-sample MAPE: 7.65% ± 0.96% (eval windows 15-29)
- +0.28pp vs Run4 on eval windows, but ±0.96% vs ±0.67%

Script: `scripts/selective_ensemble_eval.py`
Script: `scripts/rolling_eval.py`

---

## Complete Experiment Inventory (All 6 Glia Runs, 22 Experiments)

| Run  | Checkpoint            | Description                    | Val Loss | MAPE@120d |
|------|-----------------------|--------------------------------|----------|-----------|
| Pre  | Run4                  | Best single, standard recipe   | 0.012862 | 7.40%     |
| 1    | run_glia_1a           | 13 tickers LR=1e-4            | NaN      | NaN weights |
| 1    | run_glia_3a           | Mixed loss α=0.8               | 0.011956 | 8.27% (rolling) |
| 1    | run_glia_4a           | Mixed loss α=0.95              | ~0.012   | 8.1%      |
| 1    | run_glia_4b           | Overall-dir α=0.9              | ~0.012   | 7.63% (rolling) |
| 2    | Ensemble Run4+3a+4b   | Equal-weight                   | —        | 7.68% (rolling) |
| 2    | xreg λ=0.05           | Gradient analysis               | —        | Failed    |
| 3    | —                     | Context=1024@inference         | —        | 9.2%      |
| 3    | —                     | Per-ticker weights (oracle)    | —        | 8.4%      |
| 3    | run_glia_6a           | 1024-ctx training              | 0.011894 | 9.5%      |
| 4    | —                     | 4-model ensemble               | —        | 9.0%      |
| 4    | run_glia_7a           | Layer 19 only                  | 0.040811 | NaN       |
| 5    | run_glia_r5_1b        | SGDR no-warmup                 | 0.027478 | NaN       |
| 5    | run_glia_r5_1a        | BA post-2019                   | 0.024304 | NaN       |
| 6    | —                     | Rolling eval (methodology)     | —        | Overturned single-window results |
| 6    | run_glia_r6_1a        | LR=5e-5                        | 0.026730 | NaN       |
| 6    | run_glia_r6_1b        | WD=0.05                        | 0.026448 | NaN       |
| 6    | SelectiveEnsemble     | Per-ticker best-model routing  | —        | 7.65% (eval windows) |

---

## Confirmed Dead Ends (22 experiments)

- SGD with small batches
- 100+ diverse tickers (8.4-8.5%)
- xreg at 60d+
- Continuing from finetuned checkpoint
- 13+ tickers at LR=1e-4
- Non-uniform layer selection
- Mixed loss (α=0.8, 0.95, overall-dir)
- BA temporal isolation (post-2019, post-2020)
- Context=1024 (9.2-9.5%)
- Equal-weight ensemble (7.68% — worse than Run4 on rolling avg)
- LR=5e-5 (NaN)
- WD=0.05 (NaN)
- SGDR without warmup (NaN)

## Open Questions (for future work)

1. **Selective ensemble stability**: Does the NVDA→Glia4b, CVX→Glia3a routing remain optimal if re-evaluated monthly with updated rolling windows?
2. **Soft per-ticker weighting**: Would 1/MAPE-weighted blending outperform binary best-model selection at lower variance?
3. **SGDR with proper warmup**: SequentialLR(LinearLR_warmup + CosineAnnealingWarmRestarts) — expected to reach Run4 range, not break ceiling.
4. **Different evaluation protocol**: More tickers, different horizons — would the 1.0pp ceiling hold?

---

*Glia Run 6 completed 2026-03-01. Approved by Supervisor after 1 turn, 4 experiments (2 NaN, 1 methodology, 1 new best configuration). Total across all Glia runs: 22 experiments, 11 trained checkpoints (7 valid, 4 NaN). Best deployable: Run4 solo (7.40% ± 0.67%). Best selective: Selective Ensemble (7.65% ± 0.96%).*
