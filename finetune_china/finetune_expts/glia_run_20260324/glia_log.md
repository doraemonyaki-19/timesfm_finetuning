# Glia Run Log - China/HK (Expanded Tickers v2)

Started: 2026-03-24
Output dir: finetune_china/finetune_expts/glia_run_20260324
Max turns: 6

---

## Experiment: China v2 — 17 tickers (expanded from 10), stride=32

### Hypothesis

Expanding training from 10 to 18 tickers (17 actually loaded — 0011.HK/Hang Seng Bank download failed) by adding long-history HK blue chips across new sectors will improve on Glia2a's rolling MAPE of 10.86%.

**New tickers added:** CK Hutchison (0001.HK), SHK Properties (0016.HK), CLP Holdings (0002.HK), HK & China Gas (0003.HK), ICBC (1398.HK), CNOOC (0883.HK), Galaxy Entertainment (0027.HK). All with 15-20+ years of history.

**Failed download:** Hang Seng Bank (0011.HK) — HTTP 404, possibly delisted. 17/18 tickers loaded.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Tickers | 17 (vs 10 in Glia2a) |
| Sliding windows | 1835 (vs 901 in Glia2a) |
| Train/Val split | 1377 / 458 |
| Steps per epoch | 173 |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Frozen layers | 17/20 (27.7% trainable) |
| Grad accumulation | 16× (effective batch=128) |
| Scheduler | Cosine with 5-epoch warmup |
| Stride | 32 |
| Max context | 512 |
| Horizon | 128 |
| Early stopping | 15 epochs patience |

### Training Progress

| Epoch | Val Loss | Notes |
|-------|----------|-------|
| 1 | 0.029199 | Initial |
| 5 | 0.025476 | Warmup complete |
| 10 | 0.021249 | Steady descent |
| 14 | 0.019488 | |
| 19 | 0.018058 | |
| 22 | 0.017156 | |
| 24 | 0.016512 | Approaching 0.015 threshold |
| 28 | 0.015816 | |
| 33 | 0.015639 | |
| **35** | **0.014907** | **Best — crossed 0.015 threshold** |
| 50 | 0.017715 | Early stopping triggered (no improvement for 15 epochs) |

**Key observations:**
- Training completed all 50 epochs, early stopped at epoch 50 (best was epoch 35)
- Best val loss: **0.014907** — just barely crossed the critical 0.015 threshold
- Compare: Glia2a (10 tickers) reached val_loss **0.012067** — significantly lower
- Train loss dropped from 0.029 → ~0.005 (healthy convergence, no NaN)

### Single-Window Evaluation (120d horizon)

| Ticker | Baseline MAPE | v2 Finetuned MAPE | Delta |
|--------|--------------|-------------------|-------|
| 0388.HK (HKEX) | 3.6% | 6.4% | **+2.8pp** (worse) |
| 2382.HK (Sunny Optical) | 27.2% | 21.9% | **-5.3pp** (better) |
| 1177.HK (Sino Biopharma) | 15.3% | 12.3% | **-3.0pp** (better) |
| **Overall** | **15.4%** | **13.5%** | **-1.8pp** |

DirAcc: Baseline 53.6% → Finetuned 48.9% (worse)

### Rolling Evaluation (15 windows, 120d horizon)

| Model | Mean MAPE | Std | vs Baseline |
|-------|-----------|-----|-------------|
| Baseline | 17.48% | ±1.66% | — |
| **China_v2_18tk** | **13.67%** | **±1.84%** | **+3.81pp** |
| China_Glia2a (prior best) | 10.86% | ±2.15% | +6.62pp |
| Ensemble (v2+Glia2a) | 11.95% | ±1.87% | +5.52pp |

**Per-Ticker Rolling Mean MAPE:**

| Model | 0388.HK | 2382.HK | 1177.HK |
|-------|---------|---------|---------|
| Baseline | 4.4% | 22.0% | 26.0% |
| China_v2_18tk | 4.2% | 15.4% | 21.5% |
| China_Glia2a | 4.9% | 9.7% | 18.0% |

### Analysis

1. **Val loss:** 0.014907 — just barely below the 0.015 NaN threshold. Glia2a reached 0.012, indicating the model struggles more with 17 diverse tickers.

2. **MAPE improvement real but inferior:** -3.81pp vs baseline is significant, but Glia2a achieves -6.62pp. The additional 7 tickers (utilities, conglomerates, real estate, banking) dilute the signal for the eval tickers (exchange, optical, pharma).

3. **0388.HK regression:** v2 worsens 0388.HK in single-window eval (+2.8pp) though rolling eval shows slight improvement (4.2% vs baseline 4.4%). This ticker's structural mismatch (exchange fees ≠ stock prices) persists.

4. **Pattern confirmed across markets:** This is the same "more data = dilution" finding from US experiments (Run 5/6 with 100 tickers regressed to 8.4-8.5% from Run 4's 7.0%). The optimal training set is **small and sector-matched** to eval tickers, not large and diverse.

5. **Ensemble doesn't help:** v2 + Glia2a ensemble (11.95%) is worse than Glia2a alone (10.86%).

### Conclusion

**Glia2a (10 tickers, stride=32) remains the best China/HK checkpoint.**

The expanded 17-ticker v2 experiment confirms the data dilution hypothesis: adding tickers from unrelated sectors (utilities, real estate, conglomerates) to a training set targeting tech/finance/pharma eval tickers reduces rather than improves performance.

**Best China configuration (unchanged):**
- Checkpoint: `finetune_china/checkpoints/run_glia_2a/best`
- Rolling MAPE: 10.86% (±2.15%) — improvement of **6.62pp** vs baseline
- Recipe: 10 HK tickers, stride=32, AdamW lr=1e-4, freeze=17, val_loss=0.012

---
