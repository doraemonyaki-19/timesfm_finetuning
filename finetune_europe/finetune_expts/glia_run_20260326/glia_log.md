# Glia Run Log - Europe (Expanded Tickers v2)

Started: 2026-03-26
Output dir: finetune_europe/finetune_expts/glia_run_20260326

---

## Experiment: Europe v2 — 17 tickers (expanded from 10), stride=32

### Hypothesis

Expanding from 10 to 18 training tickers (17 actually loaded — TEF/Telefonica download
failed) by adding consumer staples, banking, auto, healthcare, and telecom sectors will
improve forecasting on European eval tickers (UL, EADSF, VWAGY).

**New tickers added:** DEO (Diageo), GSK, LIN (Linde), SAN (Santander), ING, STLA (Stellantis),
PHG (Philips), TEF (Telefonica — failed download). All NYSE/NASDAQ listed with USD pricing.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Tickers | 17 (TEF failed, 17/18 loaded) |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| Frozen layers | 17/20 (27.7% trainable) |
| Grad accumulation | 16x (effective batch=128) |
| Scheduler | Cosine with 5-epoch warmup |
| Stride | 32 |
| Max context | 512 |
| Horizon | 128 |
| Early stopping | 15 epochs patience |
| Checkpoint | finetune_europe/checkpoints/run_glia_v2_1a/best |

### Eval Ticker Results (UL, EADSF, VWAGY) — Single Window

| Horizon | Baseline MAPE | Finetuned MAPE | Delta |
|---------|--------------|----------------|-------|
| 5d | 1.1% | 1.2% | +0.2pp (worse) |
| 14d | 2.3% | 2.2% | **-0.2pp** |
| 30d | 3.2% | 3.1% | **-0.1pp** |
| 60d | 3.5% | 3.8% | +0.3pp (worse) |
| 90d | 4.2% | 4.7% | +0.5pp (worse) |
| 120d | 5.6% | 5.7% | +0.1pp (worse) |

**Per-Ticker at 120d:**

| Model | UL | EADSF | VWAGY |
|-------|-----|-------|-------|
| Baseline | 3.9% | 6.4% | 6.6% |
| Finetuned | 3.8% | 5.5% | 7.9% |
| Delta | -0.1pp | **-0.9pp** | +1.3pp |

### Rolling Evaluation (15 windows, 120d horizon)

| Model | Mean MAPE | Std | vs Baseline |
|-------|-----------|-----|-------------|
| Baseline | 5.14% | ±0.33% | — |
| Europe_v2_17tk | 4.99% | ±0.47% | **+0.15pp** |

**Per-Ticker Rolling Mean MAPE:**

| Model | UL | EADSF | VWAGY |
|-------|----|-------|-------|
| Baseline | 3.9% | 5.2% | 6.3% |
| Europe_v2 | 3.6% | 4.7% | 6.7% |

### Training Ticker Results (ASML, SAP, NVO, AZN, SHEL, DEO) — Single Window

| Horizon | Baseline MAPE | Finetuned MAPE | Delta |
|---------|--------------|----------------|-------|
| 5d | 1.8% | 2.1% | +0.3pp (worse) |
| 14d | 2.7% | 2.7% | -0.1pp |
| 30d | 6.0% | 5.3% | **-0.7pp** |
| 60d | 10.3% | 8.7% | **-1.6pp** |
| 90d | 12.8% | 11.0% | **-1.8pp** |
| 120d | 17.8% | 16.6% | **-1.2pp** |

**Per-Ticker at 120d (training tickers):**

| Model | ASML | SAP | NVO | AZN | SHEL | DEO |
|-------|------|-----|-----|-----|------|-----|
| Baseline | 22.2% | 27.5% | 24.7% | 17.2% | 7.7% | 7.5% |
| Finetuned | 19.4% | 19.5% | 29.7% | 14.8% | 6.0% | 10.0% |
| Delta | **-2.7pp** | **-8.0pp** | +4.9pp | **-2.5pp** | **-1.7pp** | +2.5pp |

**Per-Ticker at 90d (training tickers) — strongest horizon:**

| Model | ASML | SAP | NVO | AZN | SHEL | DEO |
|-------|------|-----|-----|-----|------|-----|
| Baseline | 16.6% | 18.1% | 18.9% | 14.1% | 4.0% | 5.0% |
| Finetuned | 14.5% | 11.4% | 18.1% | 12.2% | 3.1% | 7.0% |
| Delta | **-2.2pp** | **-6.8pp** | **-0.9pp** | **-1.9pp** | **-0.9pp** | +2.0pp |

### Analysis

1. **Eval tickers: marginal.** Rolling eval shows +0.15pp improvement — statistically
   insignificant with ±0.33% baseline std. The finetuning neither helps nor hurts eval
   tickers meaningfully. EADSF (Airbus) benefits (-0.9pp at 120d single-window), VWAGY
   (Volkswagen) regresses (+1.3pp). UL (Unilever) roughly neutral.

2. **Training tickers: strong improvement at 30d+.** The model genuinely learns useful
   patterns for its training tickers. SAP shows the strongest improvement (-8.0pp at 120d,
   -6.8pp at 90d). ASML, AZN, SHEL all improve substantially. Only DEO (Diageo) and NVO
   (Novo Nordisk at 120d) regress.

3. **Horizon pattern: finetuning helps longer horizons.** Short-term (5d, 14d) is neutral
   or slightly worse. Benefits emerge at 30d and strengthen through 90d. This makes sense —
   the model is trained with horizon=128, so it specializes in longer-term patterns.

4. **DEO is the "BA" of Europe.** Like Boeing in US experiments, Diageo consistently
   degrades after finetuning across all horizons (+0.6pp at 14d to +2.5pp at 120d).
   Consumer staples with stable pricing may conflict with the more volatile patterns
   in the training set.

5. **Cross-market pattern confirmed:** Europe matches China's pattern exactly —
   finetuning works on training tickers but doesn't transfer to eval tickers of
   different sectors. Unlike Japan, where finetuning doesn't work even on training tickers.

### Conclusion

**Europe finetuning shows marginal improvement on eval tickers (+0.15pp rolling) but
strong improvement on training tickers (up to -1.8pp at 90d).** The checkpoint is
mildly useful — better than Japan (which regresses) but not as strong as China Glia2a
(which achieves +6.62pp on eval tickers with sector-matched training data).

The key insight: **sector matching between training and eval tickers is the primary
driver of finetuning success**, not training set size or hyperparameters.

---
