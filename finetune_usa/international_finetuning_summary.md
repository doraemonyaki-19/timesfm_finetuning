# International Finetuning Summary — All Regions

Date: 2026-03-26
Regions: China/HK, Japan, Europe
Model: TimesFM 2.5 (200M params, PyTorch)
Recipe: AdamW lr=1e-4, wd=0.01, freeze 17/20 layers, 16x grad accum, cosine+5ep warmup, stride=32

---

## 1. Executive Summary

| Region | Train Tickers | Val Loss | Eval MAPE (Rolling) | vs Baseline | Training Ticker MAPE (120d) | vs Baseline |
|--------|--------------|----------|--------------------:|------------:|----------------------------:|------------:|
| **China (Glia2a, 10tk)** | 10 | 0.012067 | **10.86%** | **+6.62pp** | 8.7% | **-1.9pp** |
| China (v2, 17tk) | 17 | 0.014907 | 13.67% | +3.81pp | — | — |
| Japan (v2, 18tk) | 18 | 0.012469 | 19.59% | -1.49pp | 13.5% | +0.3pp |
| **Europe (v2, 17tk)** | 17 | — | **4.99%** | **+0.15pp** | 16.6% | **-1.2pp** |

**Key finding:** Finetuning effectiveness is determined by **sector alignment between training and eval tickers**, not by training set size, val_loss, or hyperparameters.

---

## 2. Region-by-Region Results

### 2.1 China/HK — BEST REGION

**Best checkpoint: Glia2a (10 tickers)**
- Path: `finetune_china/checkpoints/run_glia_2a/best`
- Rolling MAPE: 10.86% ±2.15% (baseline 17.48%) → **+6.62pp improvement**
- Val loss: 0.012067

**Why it works:** The 10 training tickers (Tencent, Alibaba, AIA, China Mobile, Meituan,
NetEase, Xiaomi, Ping An, HSBC) are sector-matched to eval tickers (HKEX, Sunny Optical,
Sino Biopharma). All are HK-listed tech/finance/consumer companies.

**Data dilution confirmed:** Expanding to 17 tickers (adding utilities, conglomerates,
real estate, banking) diluted signal: 13.67% vs 10.86%. The additional sectors don't
help forecast tech/optical/pharma eval tickers.

**Training ticker evaluation:** Glia2a genuinely improves on its own training tickers
(up to -1.9pp at 120d, 2318.HK Ping An improved -8.2pp). Confirms the model learns
real patterns, not just memorizing.

### 2.2 Japan — FINETUNING NON-VIABLE

**Status: Use baseline (no finetuned checkpoint recommended)**

**Val loss breakthrough:** 18 tickers + stride=32 broke the prior 10-ticker floor
of 0.027, reaching 0.012469 — a 54% reduction. But this did NOT translate to
forecast improvement.

**Eval ticker results (rolling):** 19.59% vs baseline 18.10% = **-1.49pp regression**
- 8035.T (Tokyo Electron): +4.9pp worse — semiconductor volatility (33% baseline MAPE)
  is amplified by patterns learned from banks/telecom/chemicals
- Only 4661.T (Oriental Land/Disney Japan) improved (-1.6pp)

**Training ticker evaluation:** Even on its own training tickers, Japan finetuning
shows no improvement (+0.3pp at 120d). This is fundamentally different from China
and Europe, where training tickers improve.

**Root cause:** Japanese equity price dynamics may be structurally different from
the pretrained model's training distribution. The model can fit training data
(val_loss 0.012) but the learned patterns don't generalize even within the
Japanese market.

### 2.3 Europe — MARGINAL

**Checkpoint: `finetune_europe/checkpoints/run_glia_v2_1a/best`**
- Rolling MAPE: 4.99% ±0.47% (baseline 5.14%) → **+0.15pp improvement**
- Statistically insignificant (within baseline std of ±0.33%)

**Eval ticker breakdown (120d single-window):**
- UL (Unilever): -0.1pp (neutral)
- EADSF (Airbus): -0.9pp (improved)
- VWAGY (Volkswagen): +1.3pp (regressed)

**Training ticker evaluation — strong improvements at longer horizons:**

| Horizon | Baseline | Finetuned | Delta |
|---------|----------|-----------|-------|
| 30d | 6.0% | 5.3% | -0.7pp |
| 60d | 10.3% | 8.7% | **-1.6pp** |
| 90d | 12.8% | 11.0% | **-1.8pp** |
| 120d | 17.8% | 16.6% | **-1.2pp** |

SAP improved -8.0pp at 120d, ASML -2.7pp, AZN -2.5pp. DEO (Diageo) is the
"BA of Europe" — consistently degrades across all horizons.

---

## 3. Cross-Market Insights

### 3.1 The Sector Matching Hypothesis (CONFIRMED)

| Region | Training-Eval Sector Match | Eval Improvement |
|--------|---------------------------|-----------------|
| China (Glia2a) | HIGH — tech/finance overlap | **+6.62pp** |
| China (v2) | DILUTED — added utilities/RE | +3.81pp |
| Europe | LOW — no auto/aerospace in training | +0.15pp |
| Japan | NONE — semiconductor/auto eval vs telecom/bank training | -1.49pp |

**Conclusion:** Sector matching is the #1 predictor of finetuning success.
Adding more tickers from unrelated sectors hurts, not helps.

### 3.2 Val Loss ≠ Forecast Quality

| Checkpoint | Val Loss | Eval MAPE Change |
|-----------|----------|-----------------|
| China Glia2a (10tk) | 0.012067 | +6.62pp |
| Japan v2 (18tk) | 0.012469 | -1.49pp |
| China v2 (17tk) | 0.014907 | +3.81pp |

Japan achieves nearly the same val_loss as China's best checkpoint but regresses
on eval. Val_loss measures training data fit, not forecast generalization.

### 3.3 Training Ticker Evaluation Reveals True Learning

| Region | Training Ticker Δ (120d) | Eval Ticker Δ | Interpretation |
|--------|-------------------------|---------------|----------------|
| China | -1.9pp | +6.62pp | Learns + transfers |
| Europe | -1.2pp | +0.15pp | Learns but doesn't transfer |
| Japan | +0.3pp | -1.49pp | Doesn't learn at all |

This three-way comparison is the clearest diagnostic:
- **China:** Model learns patterns that transfer to similar-sector eval tickers
- **Europe:** Model learns patterns but eval tickers are too different to benefit
- **Japan:** Model doesn't learn useful patterns even for its own training tickers

### 3.4 The "Problematic Ticker" Pattern

Each region has a ticker that consistently degrades after finetuning:
- **US:** BA (Boeing) — defense/aerospace, high volatility post-2020
- **Japan:** 8035.T (Tokyo Electron) — semiconductor, 33% baseline MAPE
- **Europe:** DEO (Diageo) — consumer staples, low volatility

These tickers share no sector but all are structural outliers within their
training sets. The model learns patterns from the majority of tickers that
actively mislead forecasts for the outlier.

---

## 4. Recommendations

### For Deployment

| Region | Recommendation | Checkpoint |
|--------|---------------|------------|
| **US** | Use Run4 finetuned | `checkpoints/best` |
| **China/HK** | Use Glia2a finetuned | `finetune_china/checkpoints/run_glia_2a/best` |
| **Japan** | Use baseline (no finetuning) | `google/timesfm-2.5-200m-pytorch` |
| **Europe** | Use baseline or finetuned (marginal) | `finetune_europe/checkpoints/run_glia_v2_1a/best` |

### For Future Work

1. **Sector-matched training sets:** For any new market, select training tickers
   from the same sectors as eval tickers. 10 well-chosen tickers > 18 diverse ones.

2. **Training ticker evaluation as diagnostic:** Always evaluate on held-out windows
   of training tickers first. If the model can't beat baseline on its own tickers,
   it won't help on eval tickers.

3. **Japan needs a fundamentally different approach.** Standard finetuning doesn't
   work — possibly due to Japanese market microstructure (tick sizes, trading hours,
   cross-shareholding effects). Consider: different model architecture, Japan-specific
   pretrained weights, or sector-level finetuning.

4. **Europe could benefit from sector-matched training.** Current eval tickers are
   Unilever (FMCG), Airbus (aerospace), Volkswagen (auto). Adding more FMCG,
   aerospace, and auto tickers to training while removing banks/telecom could
   improve transfer.

---

## 5. Experiment Log

| Region | Experiment | Tickers | Val Loss | Rolling MAPE | Delta |
|--------|-----------|---------|----------|-------------|-------|
| China | Glia2a (10tk) | 10 | 0.012067 | 10.86% | **+6.62pp** |
| China | v2 (17tk) | 17 | 0.014907 | 13.67% | +3.81pp |
| Japan | v2 (18tk) | 18 | 0.012469 | 19.59% | -1.49pp |
| Europe | v2 (17tk) | 17 | — | 4.99% | +0.15pp |

---

*Generated by Glia international finetuning experiments, March 2026*
