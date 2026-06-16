# Glia v2 Run Summary — China/HK Region

Stop reason: TERMINATE (Evaluator)
Turns completed: 2
Researchers: 2
Directions explored: 2 (beat-glia2a, fix-short-horizon)
Directions killed: 2 (both killed Turn 2)
Output dir: finetune_china/finetune_expts/glia_run_20260330_china

## Best Result

**Checkpoint:** `finetune_china/checkpoints/run_glia_2a/best`
**Config:** 10 HK tickers, stride=32, AdamW lr=1e-4, wd=0.01, freeze 17/20, cosine+5ep warmup, MSE loss
**Val loss:** 0.012067

### Rolling MAPE (15 windows, honest 7/8 split, eval windows only)

| Horizon | Baseline | Glia2a | Improvement |
|---------|----------|--------|-------------|
| 5d | 2.54% | 2.23% | **+0.31pp** |
| 14d | 4.04% | 3.78% | **+0.25pp** |
| 30d | 6.37% | 5.80% | **+0.58pp** |
| 60d | 11.45% | 8.04% | **+3.41pp** |
| 120d | 17.28% | 10.39% | **+6.89pp** |

### Per-Ticker 120d (eval windows)

| Ticker | Baseline | Glia2a | Delta |
|--------|----------|--------|-------|
| 0388.HK (HKEX) | 5.1% | 5.6% | -0.5pp (slight regression) |
| 2382.HK (Sunny Optical) | 21.7% | 6.9% | **+14.8pp** |
| 1177.HK (Sino Biopharma) | 25.1% | 18.7% | **+6.4pp** |

## Key Insights

1. **Glia2a dominates all horizons.** The previously reported short-horizon regression (5d, 14d) was window-specific noise. Honest multi-window evaluation shows improvement at every horizon from 5d to 120d.

2. **120d improvement is ticker-concentrated.** The massive +6.89pp gain at 120d is driven by 2382.HK (Sunny Optical, -14.8pp MAPE) and 1177.HK (-6.4pp). 0388.HK (HKEX exchange operator) slightly regresses — its dynamics as a monopoly exchange operator don't match any training ticker.

3. **All decay variants destroy 2382.HK.** Horizon-decay at any strength (0.3, 1.0) catastrophically regresses 2382.HK at 120d (14.3% → 27-30%). Sunny Optical's long-range price dynamics depend on full gradient signal at long forecast positions.

4. **Sector-matched tickers don't help overall.** Replacing 3 tickers with eval-sector proxies (CSPC Pharma, AAC Tech, CITIC) fixed short horizons but lost 120d, and honest eval showed the short-horizon "fix" was solving a non-problem.

5. **The valid hyperparameter space is extremely narrow.** Only lr=1e-4, wd=0.01, freeze-17, cosine+warmup, 10 tickers, stride=32 achieves val_loss <0.015 (required for valid predictions). Glia2a sits at this optimum.

## Negative Results (11 experiments total)

| Experiment | Result | Why it failed |
|-----------|--------|---------------|
| Run1 (stride=128) | val_loss 0.021, -4.9pp | Too few windows (230 vs 900) |
| 6 long-history tickers | val_loss 0.036 | Lost recent-regime signal from short-history tickers |
| 17 expanded tickers | val_loss 0.015, -1.7pp at 120d | Data dilution |
| Equal-weight ensemble | No benefit | Same as US finding |
| Horizon-decay 1.0 | 2382.HK 120d: 14.3%→27.4% | Starves long-range gradient |
| Sector-matched tickers | 1177.HK 120d: 17.2%→26.3% | Pharma proxy backfired |
| Horizon-selective routing | -0.13pp vs Glia2a | R2-Sector worse on eval windows |
| Horizon-decay 0.3 | 2382.HK 120d: 14.3%→30.2% | Same mechanism as 1.0, even mild |

## Evaluation Framework Recommendations

- **n=3 eval tickers is the binding constraint.** Statistical power is near-zero for detecting <1pp differences. The +6.89pp at 120d is large enough to be meaningful, but per-ticker attribution shows it's concentrated in 2382.HK.
- **Rolling eval (15+ windows) is essential.** Single-window evaluation has ~1-2pp noise (demonstrated by the short-horizon regression episode).
- **0388.HK may be structurally unfixable.** As a monopoly exchange operator, it has no training-set analog. Improvement would require adding exchange-operator-class tickers to training.
- **Further improvement requires fundamentally different approaches:** more eval tickers, different model architecture, or additional data modalities.

## Post-Termination: Expanded Bootstrap Evaluation (9 tickers)

After the evaluator terminated, we expanded evaluation from n=3 to n=9 tickers (adding 6 training tickers with held-out windows) and added 10,000 bootstrap resamples for confidence intervals.

### Revised Overall MAPE (9 tickers, 15 windows)

| Horizon | Improvement | 95% CI | P(worse) | Significant? |
|---------|-------------|--------|----------|--------------|
| 5d | -0.01pp | [-0.30, +0.34] | 54.7% | NO |
| 14d | +0.13pp | [-0.23, +0.51] | 24.6% | NO |
| 30d | +0.30pp | [-0.39, +1.12] | 23.9% | NO |
| 60d | -0.42pp | [-1.75, +1.02] | 73.3% | NO |
| 120d | +2.34pp | [-0.05, +5.22] | 2.8% | NO (barely) |

### Per-Ticker 120d: Winners and Losers

**Significant winners (insurance/tech/telecom):**
- 2382.HK Sunny Optical: +11.07pp [+8.53, +13.48]
- 1177.HK Sino Biopharma: +4.56pp [+2.80, +6.50]
- 0941.HK China Mobile: +2.88pp [+2.49, +3.29]
- 2318.HK Ping An: +3.65pp [+2.24, +5.18]
- 1299.HK AIA Group: +2.89pp [+2.26, +3.46]

**Significant losers (banking/conglomerate):**
- 0005.HK HSBC: -3.24pp [-3.61, -2.86]
- 0001.HK CKH: -0.59pp [-0.85, -0.34]
- 0388.HK HKEX: -0.61pp [-1.19, -0.02]

### Key Revision

The n=3 eval tickers (0388, 2382, 1177) were accidentally enriched for sectors that benefit from finetuning. With n=9 tickers:
- The +6.89pp at 120d shrinks to +2.34pp and loses significance
- 60d flips from positive to negative
- Short horizons are null effects

**Glia2a is a sector-specific model, not a general HK market improvement.**

## Revised Recommendation

**Sector-based routing:** Use Glia2a for insurance, tech hardware, telecom, and pharma tickers. Use baseline for banking, conglomerate, and exchange-operator tickers. This is more honest than claiming universal improvement and would deliver the real benefits where they exist.
