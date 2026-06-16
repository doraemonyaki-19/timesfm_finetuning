# Findings Board — China/HK Region

## Pre-run baseline
- Eval tickers (held-out): 0388.HK (HKEX), 2382.HK (Sunny Optical), 1177.HK (Sino Biopharma)
- Eval training tickers (diagnostic): 0700.HK, 0941.HK, 2318.HK, 1299.HK, 0005.HK, 0001.HK
- Training: 10 HK tickers (IDX_HSI, 0700, 0941, 1299, 9988, 3690, 9999, 2318, 0005, 1810)
- Best checkpoint: Glia2a (finetune_china/checkpoints/run_glia_2a/best), val_loss=0.012067

Rolling eval (15 windows, 2026-03-30):

| Horizon | Baseline | Glia2a | Delta |
|---------|----------|--------|-------|
| 5d | 2.55% | 2.87% | -0.32pp (WORSE) |
| 14d | 3.57% | 3.95% | -0.38pp (WORSE) |
| 30d | 5.41% | 4.83% | +0.58pp |
| 60d | 7.18% | 6.49% | +0.69pp |
| 120d | 17.16% | 12.15% | **+5.01pp** |

Per-ticker 120d: 0388.HK 4.4%→5.0% (+0.6pp worse), 2382.HK 25.4%→14.3% (-11.1pp), 1177.HK 21.8%→17.2% (-4.6pp)

Key patterns:
- Glia2a REGRESSES at short horizons (5d, 14d) — opposite of US where finetuning helps short too
- Massive gains at 120d driven by 2382.HK (Sunny Optical) improvement
- 0388.HK (HKEX exchange operator) regresses at ALL horizons — structural mismatch
- V2_17tk (expanded 17 tickers) is worse than Glia2a at every horizon. Data dilution confirmed.
- Ensemble (Glia2a + V2_17tk) is worse than Glia2a solo.

Prior experiments (5 total):
- Run1: stride=128 → val_loss 0.020608 → regressed -4.9pp. Deleted.
- run_glia_1a: 6 long-history tickers → val_loss 0.035757. Much worse.
- run_glia_2a: 10tk, stride=32 → val_loss 0.012067 → BEST (-5.01pp at 120d rolling)
- run_glia_v2_1a: 17tk, stride=32 → val_loss 0.014907 → inferior (-3.30pp at 120d)
- Ensemble: no benefit over Glia2a solo

Known dead ends:
- Removing short-history tickers (kills recent-regime signal)
- Expanding to 17+ tickers (dilutes signal)
- Equal-weight ensemble (no benefit)
- stride=128 (too few windows for HK tickers with short history)

Open questions:
- Is 0388.HK structurally unfixable? (exchange operator ≠ stock dynamics)
- Can sector-matched proxies help individual tickers?

## Turn 1
- R1: Horizon-decay 1.0 → modest short-horizon improvement vs Glia2a (5d +0.10pp, 14d +0.22pp), still worse than baseline. Catastrophic at 60d (-1.50pp vs Glia2a) and 120d (-4.26pp vs Glia2a). 2382.HK went from 14.3%→27.4% at 120d. Trade-off unfavorable.
- R2: Sector-matched (CSPC Pharma/AAC Tech/CITIC replacing NetEase/Meituan/Xiaomi) → FIXES short-horizon regression (5d +0.33pp, 14d +0.41pp vs Glia2a), best at 30d (+0.71pp vs BL). But 120d worse (14.91% vs 12.15% Glia2a). 1177.HK regressed to 26.3% (worse than BL). 0388.HK improved to 4.0%.
- Implication: Glia2a vs R2 is a short-vs-long horizon trade-off. Selective per-horizon routing could combine strengths. R2's 1177.HK regression suggests pharma proxy backfired.

## Turn 2
- R1: Horizon-selective routing (honest 7/8 split) → OVERTURNS short-horizon regression. Glia2a dominates ALL horizons on eval windows (avg 6.05% vs BL 8.34%). R2-Sector is WORSE than Glia2a everywhere. March 30 regression was window noise. No routing needed.
- R2: Decay=0.3 (mild Goldilocks) → val_loss 0.012950. Still destroys 120d (15.62% vs 12.15% Glia2a). 2382.HK 120d: 14.3%→30.2%. Even mild decay starves 2382.HK long-range signal. ALL decay values are dead ends for China.
- NEW DEAD ENDS: horizon-selective routing (Glia2a dominates all horizons), all horizon-decay values (0.3, 1.0 — both destroy 2382.HK 120d)
- IMPORTANT: Glia2a's short-horizon regression was noise. True performance: +0.31pp at 5d, +0.25pp at 14d (eval windows). Glia2a improves at ALL horizons.

## Post-Termination: Expanded Bootstrap Evaluation (9 tickers, 15 windows, 10k resamples)
- OVERTURNS "Glia2a improves at all horizons" — that was an n=3 artifact.
- 120d: +2.34pp overall but NOT significant at 95% (CI: -0.05pp to +5.22pp, P(worse)=2.8%)
- 60d: WORSE (-0.42pp, P(worse)=73%). Hidden by n=3 eval which cherry-picked winners.
- 5d/14d: zero effect across 9 tickers.
- Per-ticker 120d: 5 tickers improve (2382, 1177, 0941, 2318, 1299), 3 tickers regress (0388, 0005, 0001).
- SECTOR EFFECT: Winners are insurance/tech/telecom matching training data. Losers are banking/conglomerate.
- Glia2a is a sector-specific model, NOT a general HK market improvement.
- Recommendation: sector-based routing (Glia2a for insurance/tech, baseline for banking/conglomerate).

