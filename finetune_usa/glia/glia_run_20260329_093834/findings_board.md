# Findings Board

## Pre-run baseline (US region)
- Baseline (pretrained): 8.08% rolling MAPE at 120d (20 windows)
- Best checkpoint Run4 (finetune_usa/checkpoints/best): 7.40% ± 0.67% at 120d (+0.68pp)
- Glia3a: 8.27% (worse than baseline on average — NVDA/BA catastrophe)
- Glia4b: 7.63% (+0.45pp)
- Equal-weight ensemble: 7.68% (worse than Run4 solo)
- Selective ensemble: 7.65% (+0.28pp vs Run4, but +43% variance)
- Ceiling after 22 experiments: ~0.68pp at 120d. Loss function, layer selection, data, context, scheduler all exhausted.
- Japan discovery: horizon-weighted loss (decay=1.0) gave +0.99pp at 14d and +2.20pp at 120d. Untested on US.

## Turn 1
- R1: Horizon-decay 1.0 on US → short-horizon gains (5d +0.21pp, 14d +0.21pp vs Run4) but **catastrophic at 120d** (10.93%, -2.47pp vs Run4, -1.14pp vs baseline). NVDA 11.5% at 120d. Japan pattern does NOT transfer to US.
- R2: Sector-matched tickers (AVGO/CI/HON replacing NEE/AMT/CAT) → **NEW BEST at 120d: 7.88%** (+0.58pp vs Run4 8.46%, +1.92pp vs baseline 9.80%). UNH massive drop 13.0%→8.6% from CI inclusion. Neutral at short horizons.
- R2 val_loss=0.01429, R1 val_loss=0.01306. Lower val_loss (R1) produced worse MAPE — confirms val_loss ≠ MAPE.
- Sector matching is the most effective lever found so far for US 120d.
- **CAUTION:** T1-R2 120d shifted from 7.88% to 9.20% next day (std 1.33%). Evaluation instability at 120d.

## Turn 2
- R1: Sector-matched (AVGO/CI/HON) + mild decay 0.3 → val_loss=0.007696 (epoch 44). 14d: 4.11%, 30d: 5.97%, 60d: 9.43%, 120d: 10.68%. Worse than T1-R1 at short horizons, worse than Run4 at long. Decay 0.3 doesn't help enough.
- R2: AMD/CI/CAT (replace AVGO→AMD, HON→CAT) → val_loss=0.014441 (epoch 25). ALL horizons worse: 120d 11.16% (-1.36pp vs baseline). AMD is too volatile for training.
- **Best across all experiments by horizon:** 5d=T1-R2 (2.27%), 14d=T1-R1 (4.05%), 30d=T1-R1 (5.62%), 60d=Run4 (8.77%), 120d=Run4 (8.46%).
- Horizon-decay helps ≤30d but hurts ≥60d on US. No single checkpoint beats Run4 at all horizons.
- Turn 2 produced no improvements. AMD is a dead end. Mild decay doesn't bridge the short/long gap.

## Evaluator (Turn 2)
- [KILL: horizon-weighted-loss-us] — 3 experiments, definitively falsified for US. No sweet spot.
- [KILL: combined-sector-decay] — Destructive interaction confirmed.
- [REDIRECT: selective-ensemble-us-run4-vs-t1r2] — Per-ticker routing between Run4 and T1-R2 (zero-cost, no training).
- Directive: Validate T1-R2 robustness first. If 120d improvement <0.3pp vs Run4, kill sector matching too.
- No new training this cycle — evaluation only.

## Turn 3 — Validation Eval (orchestrator)
- Re-ran T1-R1, T1-R2, Run4 at all horizons [5, 14, 30, 60, 120] with 15 windows. Results stable (T1-R2 120d=9.20% same as yesterday).
- **Best by horizon:** 5d=T1-R2 (2.27%), 14d=T1-R1 (4.05%), 30d=T1-R1 (5.62%), 60d=Run4 (8.77%), 120d=Run4 (8.46%)
- T1-R2 does NOT beat Run4 at 120d (9.20% vs 8.46% = -0.74pp). Original 7.88% was a fluke.
- **Clear horizon split**: T1-R1 (horizon-decay) wins ≤30d, Run4 wins ≥60d. T1-R2 only wins at 5d.
- 3-model equal-weight ensemble underperforms best individual at every horizon.
- Per-ticker 120d: Run4 wins on UNH (13.0% vs 15.0%), GS (5.9% vs 7.4%), CVX (9.3% vs 11.2%). T1-R2 wins NVDA (5.4% vs 5.8%), KO (9.7% vs 10.0%), BA (6.6% vs 6.8%).
- **Selective ensemble opportunity**: Route by horizon — T1-R1 for ≤30d forecasts, Run4 for ≥60d. Or by ticker at each horizon.

## Turn 3 — Horizon-Selective Ensemble (R1)
- Honest evaluation (7 selection + 8 eval windows): Run4 selected as best at ALL 5 horizons. Selective ensemble degenerates to Run4-only.
- T1-R1's short-horizon advantage was not robust — driven by later windows (7-14), not selection windows (0-6).
- All-window comparison: Run4 wins 14d (2.18%), 30d (3.58%), 60d (5.04%), 120d (8.57%). Baseline wins 5d (1.61%).
- **Conclusion: No checkpoint from this Glia run improves on Run4 in an honest out-of-sample test.**
- Run4 remains the best single checkpoint across all horizons and all honest evaluation protocols.

