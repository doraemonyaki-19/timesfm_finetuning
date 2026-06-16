# Glia v2 Run Summary — US Region

Stop reason: TERMINATE (Evaluator)
Turns completed: 3
Researchers: 2
Directions explored: 5 (horizon-decay, sector-matching, combined, AMD-substitution, selective-ensemble)
Directions killed: 5 (all)
Output dir: finetune_usa/glia_run_20260329_093834

## Best Result

**Run4 remains the best US checkpoint.** No improvement found.

- Checkpoint: `finetune_usa/checkpoints/best`
- Rolling MAPE at 120d: 7.40% +/- 0.67% (15-20 windows)
- Improvement vs baseline: +0.68pp (8.08% baseline)
- Recipe: AdamW lr=1e-4, wd=0.01, freeze 17/20 layers, cosine+5ep warmup, 10 S&P500 tickers, MSE loss, context=512

## Key Insights

1. **Japan→US transfer fails for loss functions.** Horizon-weighted decay (decay=1.0) that gave +2.20pp in Japan was catastrophic for US at 120d (-1.14pp vs baseline). US equities have stronger long-range dependencies that require full gradient signal at all positions.

2. **Sector matching helps individual tickers but not the aggregate.** CI (health insurance) in training massively improved UNH forecasts (13.0%→8.6% at 120d), but the improvement was unstable day-to-day and didn't survive honest selection/eval splits. GS actually regressed when its presumed sector match (HON→aerospace) was introduced.

3. **Evaluation noise is the binding constraint at this scale.** Single-window 120d MAPE fluctuates ~1.3pp. T1-R2's "breakthrough" (7.88%) reverted to 9.20% the next day. Only improvements >1.5pp can be reliably detected with n=6 eval tickers and 15 windows.

4. **Val loss diverges from MAPE.** T1-R1 had the lowest val loss (0.01306) but the worst 120d MAPE (13.63%). T2-R1 had val_loss=0.007696 (suspiciously low — overfitting to early positions) and poor MAPE. MSE val loss optimizes all positions equally, while MAPE at horizon H only uses positions 0 to H-1.

5. **No ensemble improves on Run4.** Equal-weight ensemble (7.68%) is worse than Run4 (7.40%). Horizon-selective ensemble degenerates to Run4-only in honest evaluation. Per-ticker selective ensemble gives +0.28pp but with +43% variance.

## Experiments Summary

| Experiment | Training Tickers | Loss | 120d MAPE | vs Run4 | Verdict |
|-----------|-----------------|------|-----------|---------|---------|
| T1-R1: Horizon-decay 1.0 | Standard 10 | MSE + decay=1.0 | 13.63% | -5.17pp | FAILED |
| T1-R2: Sector-matched | AVGO/CI/HON replacing NEE/AMT/CAT | MSE | 9.20% | -0.74pp | FAILED |
| T2-R1: Sector + decay 0.3 | AVGO/CI/HON | MSE + decay=0.3 | 10.68% | -2.22pp | FAILED |
| T2-R2: AMD substitution | AMD/CI/CAT replacing AVGO/HON/NEE | MSE | 11.16% | -2.70pp | FAILED |
| T3: Horizon-selective | N/A (eval only) | N/A | Run4-only | 0pp | Run4 dominates |

## Evaluation Framework Recommendations

1. **Increase eval tickers.** n=6 provides near-zero statistical power for improvements <0.5pp. Consider expanding to 10-12 diverse tickers.
2. **Use aligned-context evaluation** (horizon_selective_eval.py approach) for cross-horizon comparisons, not per-horizon rolling_eval.py.
3. **Two-day consistency check.** Any claimed improvement should be verified on consecutive days. T1-R2 would have been caught immediately.
4. **Report median MAPE** alongside mean — robust to single-ticker outliers like UNH dominating the aggregate.

## Remaining Opportunities (Outside Current Framework)

- Different base model (Chronos, Moirai) — different architecture may have different ceiling
- Larger training dataset with careful sector balancing
- More eval tickers for higher statistical power
- Region-specific model selection (Japan: horizon-decay; US: standard MSE)
