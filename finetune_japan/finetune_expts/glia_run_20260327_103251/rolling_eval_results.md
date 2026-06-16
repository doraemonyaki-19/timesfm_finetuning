# Rolling Evaluation Results

Date: 2026-03-29
Windows: 15, Horizon: 7d, Max-context: 512

## Summary (mean กำ std across 15 windows)

| Model | Mean MAPE | Std | Min | Max | DirAcc |
|-------|-----------|-----|-----|-----|--------|
| Baseline | 3.25% | กำ1.09% | 1.7% | 4.9% | 52.1% |
| semi_10tk | 3.60% | กำ1.46% | 1.6% | 6.0% | 53.3% |
| hdecay_1.0 | 2.68% | กำ0.82% | 1.4% | 4.0% | 50.5% |
| Ensemble | 3.04% | กำ1.16% | 1.6% | 4.9% | 54.0% |

## Improvement vs Baseline

- **semi_10tk**: -0.35pp กำ 0.40pp
- **hdecay_1.0**: +0.57pp กำ 0.56pp
- **Ensemble**: +0.21pp กำ 0.18pp

## Per-Ticker Mean MAPE

| Model | 8035.T | 6902.T | 4661.T |
|-------|-------|-------|-------|
| Baseline | 3.8% | 4.2% | 1.7% |
| semi_10tk | 5.0% | 4.0% | 1.8% |
| hdecay_1.0 | 3.2% | 3.1% | 1.8% |
| Ensemble | 3.9% | 3.5% | 1.8% |
