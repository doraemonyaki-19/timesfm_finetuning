# Rolling Evaluation Results

Date: 2026-03-25
Windows: 15, Horizon: 120d, Max-context: 512

## Summary (mean กำ std across 15 windows)

| Model | Mean MAPE | Std | Min | Max | DirAcc |
|-------|-----------|-----|-----|-----|--------|
| Baseline | 18.10% | กำ1.15% | 15.9% | 19.6% | 50.9% |
| Japan_v2_18tk | 19.59% | กำ0.83% | 18.2% | 20.9% | 50.2% |
| Ensemble | 19.59% | กำ0.83% | 18.2% | 20.9% | 50.2% |

## Improvement vs Baseline

- **Japan_v2_18tk**: -1.49pp กำ 0.95pp
- **Ensemble**: -1.49pp กำ 0.95pp

## Per-Ticker Mean MAPE

| Model | 8035.T | 6902.T | 4661.T |
|-------|-------|-------|-------|
| Baseline | 33.2% | 3.1% | 18.0% |
| Japan_v2_18tk | 38.1% | 4.3% | 16.4% |
| Ensemble | 38.1% | 4.3% | 16.4% |
