# Rolling Evaluation Results

Date: 2026-03-27
Windows: 15, Horizon: 60d, Max-context: 512

## Summary (mean กำ std across 15 windows)

| Model | Mean MAPE | Std | Min | Max | DirAcc |
|-------|-----------|-----|-----|-----|--------|
| Baseline | 8.77% | กำ0.75% | 7.5% | 10.6% | 50.5% |
| v2_1a | 12.56% | กำ0.90% | 11.2% | 14.2% | 49.2% |
| v2_2phase | 11.88% | กำ0.95% | 10.7% | 13.6% | 48.5% |
| Ensemble | 12.21% | กำ0.92% | 10.9% | 13.9% | 48.9% |

## Improvement vs Baseline

- **v2_1a**: -3.79pp กำ 1.13pp
- **v2_2phase**: -3.11pp กำ 1.23pp
- **Ensemble**: -3.44pp กำ 1.17pp

## Per-Ticker Mean MAPE

| Model | 8035.T | 6902.T | 4661.T |
|-------|-------|-------|-------|
| Baseline | 18.6% | 4.1% | 3.6% |
| v2_1a | 26.7% | 4.7% | 6.4% |
| v2_2phase | 26.1% | 4.8% | 4.7% |
| Ensemble | 26.4% | 4.7% | 5.5% |
