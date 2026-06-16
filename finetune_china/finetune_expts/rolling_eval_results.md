# Rolling Evaluation Results

Date: 2026-03-14
Windows: 15, Horizon: 120d, Max-context: 512

## Summary (mean กำ std across 15 windows)

| Model | Mean MAPE | Std | Min | Max | DirAcc |
|-------|-----------|-----|-----|-----|--------|
| Baseline | 15.96% | กำ2.83% | 11.6% | 20.4% | 51.0% |
| China_Glia2a | 10.20% | กำ1.76% | 6.6% | 12.9% | 51.1% |
| Ensemble | 10.20% | กำ1.76% | 6.6% | 12.9% | 51.1% |

## Improvement vs Baseline

- **China_Glia2a**: +5.75pp กำ 2.58pp
- **Ensemble**: +5.75pp กำ 2.58pp

## Per-Ticker Mean MAPE

| Model | 0388.HK | 2382.HK | 1177.HK |
|-------|-------|-------|-------|
| Baseline | 3.5% | 19.5% | 24.8% |
| China_Glia2a | 4.4% | 10.6% | 15.5% |
| Ensemble | 4.4% | 10.6% | 15.5% |
