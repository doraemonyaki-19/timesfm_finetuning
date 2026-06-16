# Rolling Evaluation Results

Date: 2026-03-31
Windows: 15, Horizon: 120d, Max-context: 512

## Summary (mean กำ std across 15 windows)

| Model | Mean MAPE | Std | Min | Max | DirAcc |
|-------|-----------|-----|-----|-----|--------|
| Baseline | 11.96% | กำ0.68% | 10.7% | 13.6% | 49.2% |
| Glia2a | 9.62% | กำ0.88% | 8.4% | 11.4% | 49.0% |
| Ensemble | 9.62% | กำ0.88% | 8.4% | 11.4% | 49.0% |

## Improvement vs Baseline

- **Glia2a**: +2.34pp กำ 0.52pp
- **Ensemble**: +2.34pp กำ 0.52pp

## Bootstrap Confidence Intervals (10000 resamples)

### Glia2a

- MAPE: 9.63% [95% CI: 7.06% กV 12.50%]
- Improvement vs BL: +2.34pp [95% CI: -0.05pp กV +5.22pp]
- P(worse than baseline): 2.8%
- Significant at 95%: **NO**

### Ensemble

- MAPE: 9.63% [95% CI: 7.06% กV 12.50%]
- Improvement vs BL: +2.34pp [95% CI: -0.05pp กV +5.22pp]
- P(worse than baseline): 2.8%
- Significant at 95%: **NO**


## Per-Ticker Mean MAPE

| Model | 0388.HK | 2382.HK | 1177.HK | 0700.HK | 0941.HK | 2318.HK | 1299.HK | 0005.HK | 0001.HK |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Baseline | 4.4% | 25.4% | 21.8% | 6.9% | 6.7% | 13.3% | 12.6% | 6.5% | 10.2% |
| Glia2a | 5.0% | 14.3% | 17.2% | 6.5% | 3.9% | 9.6% | 9.7% | 9.8% | 10.8% |
| Ensemble | 5.0% | 14.3% | 17.2% | 6.5% | 3.9% | 9.6% | 9.7% | 9.8% | 10.8% |
