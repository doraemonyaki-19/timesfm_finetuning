# Rolling Evaluation Results

Date: 2026-03-14
Windows: 15, Horizon: 120d, Max-context: 512

## Summary (mean กำ std across 15 windows)

| Model | Mean MAPE | Std | Min | Max | DirAcc |
|-------|-----------|-----|-----|-----|--------|
| Baseline | 5.44% | กำ0.59% | 4.7% | 6.6% | 49.0% |
| Europe_Run1 | 5.24% | กำ0.55% | 4.6% | 6.4% | 48.5% |
| Ensemble | 5.24% | กำ0.55% | 4.6% | 6.4% | 48.5% |

## Improvement vs Baseline

- **Europe_Run1**: +0.20pp กำ 0.16pp
- **Ensemble**: +0.20pp กำ 0.16pp

## Per-Ticker Mean MAPE

| Model | UL | EADSF | VWAGY |
|-------|-------|-------|-------|
| Baseline | 4.5% | 6.0% | 5.8% |
| Europe_Run1 | 4.5% | 5.7% | 5.6% |
| Ensemble | 4.5% | 5.7% | 5.6% |
