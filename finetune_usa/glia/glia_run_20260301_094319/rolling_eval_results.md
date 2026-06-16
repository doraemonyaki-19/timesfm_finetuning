# Rolling Evaluation Results

Date: 2026-03-01
Windows: 20, Horizon: 120d, Max-context: 512

## Summary (mean กำ std across 20 windows)

| Model | Mean MAPE | Std | Min | Max | DirAcc |
|-------|-----------|-----|-----|-----|--------|
| Baseline | 8.08% | กำ0.83% | 7.0% | 10.2% | 50.9% |
| Run4 | 7.40% | กำ0.67% | 6.6% | 8.7% | 51.1% |
| Glia3a | 8.27% | กำ1.00% | 6.9% | 10.9% | 50.7% |
| Glia4b | 7.63% | กำ0.89% | 6.5% | 9.6% | 51.1% |
| Ensemble | 7.68% | กำ0.83% | 6.6% | 9.6% | 50.9% |

## Improvement vs Baseline

- **Run4**: +0.68pp กำ 0.38pp
- **Glia3a**: -0.20pp กำ 0.61pp
- **Glia4b**: +0.45pp กำ 0.55pp
- **Ensemble**: +0.39pp กำ 0.46pp

## Per-Ticker Mean MAPE

| Model | NVDA | UNH | GS | CVX | KO | BA |
|-------|-------|-------|-------|-------|-------|-------|
| Baseline | 7.3% | 11.0% | 9.8% | 7.2% | 5.0% | 8.1% |
| Run4 | 5.6% | 9.7% | 9.5% | 5.7% | 4.7% | 9.1% |
| Glia3a | 8.1% | 10.3% | 9.1% | 4.7% | 5.9% | 11.5% |
| Glia4b | 5.3% | 11.8% | 8.9% | 5.1% | 4.7% | 9.9% |
| Ensemble | 6.2% | 10.4% | 9.2% | 5.2% | 5.1% | 10.1% |
