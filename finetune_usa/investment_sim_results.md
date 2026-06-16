# Investment Simulation Results

Date: 2026-03-01
Simulation period: ~12 months (2025-02-26 to 2026-01-28)
Tickers: AAPL, JNJ, JPM, XOM, PG, CAT, NEE, AMT, WMT, NVDA, UNH, GS, CVX, KO, BA
Benchmark: ^GSPC

## Performance Summary

| Strategy | Total Return | Ann. Return | Max Drawdown | Sharpe |
|----------|-------------|------------|--------------|--------|
| SP500 | +15.5% | +15.4% | -16.3% | 0.63 |
| EqualHold | +29.2% | +29.0% | -12.5% | 1.50 |
| EqualRebal | +28.9% | +28.7% | -12.7% | 1.48 |
| ForecastTop5 | +18.7% | +18.6% | -19.7% | 0.66 |
| ForecastAllPos | +36.5% | +36.4% | -16.8% | 1.23 |
| SelectiveTop5 | +31.0% | +30.9% | -15.9% | 1.22 |

## Notes

- ForecastTop5 / SelectiveTop5: equal weight across top 5 tickers by predicted 1-month return
- ForecastAllPos: weight proportional to predicted positive returns (Run4)
- SelectiveTop5 routing: NVDA/GS/KO¡÷Glia4b, CVX¡÷Glia3a, others¡÷Run4
- Risk-free rate: 4.5% annualized
- No transaction costs, no short selling
