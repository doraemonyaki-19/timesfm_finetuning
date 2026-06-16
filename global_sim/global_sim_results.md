# Global Multi-Region Portfolio Simulation

Date: 2026-03-14
Simulation period: 2025-03-12 to 2026-02-11 (12 rebalance dates)
Tickers: NVDA, UNH, GS, CVX, KO, BA, 0388.HK, 2382.HK, 1177.HK, UL, EADSF, VWAGY, 8035.T, 6902.T, 4661.T (15 total)
Regions: USA, China, Europe, Japan
Benchmark: ^GSPC

## Performance Summary

| Strategy | Total Return | Ann. Return | Max Drawdown | Sharpe | Volatility | Final Value |
|----------|-------------|------------|--------------|--------|------------|-------------|
| SP500 | +18.4% | +17.7% | -13.7% | 0.76 | 17.9% | $119,026 |
| GlobalEqual | +18.8% | +18.0% | -14.7% | 0.88 | 15.2% | $118,765 |
| GlobalRebal | +19.6% | +18.8% | -14.7% | 0.95 | 14.8% | $119,633 |
| RegionEqual | +17.6% | +16.9% | -15.6% | 0.82 | 15.3% | $117,306 |
| ForecastGlobal5 | +41.2% | +39.4% | -14.2% | 1.57 | 19.6% | $143,337 |
| ForecastRegion1 | +27.5% | +26.4% | -17.2% | 0.98 | 22.1% | $128,076 |
| RegionRotation2 | +21.9% | +21.0% | -14.3% | 0.95 | 17.2% | $122,337 |

## RegionRotation2 ¡X Per-Region Allocation History

| Date | USA | China | Europe | Japan |
|------|------|------|------|------|
| 2025-03-12 | **60%** | 0% | 0% | 40% |
| 2025-04-10 | 40% | 0% | **60%** | 0% |
| 2025-05-12 | **60%** | 40% | 0% | 0% |
| 2025-06-11 | **60%** | 0% | 0% | 40% |
| 2025-07-14 | **60%** | 0% | 0% | 40% |
| 2025-08-12 | **60%** | 0% | 0% | 40% |
| 2025-09-11 | 40% | 0% | 0% | **60%** |
| 2025-10-10 | **60%** | 40% | 0% | 0% |
| 2025-11-10 | **60%** | 0% | 0% | 40% |
| 2025-12-10 | 40% | 0% | **60%** | 0% |
| 2026-01-12 | **60%** | 40% | 0% | 0% |
| 2026-02-11 | **60%** | 40% | 0% | 0% |

### Region Rotation Frequency (top-2)

- **USA**: 12/12 (100%)
- **China**: 4/12 (33%)
- **Europe**: 2/12 (17%)
- **Japan**: 6/12 (50%)

## ForecastGlobal5 ¡X Monthly Top-5 Selections

| Date | Tickers |
|------|---------|
| 2025-03-12 | 1177.HK, 8035.T, BA, NVDA, UNH |
| 2025-04-10 | 2382.HK, 6902.T, BA, EADSF, KO |
| 2025-05-12 | 1177.HK, 2382.HK, CVX, GS, UNH |
| 2025-06-11 | 6902.T, CVX, KO, NVDA, UNH |
| 2025-07-14 | 6902.T, BA, GS, KO, UNH |
| 2025-08-12 | GS, KO, NVDA, UNH, VWAGY |
| 2025-09-11 | 0388.HK, 6902.T, 8035.T, NVDA, UNH |
| 2025-10-10 | 0388.HK, 6902.T, GS, NVDA, UNH |
| 2025-11-10 | 0388.HK, 4661.T, 6902.T, GS, UNH |
| 2025-12-10 | 1177.HK, EADSF, GS, UL, UNH |
| 2026-01-12 | 0388.HK, 1177.HK, BA, GS, NVDA |
| 2026-02-11 | 1177.HK, CVX, GS, KO, UNH |

## ForecastRegion1 ¡X Monthly Top-1 Per Region

| Date | Tickers |
|------|---------|
| 2025-03-12 | USA:BA, China:1177.HK, Europe:VWAGY, Japan:8035.T |
| 2025-04-10 | USA:BA, China:2382.HK, Europe:EADSF, Japan:6902.T |
| 2025-05-12 | USA:UNH, China:1177.HK, Europe:VWAGY, Japan:6902.T |
| 2025-06-11 | USA:UNH, China:1177.HK, Europe:VWAGY, Japan:6902.T |
| 2025-07-14 | USA:UNH, China:0388.HK, Europe:VWAGY, Japan:6902.T |
| 2025-08-12 | USA:UNH, China:0388.HK, Europe:VWAGY, Japan:8035.T |
| 2025-09-11 | USA:NVDA, China:0388.HK, Europe:VWAGY, Japan:8035.T |
| 2025-10-10 | USA:GS, China:0388.HK, Europe:UL, Japan:6902.T |
| 2025-11-10 | USA:GS, China:0388.HK, Europe:UL, Japan:4661.T |
| 2025-12-10 | USA:GS, China:1177.HK, Europe:UL, Japan:6902.T |
| 2026-01-12 | USA:NVDA, China:1177.HK, Europe:UL, Japan:6902.T |
| 2026-02-11 | USA:UNH, China:1177.HK, Europe:EADSF, Japan:6902.T |

## Notes

- Multi-currency: works in return space (daily % change), equivalent to a currency-hedged allocation.
- Forecast horizon: 21 trading days (~1 month).
- Risk-free rate: 4.5% annualized.
- No transaction costs, no short selling.
- NaN returns treated as 0 (market closed).
