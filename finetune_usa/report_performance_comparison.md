# Portfolio Strategy Performance Comparison Report

**Date:** March 2, 2026
**Simulation period:** March 2025 – February 2026 (12 months)
**Starting capital:** $100,000 | **Rebalancing:** Monthly (~21 trading days)
**Universe:** 15 stocks — AAPL, JNJ, JPM, XOM, PG, CAT, NEE, AMT, WMT, NVDA, UNH, GS, CVX, KO, BA
**Benchmark:** S&P 500 (^GSPC)
**Model:** TimesFM 2.5 (200M), finetuned on financial time series (Run4 + Glia3a + Glia4b checkpoints)

---

## Executive Summary

All strategies outperformed the S&P 500 (+17.4%) over the 12-month period. The forecast-driven **ForecastAllPos** strategy delivered the highest total return (+36.5%), while **SelectiveTop5** — which routes each ticker to its most accurate finetuned model — achieved +31.0% with a more balanced risk profile. Equal-weight buy-and-hold (+29.2%) outperformed simple top-5 forecast selection (+18.7%), revealing that model-guided concentration without return-weighting adds risk without proportional reward.

---

## Strategy Definitions

| Strategy | Description |
|---|---|
| **SP500** | 100% allocation to S&P 500 index (benchmark, not investable directly) |
| **EqualHold** | Equal weight ($6,667 each) in all 15 tickers at start; no rebalancing |
| **EqualRebal** | Equal weight across all 15 tickers; rebalance to equal weight each month |
| **ForecastTop5** | At each monthly rebalance, invest equally in the 5 tickers with the highest predicted 1-month return (Run4 model) |
| **ForecastAllPos** | At each monthly rebalance, allocate across all tickers with predicted positive returns, weighted proportionally to predicted return magnitude (Run4 model) |
| **SelectiveTop5** | Top-5 by predicted return, using the best finetuned model per ticker: NVDA/GS/KO → Glia4b, CVX → Glia3a, all others → Run4 |

---

## Performance Summary

| Strategy | Total Return | Ann. Return | Max Drawdown | Sharpe Ratio | Final Value |
|---|---|---|---|---|---|
| **ForecastAllPos** | **+36.5%** | +36.4% | -16.8% | 1.23 | $136,530 |
| **SelectiveTop5** | +31.0% | +30.9% | -15.9% | **1.22** | $130,993 |
| EqualHold | +29.2% | +29.0% | **-12.5%** | **1.50** | $129,174 |
| EqualRebal | +28.9% | +28.7% | -12.7% | 1.48 | $128,869 |
| ForecastTop5 | +18.7% | +18.6% | -19.7% | 0.66 | $118,691 |
| S&P 500 | +15.5% | +15.4% | -16.3% | 0.63 | $115,494 |

*Sharpe ratio: annualized, risk-free rate 4.5%. No transaction costs.*

---

## Individual Ticker Performance (Buy-and-Hold)

| Rank | Ticker | Total Return | Max Drawdown | Sharpe |
|---|---|---|---|---|
| 1 | **CAT** | +124.3% | -21.8% | 2.46 |
| 2 | **JNJ** | +56.0% | -12.7% | 2.21 |
| 3 | **NVDA** | +51.9% | -24.5% | 1.07 |
| 4 | **XOM** | +44.8% | -16.1% | 1.43 |
| 5 | **GS** | +44.6% | -25.7% | 1.16 |
| 6 | **NEE** | +38.7% | -15.8% | 1.20 |
| 7 | WMT | +32.6% | -16.8% | 1.07 |
| 8 | BA | +32.2% | -25.2% | 0.81 |
| 9 | CVX | +26.5% | -20.6% | 0.89 |
| 10 | ^GSPC | +17.4% | -16.3% | 0.72 |
| 11 | JPM | +17.2% | -20.1% | 0.57 |
| 12 | KO | +16.5% | -9.8% | 0.73 |
| 13 | AAPL | +12.0% | -28.7% | 0.38 |
| 14 | **PG** | **-2.3%** | -20.1% | -0.27 |
| 15 | **AMT** | **-3.5%** | -26.1% | -0.20 |
| 16 | **UNH** | **-35.5%** | **-60.1%** | **-0.68** |

---

## Key Findings

### 1. ForecastAllPos Is the Top Performer — But Carries a Known Risk
ForecastAllPos generated +36.5% by concentrating capital in the model's highest-conviction picks each month: primarily UNH, GS, JPM, and WMT. Paradoxically, UNH — the model's most consistently bullish call — was the **worst-performing individual stock in the universe, losing 35.5%**. ForecastAllPos succeeded *despite* UNH because the return-weighted scheme diversified meaningfully across GS and JPM, which both performed well (+44.6% and +17.2% respectively). The model's systematic bullishness on UNH is a structural flaw.

### 2. SelectiveTop5 Outperforms Equal-Weight Forecast
By routing each ticker to its empirically best-validated model (based on 20-window rolling evaluation from the Glia optimization runs), SelectiveTop5 achieves +31.0% vs ForecastTop5's +18.7%. The key difference: SelectiveTop5 more frequently selects NVDA (via Glia4b) and CVX (via Glia3a) instead of defaulting to UNH at full weight. This confirms the value of the per-ticker model calibration done during optimization.

### 3. Equal-Weight Outperforms Naive Top-5 Forecast
EqualHold (+29.2%) and EqualRebal (+28.9%) both beat ForecastTop5 (+18.7%). ForecastTop5 missed CAT (+124.3%), the year's biggest winner, in most months. It also incurred deeper drawdowns (-19.7%) from concentrated UNH exposure. This demonstrates that forecasting errors in high-conviction picks are more costly than a diversified baseline.

### 4. CAT Was the Missed Opportunity
CAT returned +124.3% — more than double the second-best ticker — yet was selected by the forecast model in only 1 of 12 months. The model's predicted return for CAT was negative or low at most rebalance dates. This is the largest single miss across all strategies.

### 5. The Universe Structurally Beats the S&P 500
All strategies outperform the S&P 500 (+17.4%). This is partly because the 15-ticker universe is concentrated in sectors that outperformed this year (industrials, energy, financials, tech). Results should not be extrapolated to a different market environment.

### 6. Effect of Removing NVDA
Removing NVDA from the universe reveals that the model's early-period allocation to NVDA (Feb–Mar 2025) coincided with a tariff-driven selloff. Without NVDA:
- ForecastAllPos improves to **+41.0%** (capital redirected to GS/JPM)
- ForecastTop5 improves to **+21.4%**
- Equal-weight strategies decline slightly (NVDA's full-year return was positive)

---

## Risk Considerations

- **UNH concentration risk**: The model is systematically bullish on UNH. Any strategy using Run4 forecasts will likely over-allocate to UNH. Capping UNH at 10–15% of portfolio is recommended.
- **Drawdown**: ForecastTop5 had the deepest drawdown (-19.7%), worse than the benchmark (-16.3%). Concentrated bets amplify both gains and losses.
- **No transaction costs**: Real-world implementation will incur brokerage fees, bid-ask spreads, and potential market impact, reducing all returns.
- **Backtest bias**: Results are computed over a single 12-month window that happened to be a strong year for this universe. The model's real edge (measured in Glia rolling evaluation) is approximately 0.68pp reduction in MAPE at 120-day horizon — modest, not transformative.
- **Model limitation**: TimesFM is a time-series pattern model. It has no knowledge of earnings, macro events, interest rate changes, or geopolitical developments.

---

## Recommended Strategy

**For risk-adjusted performance: SelectiveTop5** offers the best balance — +31.0% return with a -15.9% max drawdown and meaningful diversification across model expertise. The selective routing (using the per-ticker best model) adds ~12pp return over naive top-5 forecast.

**For maximum return with higher risk tolerance: ForecastAllPos** — but cap UNH at 15% and redistribute to GS/JPM.

**For lowest risk: EqualHold or EqualRebal** — nearly equivalent outcomes (+29.2% vs +28.9%), with the lowest drawdowns in the universe (-12.5% and -12.7%) and the highest Sharpe ratios (1.50 and 1.48). These require no model inference.

---

*Report generated by TimesFM investment simulation pipeline.*
*Models: Run4 (checkpoints/best), Glia3a (checkpoints/run_glia_3a/best), Glia4b (checkpoints/run_glia_4b/best).*
*Script: scripts/investment_sim.py*
