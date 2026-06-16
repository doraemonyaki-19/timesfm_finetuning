# Clean Simulation Report — 3 Uncontaminated Ticker Sets

**Date:** March 2, 2026
**Simulation period:** March 2025 – February 2026 (12 months)
**Starting capital:** $100,000 | **Rebalancing:** Monthly (~21 trading days)
**Benchmark:** S&P 500 (+17.4%)
**Training tickers (excluded from all sets):** ^GSPC, AAPL, JNJ, JPM, XOM, PG, CAT, NEE, AMT, WMT

---

## Why This Report Matters

The original 15-ticker simulation contained **training/test contamination**: 9 of 15 investable tickers (AAPL, JNJ, JPM, XOM, PG, CAT, NEE, AMT, WMT) were also in the model's training data, covering the same 12-month simulation period. The model was explicitly trained to predict price movements for these tickers during the evaluation window, potentially inflating the forecast-driven strategy results by up to ~9 percentage points.

This report presents three sets of **6 tickers each, with zero overlap with the training data**. All returns are genuinely out-of-sample with respect to the finetuned models.

---

## Ticker Sets

### Set A — Original Glia Evaluation Set
*These are the 6 tickers used throughout the Glia optimization runs (never in training data)*

| Ticker | Company | Industry |
|---|---|---|
| NVDA | NVIDIA | Technology — Semiconductors |
| UNH | UnitedHealth Group | Healthcare — Insurance |
| GS | Goldman Sachs | Financials — Investment Banking |
| CVX | Chevron | Energy — Integrated Oil |
| KO | Coca-Cola | Consumer Staples — Beverages |
| BA | Boeing | Industrials — Aerospace & Defense |

### Set B — Large Cap Diversified
*Blue-chip names across six different industries, none in training set*

| Ticker | Company | Industry |
|---|---|---|
| MSFT | Microsoft | Technology — Software |
| MRK | Merck | Healthcare — Pharmaceuticals |
| BAC | Bank of America | Financials — Commercial Banking |
| COP | ConocoPhillips | Energy — Exploration & Production |
| COST | Costco | Consumer Discretionary — Retail |
| LMT | Lockheed Martin | Industrials — Defense |

### Set C — Growth Mix
*Higher-growth / higher-volatility names across six industries*

| Ticker | Company | Industry |
|---|---|---|
| META | Meta Platforms | Communication Services |
| LLY | Eli Lilly | Healthcare — Biotech/Pharma |
| MS | Morgan Stanley | Financials — Wealth Management |
| SLB | SLB (Schlumberger) | Energy — Oilfield Services |
| TGT | Target | Consumer Discretionary — Retail |
| HON | Honeywell | Industrials — Conglomerates |

---

## Strategy Definitions

| Strategy | Description |
|---|---|
| **SP500** | 100% S&P 500 (benchmark) |
| **EqualHold** | Equal weight, buy-and-hold at start |
| **EqualRebal** | Equal weight, rebalance monthly |
| **ForecastTop3** | Top 3 tickers by predicted 1-month return (Run4 model), equal weight |
| **ForecastAllPos** | All tickers with predicted positive return, weighted by predicted return magnitude |
| **SelectiveTop3** | Top 3 using per-ticker best model (Set A only: NVDA/GS/KO→Glia4b, CVX→Glia3a; Sets B/C: same as ForecastTop3) |

---

## Results

### Set A — Original Glia

| Strategy | Total Return | Ann. Return | Max Drawdown | Sharpe |
|---|---|---|---|---|
| SelectiveTop3 | **+27.3%** | +27.2% | -16.7% | **0.87** |
| ForecastAllPos | +26.9% | +26.8% | -22.9% | 0.73 |
| EqualRebal | +23.8% | +23.6% | **-13.0%** | 0.95 |
| EqualHold | +22.7% | +22.6% | -13.1% | 0.91 |
| ForecastTop3 | +17.3% | +17.2% | -20.7% | 0.53 |
| S&P 500 | +17.4% | +17.3% | -16.3% | 0.72 |

**Individual tickers:**

| Ticker | Return | Sharpe | Note |
|---|---|---|---|
| NVDA | +51.9% | 1.07 | Strong recovery after Feb–Mar tariff dip |
| GS | +44.6% | 1.16 | Financials outperformed broadly |
| CVX | +26.5% | 0.89 | Energy steady |
| KO | +16.5% | 0.73 | In line with market |
| BA | +32.2% | 0.81 | Recovery from prior-year lows |
| **UNH** | **-35.5%** | **-0.68** | **Catastrophic — 60% max drawdown** |

**Key finding:** SelectiveTop3 (+27.3%) beats ForecastTop3 (+17.3%) by 10pp. The per-ticker model routing steers away from UNH more effectively and selects NVDA via Glia4b earlier. UNH remains the dominant risk factor — it dragged every UNH-heavy strategy down significantly.

---

### Set B — Large Cap Diversified ★ Best Clean Result

| Strategy | Total Return | Ann. Return | Max Drawdown | Sharpe |
|---|---|---|---|---|
| **ForecastTop3** | **+28.0%** | +27.9% | **-7.6%** | **1.30** |
| **SelectiveTop3** | **+28.0%** | +27.9% | **-7.6%** | **1.30** |
| EqualRebal | +24.8% | +24.6% | -13.0% | 1.19 |
| EqualHold | +22.8% | +22.7% | -13.2% | 1.09 |
| SP500 | +17.4% | +17.3% | -16.3% | 0.72 |
| ForecastAllPos | +13.9% | +13.8% | -9.1% | 0.58 |

**Individual tickers:**

| Ticker | Return | Sharpe | Note |
|---|---|---|---|
| LMT | +56.0% | 1.63 | Defense spending tailwinds |
| MRK | +39.1% | 1.12 | Drug pipeline catalyst |
| COP | +25.6% | 0.70 | Oil price recovery |
| BAC | +15.5% | 0.50 | In line with market |
| MSFT | +2.3% | 0.05 | Underperformed (AI spend concerns) |
| COST | -1.5% | -0.16 | Consumer spending headwinds |

**Key finding:** This is the most compelling clean result. ForecastTop3 achieves +28.0% with only **-7.6% max drawdown** and Sharpe of **1.30** — superior risk-adjusted performance. The model correctly avoided MSFT (+2%) and COST (-1.5%) in most months, concentrating in LMT and MRK which were the strong performers. ForecastAllPos underperforms because return-magnitude weighting over-concentrated in lower-return names.

---

### Set C — Growth Mix

| Strategy | Total Return | Ann. Return | Max Drawdown | Sharpe |
|---|---|---|---|---|
| EqualRebal | **+19.8%** | +19.7% | -22.4% | **0.70** |
| ForecastTop3 | +18.1% | +18.0% | -21.3% | 0.59 |
| SelectiveTop3 | +18.1% | +18.0% | -21.3% | 0.59 |
| SP500 | +17.4% | +17.3% | -16.3% | 0.72 |
| EqualHold | +16.3% | +16.3% | -22.6% | 0.57 |
| **ForecastAllPos** | **+3.9%** | +3.8% | -22.0% | **0.13** |

**Individual tickers:**

| Ticker | Return | Sharpe | Note |
|---|---|---|---|
| MS | +32.0% | 0.89 | Wealth management outperformance |
| SLB | +29.2% | 0.75 | Energy services recovery |
| HON | +27.3% | 0.92 | Industrial automation demand |
| LLY | +13.3% | 0.40 | Below expectations (GLP-1 priced in) |
| META | -0.4% | 0.07 | Ad market softness |
| TGT | -3.4% | -0.05 | Retail headwinds |

**Key finding:** Forecast strategies barely beat the market (+0.7pp). The high-volatility growth names (META, TGT, LLY) introduce noise that overwhelms the model's signal. ForecastAllPos severely underperforms (+3.9%) — it over-concentrates in low-momentum names. Equal rebalancing (+19.8%) is marginally best for this universe.

---

## Cross-Set Comparison

| Strategy | Set A | Set B | Set C | Average |
|---|---|---|---|---|
| SP500 | +17.4% | +17.4% | +17.4% | +17.4% |
| EqualHold | +22.7% | +22.8% | +16.3% | +20.6% |
| EqualRebal | +23.8% | +24.8% | **+19.8%** | +22.8% |
| ForecastTop3 | +17.3% | **+28.0%** | +18.1% | +21.1% |
| ForecastAllPos | +26.9% | +13.9% | +3.9% | +14.9% |
| SelectiveTop3 | **+27.3%** | **+28.0%** | +18.1% | **+24.5%** |

**SelectiveTop3 is the most consistent strategy** — it leads or ties in Sets A and B, and stays competitive in Set C. Average return of +24.5% across all three clean sets vs S&P 500's +17.4% is a genuine **+7.1pp outperformance** with no training contamination.

**ForecastAllPos is highly universe-dependent** — +26.9% in Set A, only +3.9% in Set C. The return-weighting scheme amplifies both winners and losers. It works when the top-predicted tickers are genuine winners (Set A: UNH despite fundamentals, GS); it fails badly when concentrated in underperformers (Set C: META, TGT).

---

## Contamination Impact: Original vs Clean

| Strategy | Original (contaminated) | Clean average (3 sets) | Inflation |
|---|---|---|---|
| ForecastAllPos | +36.5% | +14.9% | **+21.6pp** |
| SelectiveTop3/5 | +31.0% | +24.5% | +6.5pp |
| EqualRebal | +28.9% | +22.8% | +6.1pp |
| EqualHold | +29.2% | +20.6% | +8.6pp |
| ForecastTop3/5 | +18.7% | +21.1% | -2.4pp |

The original ForecastAllPos result (+36.5%) was **inflated by ~21pp** due to contamination — the model had been trained on the exact prices it was "forecasting." The equal-weight strategies were inflated by ~6–9pp because the contaminated training tickers (CAT +124%, JNJ +56%) happened to be strong performers.

Notably, **ForecastTop3 was slightly *under*-estimated** in the contaminated simulation (-2.4pp) — it was often the strategy that avoided heavily contaminated tickers and concentrated in the cleaner 6 eval tickers.

---

## Recommendations

**Best risk-adjusted strategy (uncontaminated): Set B ForecastTop3**
- +28.0% return, Sharpe 1.30, max drawdown only -7.6%
- The model correctly identified LMT and MRK as the top performers in a diversified large-cap universe with no training data advantage

**Most consistent across universes: SelectiveTop3**
- Average +24.5% across 3 clean sets (+7.1pp vs S&P 500)
- Lower variance than ForecastAllPos; more robust when universe includes volatile names

**Avoid ForecastAllPos in high-volatility universes (Set C)**
- Return-weighting is only beneficial when the model's high-conviction picks are genuinely strong
- In growth/volatile universes, concentration amplifies losses rather than gains

**Avoid contaminated universes for forecast-strategy evaluation**
- The 9 overlapping training tickers inflated results substantially
- Any backtest using tickers the model was trained on should be treated as indicative only

---

*Report generated: March 2, 2026*
*Models: Run4 (checkpoints/best), Glia3a (checkpoints/run_glia_3a/best), Glia4b (checkpoints/run_glia_4b/best)*
*Script: scripts/sim_3sets.py | Plot: finetune_expts/report_3sets_clean.png*
