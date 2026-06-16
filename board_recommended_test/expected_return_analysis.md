# Expected Return Analysis: Portfolio vs S&P 500

**Date:** 2026-04-12
**Portfolio:** Board-recommended paper trade ($1M notional)

## OOS Bootstrap Edges (with dead zone enforcement)

| Cell | Weight | Edge/Period | CI Low | CI High | Rebal/yr | Annualized Contribution |
|------|--------|-------------|--------|---------|----------|------------------------|
| EU 120d | 25% | 20.4% | 17.1% | 23.3% | 2x | **10.2%** |
| JP 120d | 20% | 9.0% | 5.3% | 12.9% | 2x | **3.6%** |
| EU 60d | 15% | 7.2% | 5.9% | 8.4% | 4x | **4.3%** |
| CN 120d | 10% | 2.7% | 1.5% | 4.0% | 2x | **0.5%** |
| JP 60d | 10% | 2.2% | 0.9% | 3.6% | 4x | **0.9%** |
| US 60d | 10% | 3.2% | 1.7% | 4.7% | 4x | **1.3%** |
| Cash | 10% | 0% | -- | -- | -- | **0%** |

**Weighted annualized expected return: ~20.8%**
- 95% CI lower bound: ~15.6%
- 95% CI upper bound: ~26.0%

## S&P 500 Benchmark

- 2026 YTD (mid-April): approximately **-4%**
- Long-term annualized (10yr): **~14.2%**
- Wall Street 2026 consensus: **~12%**

## The Honest Comparison

On paper, the portfolio's 20.8% annualized edge looks like it beats the S&P's
long-run 10-14% by a wide margin, even at the CI lower bound (15.6%).

### What makes this number wrong (inversion)

1. **These edges are OOS but not live.** Bootstrap windows look backward. The
   edge is measured on historical data where the model already existed. Markets
   adapt. The moment real capital follows these signals, the edge begins to
   decay.

2. **Concentration risk the S&P doesn't have.** You're trading 39 names across
   4 regions. The S&P is 500. One bad stop-loss cascade (you already lost
   $15,901 on day one from 4 stops) and your realized return diverges sharply
   from the statistical expectation.

3. **The 20.4% EU 120d edge is doing all the work.** Strip out Europe and
   you're at ~6.5% annualized. That's a single-region bet dressed up as a
   diversified portfolio. If European equities decorrelate from the model's
   training distribution, half your expected return evaporates.

4. **Borrow costs and slippage aren't in the edge.** The 1.5% borrow rate on
   shorts, bid-ask spreads on OTC ADRs (VWAGY, EADSF), and FX impact on
   Japan/China positions will all eat into that 20.8%.

5. **The annualization assumes perfect rebalancing.** 60d cells rebalancing
   4x/year assumes you reinvest at the same edge each cycle. Edge decay means
   each subsequent cycle likely earns less.

## Honest Expected Range

| Scenario | Annualized Return | vs S&P (10-14%) |
|----------|-------------------|-----------------|
| Bootstrap point estimate | ~20.8% | +7 to +11pp |
| CI lower (95%) | ~15.6% | +2 to +6pp |
| After friction (borrow, slippage, FX) | ~12-17% | -2 to +7pp |
| If EU edge decays 50% | ~8-12% | Roughly even |

## Decision Criterion

Run the paper trade for 6 months. If the realized Sharpe exceeds 0.8 after
costs, you have something. If it doesn't, the bootstrap told you a flattering
story and the market told you the truth.
