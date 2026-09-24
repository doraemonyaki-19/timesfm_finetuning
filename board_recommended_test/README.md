# Board-Recommended Paper Trade Test

**Initiated:** 2026-04-11 (dry-run) · **Live basis:** 2026-05-14 (re-init on real prices)  
**Notional:** $1,000,000  
**Status:** Active (live MTM) · **Last run:** 2026-09-24 (95 trading days held)  
**Net P&L to-date:** **+$31,819 (+3.18%)** (Realized: −$3,869 · Unrealized: +$36,031 · Borrow: −$342). Peak was +$42,204 (+4.22%) on 2026-09-08.  

## Purpose

This directory houses a lifecycle paper-trade simulation tracking the board-approved allocation across TimesFM regional finetuned models. The test validates whether the out-of-sample (OOS) statistical edge observed in bootstrap evaluations translates to durable, positive P&L in a realistic, rule-governed portfolio before live capital deployment.

**Board Governance Lineage:**
- Strategy & Allocation Baseline: [`personas/board_meetings/timesfm_lsf_strategy_apr2026.md`](file:///c:/Users/ylchen/workspace/personas/board_meetings/timesfm_lsf_strategy_apr2026.md)
- 60-Day Lifecycle & MTM Evaluation (2026-09-08): [`personas/board_meetings/timesfm_paper_trade_evaluation_sep2026.md`](file:///c:/Users/ylchen/workspace/personas/board_meetings/timesfm_paper_trade_evaluation_sep2026.md)
- Operational Directives: [`personas/recommendations/timesfm_paper_trade_recommendations_sep2026.md`](file:///c:/Users/ylchen/workspace/personas/recommendations/timesfm_paper_trade_recommendations_sep2026.md)

---

## Board-Adopted Resolutions (September 8, 2026 Meeting)

At the September 8, 2026 Board of Directors meeting, the board unanimously ratified the first completed 60-day lifecycle and adopted the following binding governance resolutions:

1. **Ratification of 60d Expiry Harvest:**
   * Formally approved the completion of the 60-day horizon run.
   * 22 positions expired, locking in **+$25,096 in gross realized gains** (+$21,779 net of period borrow/stops) with a **68.2% directional hit rate** (15W / 7L).
   * Cumulative net portfolio equity reached an all-time high of **$1,042,204 (+4.22%)**.

2. **Europe 60d All-Short Posture & Cell-Level Stop-Loss:**
   * Ratified the EUv2 rebalanced allocation: 6 short positions opened on European large caps (EADSF, ASML, SAP, AZN, SHEL, DEO; 3 flat).
   * **New Governance Rule Adopted:** Implemented a **−7.5% Cell-Level Stop-Loss ($11,250 max drawdown on the $150k Europe 60d allocation)**. If cumulative realized + unrealized losses in the Europe 60d cell reach $11,250, all remaining positions in the cell are immediately flattened to cash until the December 7, 2026 rebalance.

3. **November 10, 2026 Horizon Alignment:**
   * The 120-day unexpired cohort (currently holding **+$46,320 in open profit** across Japan, Europe, and China) is confirmed for scheduled maturity on **2026-11-10**.

4. **Gen 2 Continuous Confidence-Weighting:**
   * Directed research team to prototype continuous entropy- and return-scaled position sizing to replace binary equal-weighted buckets for future model iterations.

---

## Board-Approved Allocation

| Cell | Region | Horizon | Weight | Allocation | Strategy | Model | Dead Zone | Current Status |
|------|--------|---------|:------:|:----------:|:--------:|:-----:|:---------:|:--------------:|
| **EUROPE 120d** | Europe | 120 days | 25% | $250,000 | LSF | EUv2 | 1.0% | Open (+$12,396 unrealized) |
| **JAPAN 120d** | Japan | 120 days | 20% | $200,000 | LSF | hdecay1.0 | 2.0% | Open (+$22,913 unrealized) |
| **EUROPE 60d** | Europe | 60 days | 15% | $150,000 | LSF | EUv2 | 2.0% | Open (6 Short, +$1,282 unreal) |
| **CHINA 120d** | China | 120 days | 10% | $100,000 | LF (Long/Flat) | Glia2a | 2.0% | Open (−$62 unrealized) |
| **JAPAN 60d** | Japan | 60 days | 10% | $100,000 | LSF | hdecay1.0 | 2.0% | Open (5L / 2S, −$809 unreal) |
| **US 60d** | US | 60 days | 10% | $100,000 | LF (Long/Flat) | Run4 | 0.0% | Open (5 Long, +$311 unreal) |
| **Cash** | -- | -- | 10% | $100,000 | -- | -- | -- | Unencumbered Buffer |

*Blocked cells:* **US 120d** (permanently blocked due to 28.9% directional accuracy).

---

## Governance Rules & Risk Limits

| Rule | Threshold / Mechanism | Enforcement |
|:---|:---|:---|
| **Single-Name Short Cap** | Max 3.0% of notional ($30,000 max per position) | Enforced at position opening |
| **Position Stop-Loss** | **−10.0%** on individual short positions | Automatic market close on daily run |
| **Cell-Level Stop-Loss (NEW)** | **−7.5% of cell capital ($11,250 on Europe 60d)** | Automatic cell liquidation to cash (`CELL_STOP_LOSS_PCT` in `paper_trade.py`, since 2026-09-24). Counts the current cohort's realized + unrealized P&L net of borrow. Closes are journaled as `CELL-STOP` and the cell stays flat until `next_rebalance`. |
| **Max Portfolio Net Short** | Max 40.0% of total portfolio notional | Forecast/rebalance rejection |
| **Borrow Financing Cost** | **1.5% annualized** accrued daily on all short notional | Deducted from portfolio cash/equity |
| **Dead Zone Filter** | Per-cell optimal DZ from config (if \|pred\| < DZ → FLAT) | Prevents low-conviction noise trades |
| **No-Short Blocklist** | Sovereign/policy-adjacent: `0388.HK`, `0005.HK`, `0941.HK` | Reverts to Long/Flat even in LSF cells |
| **VIX Scaling Breakers** | VIX > 25 → 50% short cut; VIX > 35 → 100% short liquidation | Real-time volatility circuit breaker |

---

## Tickers Traded

### Europe (9 tickers)
* **Eval (held-out):** UL, EADSF, VWAGY
* **Train (in-sample):** ASML, SAP, NVO, AZN, SHEL, DEO

### Japan (9 tickers)
* **Eval (held-out):** 8035.T, 6902.T, 4661.T
* **Train (in-sample):** 7203.T, 6758.T, 9984.T, 8306.T, 4502.T, 7974.T

### China (9 tickers — Long/Flat Only)
* **Eval (held-out):** 0388.HK, 2382.HK, 1177.HK
* **Train (in-sample):** 0700.HK, 0941.HK, 2318.HK, 1299.HK, 0005.HK, 0001.HK
* *No-short blocklist enforced on:* 0388.HK (HKEX), 0005.HK (HSBC), 0941.HK (China Mobile).

### US (12 tickers — Long/Flat Only)
* **Eval (held-out):** NVDA, UNH, GS, CVX, KO, BA
* **Train (in-sample):** AAPL, MSFT, AMZN, GOOGL, META, JPM
* *US 120d is permanently blocked.*

---

## Current Portfolio Snapshot (as of 2026-09-24)

**Open Positions:** 33 (22 Long, 11 Short) · **Days Held:** 95 trading days (120d cohort) / ~12 (60d cohort)  
**Exposure:** Long: $363,889 (36.4%) · Short: −$194,444 (19.4%) · **Net Long: +$169,444 (+16.9%)**  
**Cash Reserve:** $100,000.00 (10.0%) · **Unrealized P&L:** **+$36,031.12**  
**Governance Check:** **PASS** (all short caps, net short exposure, and blocklists fully compliant).

| Cell | Open Count | Stance | Unrealized P&L | Cell Capital | Performance Driver / Thesis |
|:---|:---:|:---:|---:|---:|:---|
| **JAPAN 120d** | 8 | 6L / 2S | **+$22,913** | $200,000 | 4661.T (+9.2k), 8306.T (+5.7k), 7974.T (+4.8k); 6758.T short −1.6k |
| **EUROPE 120d** | 5 | 4L / 1S | **+$12,396** | $250,000 | SAP (+7.9k), SHEL (+3.6k), AZN short (+3.2k); NVO (−4.6k) |
| **CHINA 120d** | 2 | 2L / 0S | **−$62** | $100,000 | HSBC 0005.HK (+1.5k) offset by AIA 1299.HK (−1.6k) |
| **EUROPE 60d** | 6 | 0L / 6S | **+$1,282** | $150,000 | EADSF (+0.7k), DEO (+0.6k); far from the −$11,250 cell stop |
| **JAPAN 60d** | 7 | 5L / 2S | **−$809** | $100,000 | 9984.T (−0.4k), 4502.T short (−0.4k) |
| **US 60d** | 5 | 5L / 0S | **+$311** | $100,000 | META (+1.7k) vs GS (−0.8k), UNH (−0.5k), JPM (−0.5k) |
| **Total** | **33** | **22L / 11S** | **+$36,031** | **$900,000** | **Net Portfolio Equity: $1,031,819 (+3.18%)** |

---

## Closed Trades Summary (`journal.csv`)

Total closed trades to date: **28** (22 Expired, 6 Stopped Out).  
* **Total Gross Realized P&L:** **−$3,869.39** (dramatic recovery from −$25,648 prior to 60d expirations).  
* **Accrued Borrow Costs (Closed):** **$262.33**  
* **Net Closed Realized P&L:** **−$4,131.72**  

### Per-Cell Realized Track Record:
| Cell | Trades Closed | Win / Loss | Gross Realized P&L | Borrow Cost | Net Realized P&L |
|:---|:---:|:---:|---:|---:|---:|
| **EUROPE 120d** | 3 | 0W / 3L | −$11,698.17 | $179.22 | −$11,877.39 |
| **EUROPE 60d** | 6 | 5W / 1L | **+$8,487.54** | $15.07 | **+$8,472.47** |
| **JAPAN 120d** | 1 | 0W / 1L | −$9,357.64 | $20.09 | −$9,377.73 |
| **JAPAN 60d** | 6 | 4W / 2L | **+$4,695.19** | $47.95 | **+$4,647.24** |
| **US 60d** | 12 | 6W / 6L | **+$4,003.69** | $0.00 | **+$4,003.69** |
| **Total** | **28** | **15W / 13L** | **−$3,869.39** | **$262.33** | **−$4,131.72** |

### Stopped-Out Shorts to Date (6 trades, −$25,648 cumulative loss):
* `2026-06-15`: **ASML** (Europe 120d) −$5,402.13 | **8035.T** (Japan 120d) −$9,357.64
* `2026-06-15`: **ASML** (Europe 60d) −$3,241.28 | **8035.T** (Japan 60d) −$4,678.82
* `2026-07-27`: **EADSF** (Europe 120d) −$3,327.97
* `2026-09-08`: **DEO** (Europe 120d) −$2,968.07

---

## Rebalance Schedule

Positions are horizon-aligned. Each cell closes and rebalances when its batch reaches its trading day horizon:

| Cell | Horizon | Current Cohort Status | Next Rebalance Date | Action at Rebalance |
|:---|:---:|:---:|:---:|:---|
| **EUROPE 120d** | 120d | 95d held (+12.4k open) | **2026-11-10** | Close 5 positions, re-run EUv2 model |
| **JAPAN 120d** | 120d | 95d held (+22.9k open) | **2026-11-10** | Close 8 positions, re-run hdecay1.0 model |
| **CHINA 120d** | 120d | 95d held (−62 open) | **2026-11-10** | Close 2 positions, re-run Glia2a router |
| **EUROPE 60d** | 60d | ~12d held (+1.3k open) | **2026-12-07** | Close 6 positions, re-run EUv2 model |
| **JAPAN 60d** | 60d | ~12d held (−0.8k open) | **2026-12-07** | Close 7 positions, re-run hdecay1.0 model |
| **US 60d** | 60d | ~12d held (+0.3k open) | **2026-12-07** | Close 5 positions, re-run Run4 model |

---

## MTM Reports

- `mtm_2026-05-14.md` — Live portfolio initiation at real price basis ($1,000,000 notional).
- `mtm_2026-06-15.md` — First live MTM; 4 semiconductor shorts stopped out (−$16,222 / −1.62%).
- `mtm_2026-07-12.md` — Long book recovery; deficit reduced to −$5,844 (−0.58%).
- `mtm_2026-09-08.md` — First 60d lifecycle expiration (+$21.8k realized), DEO stop-loss, Europe 60d rebalanced 100% short, portfolio reaches all-time high **+$42,204 (+4.22%)**.
- `mtm_2026-09-24.md` — Quiet mark with no closes. 120d book gives back $11.1k (NVO long −16.7%), net falls to **+$31,819 (+3.18%)**. Notes that the cell-level stop is not coded and that the board's NVO figure is wrong.

---

## Execution & Operating Guide

All commands run from the project root using the project virtual environment:

```bash
cd timesfm_finetuning

# Daily run (MTM -> enforce stop-losses -> check expiries -> rebalance)
C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe board_recommended_test/paper_trade.py run

# Show current portfolio status & exposure
C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe board_recommended_test/paper_trade.py status

# Show closed-trade journal & per-cell summary
C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe board_recommended_test/paper_trade.py history
```

## Files

| File | Description |
|:---|:---|
| `paper_trade.py` | Full lifecycle simulator (MTM, stop-loss, expiry, model inference, rebalancing) |
| `state.json` | Live portfolio state (open positions, cell metadata, cash, exposure) |
| `journal.csv` | Immutable, append-only log of all closed trades with P&L and reasons |
| `run_mtm.bat` | Scheduled batch runner for automated daily/weekly MTM logging |
| `README.md` | This governance and operational tracking document |
