# 2-Week Investment Plan — $10,000

**Date:** March 3, 2026
**Horizon:** 10 trading days (March 3 – March 16, 2026)
**Model:** TimesFM 2.5, finetuned checkpoints (Run4, Glia3a, Glia4b)

> ⚠ **Disclaimer:** This plan is based on a machine-learning time-series model. It is not professional financial advice. The model has no knowledge of upcoming earnings, macro events, or news. All predicted price levels carry uncertainty. Past simulation performance does not guarantee future results.

---

## Daily Update Procedure

Run this each trading day after market close to pull live prices and regenerate the chart:

```shell
cd /c/Users/ylchen/workspace/timesfm
source .venv/Scripts/activate
python scripts/sim_2week_live.py
```

Outputs:
- **`finetune_expts/sim_2week_live.json`** — raw daily portfolio values, per-ticker breakdown, entry prices
- **`finetune_expts/sim_2week_live.png`** — portfolio value chart (actual vs forecast), per-ticker return bars, running return

After running, copy the per-ticker prices and portfolio total into a new `## March XX Update` section at the top of this file (above the previous day's update).

---

## March 12 Update

*Updated: March 12, 2026 — Day 8 of 10*

### Actual Performance (Full History)

| Day | Date | Portfolio (adj) | Return | Original Forecast | Gap |
|-----|------|-----------------|--------|-------------------|-----|
| 1 | Mar 3 | $10,113.85 | +1.14% | +2.94% | -1.80pp |
| 2 | Mar 4 | $10,162.33 | +1.62% | +3.06% | -1.44pp |
| 3 | Mar 5 | $10,088.09 | +0.88% | +3.19% | -2.31pp |
| 4 | Mar 6 | $9,940.51 | -0.59% | +3.20% | -3.79pp |
| 5 | Mar 9 | $9,990.52 | -0.09% | +3.92% | -4.01pp |
| 6 | Mar 10 | $9,992.95 | -0.07% | +3.70% | -3.77pp |
| 7 | Mar 11 | $10,059.38 | +0.59% | +4.00% | -3.41pp |
| **8** | **Mar 12** | **$10,004.44** | **+0.04%** | +4.32% | **-4.28pp** |

**Broad market context:** A macro shock drove today's session. Oil prices surged toward $100/bbl on fears of Strait of Hormuz closure (escalating Iran conflict). The Dow fell ~700 points. CVX surged on energy exposure; financials (GS, BAC) and UNH sold off. The portfolio held near flat (+0.04%) thanks to CVX strength and the 40% cash buffer.

**GS exit (March 9) confirmed again correct:** GS closed today at $787.52, down $44.51 from our exit price of $832.03 on March 9. Holding GS would have added **-$106.47** in losses (2.392 shares × $44.51).

### Per-Ticker Status (March 12 Close)

| Ticker | Entry | Mar 12 Close | Return | Held Shares | Position Value | vs Forecast |
|--------|-------|--------------|--------|-------------|----------------|-------------|
| **NVDA** | $178.49 | $183.14 | **+2.61%** | 5.605 (half) | $1,027 | -$2.80 |
| **BAC** | $48.76 | $47.13 | **-3.34%** ⚠ | 41.017 | $1,933 | -$1.35 |
| **CVX** | $190.52 | $196.97 | **+3.39%** ✅ | 10.498 | $2,068 | **+$4.15** |
| **UNH** | $288.99 | $277.05 | **-4.13%** ⚠ | 3.460 (half) | $959 | -$10.50 |
| **GS** | $836.00 | $787.52 | — | 0 (exited) | — | — |
| **Cash** | — | — | — | — | $4,018 | — |
| **Total** | | | | | **$10,004** | |

### Actual vs. Forecast (March 12)

| Ticker | Forecast (from Mar 11) | Actual Close | Difference |
|--------|------------------------|--------------|------------|
| NVDA | $185.94 | $183.14 | -$2.80 ❌ |
| BAC | $48.48 | $47.13 | -$1.35 ❌ |
| CVX | $192.82 | $196.97 | **+$4.15** ✅ |
| UNH | $287.55 | $277.05 | -$10.50 ❌ |

CVX again the sole outperformer (+$4.15 vs forecast), driven by the oil price surge. UNH delivered the largest miss (-$10.50), now sitting at its original stop-loss of $277.00.

### Stop-Loss Check (March 12 Close)

| Ticker | Stop Level | Current Price | Buffer | Status |
|--------|------------|---------------|--------|--------|
| CVX | $184.00 (tightened) | $196.97 | +$12.97 | Safe |
| NVDA | $183.00 (watch) | $183.14 | +$0.14 | ⚠ On edge |
| BAC | $46.80 (original) | $47.13 | +$0.33 | ⚠ Tight |
| UNH | $277.00 (original) | $277.05 | +$0.05 | ⚠ Essentially triggered |

### Mar 13 Trading Decision

**All positions: Hold into the close (final 2 days).**

Rationale:
- **UNH**: Stop-loss nominally triggered ($277.05 vs $277.00). However, only 2 trading days remain. Transaction cost and execution risk outweigh the marginal protection from exiting $959 into cash. **Hold — but exit at open if UNH opens below $275.**
- **CVX**: Already past the original Mar 16 target of $191.58. The oil-driven surge may fade as geopolitical news evolves. However, with only 2 days left and strong momentum, holding captures any continuation. **Hold.**
- **NVDA**: Barely above $183.00 watch level. No reason to exit — would only crystallize a small gain vs. avoiding a position with 2 days of potential upside. **Hold.**
- **BAC**: $0.33 above the $46.80 stop. If BAC opens below $46.80 on March 13, exit. Otherwise hold. **Hold unless stop triggered at open.**
- **Cash (40%)**: Unchanged. Provides full floor for any gap-down on final 2 days.

**Watch levels for March 13:**
- UNH below $275.00: exit at open
- BAC below $46.80: exit at open (plan stop-loss)
- NVDA below $181.00: exit trigger
- CVX below $193.00: oil reversal signal; consider tightening

### Updated Portfolio (March 12 Close)

| Position | Shares | Price | Value | % Portfolio |
|----------|--------|-------|-------|-------------|
| NVDA | 5.605 | $183.14 | $1,027 | 10.3% |
| BAC | 41.017 | $47.13 | $1,933 | 19.3% |
| CVX | 10.498 | $196.97 | $2,068 | 20.6% |
| UNH | 3.460 | $277.05 | $959 | 9.6% |
| **Cash** | — | — | **$4,018** | **40.2%** |
| **Total** | | | **$10,004** | **100%** |

### Mar 11 Forecasts for Mar 13–16 (Now Stale)

The March 11 model context produced these forecasts — they are now significantly stale following today's macro shock:

| Date | NVDA | BAC | CVX | UNH | Adj Portfolio | Return |
|------|------|-----|-----|-----|---------------|--------|
| Mar 13 | $184.91 | $48.79 | $191.88 | $288.57 | $10,069 | +0.69% |
| **Mar 16** | **$185.11** | **$48.85** | **$192.40** | **$288.55** | **$10,078** | **+0.78%** |

CVX has already surpassed the Mar 16 forecast ($196.97 vs $192.40). UNH is $11.50 below its Mar 16 forecast. These forecasts should be treated as directional guidance only.

### Final Scenarios (March 16)

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **CVX holds gains, others flat** | **~$10,004** | **+0.04%** |
| Oil retreats, CVX -3%, BAC/UNH recover +1% | ~$9,960 | -0.40% |
| CVX +2% more, others flat | ~$10,025 | +0.25% |
| Bear — UNH/BAC hit stops, CVX flat | ~$9,980 | -0.20% |

*With 40% cash and only 2 days remaining, maximum practical drawdown is approximately -0.5%. Portfolio is essentially at breakeven regardless of equity moves.*

---

## March 11 Update

*Updated: March 11, 2026 — Day 7 of 10*

### Actual Performance (Full History)

| Day | Date | Portfolio (adj) | Return | Original Forecast | Gap |
|-----|------|-----------------|--------|-------------------|-----|
| 1 | Mar 3 | $10,113.85 | +1.14% | +2.94% | -1.80pp |
| 2 | Mar 4 | $10,162.33 | +1.62% | +3.06% | -1.44pp |
| 3 | Mar 5 | $10,088.09 | +0.88% | +3.19% | -2.31pp |
| 4 | Mar 6 | $9,940.51 | -0.59% | +3.20% | -3.79pp |
| 5 | Mar 9 | $9,990.52 | -0.09% | +3.92% | -4.01pp |
| 6 | Mar 10 | $9,992.95 | -0.07% | +3.70% | -3.77pp |
| **7** | **Mar 11** | **$10,059.38** | **+0.59%** | +4.00% | **-3.41pp** |

**Portfolio crossed back above $10,000 for the first time since Day 2.** Actual post-trade portfolio (accounting for GS exit, UNH half, and today's NVDA partial sell): **$10,051.59 (+0.52%)**.

### Trades Executed at March 11 Open

Per the March 10 decision:

| Trade | Shares | Price | Proceeds | Realized Gain |
|-------|--------|-------|----------|---------------|
| Sell 5.60 NVDA | 5.60 | $185.92 (open) | $1,041.15 | **+$41.61** |
| **Cash freed** | | | **$1,041.15** | |

New cash position: $2,977.28 + $1,041.15 = **$4,018.43**

### Per-Ticker Status (March 11 Close)

| Ticker | Entry | Mar 11 Close | Return | Held Shares | Position Value | vs Forecast |
|--------|-------|--------------|--------|-------------|----------------|-------------|
| **NVDA** | $178.49 | $186.03 | **+4.22%** ✅ | 5.605 (half) | $1,043 | +$1.16 |
| **BAC** | $48.76 | $48.52 | -0.49% | 41.017 | $1,990 | -$0.19 |
| **CVX** | $190.52 | $191.79 | **+0.67%** ✅ | 10.498 | $2,013 | **+$3.83** |
| **UNH** | $288.99 | $285.25 | -1.29% | 3.460 (half) | $987 | +$0.20 |
| **GS** | $836.00 | $823.76 | — | 0 (exited) | — | — |
| **Cash** | — | — | — | — | $4,018 | — |
| **Total (post-trade)** | | | | | **$10,052** | |

### Today's Key Development: CVX Surge

CVX was the standout mover of the day. It opened at $186.59 — dangerously close to the tightened stop-loss of $184.00 — then staged a strong intraday rally to close at **$191.79**, its highest close since entry.

- Beat the Mar 11 model forecast ($187.96) by **+$3.83**
- Now **+0.67% above entry** for the first time since March 4
- The tightened stop ($184.00) was never triggered

This is a significant positive surprise. The model was predicting CVX would remain depressed through Mar 16 ($187.54 target) — that forecast is now stale.

**GS exit confirmed correct:** GS closed at $823.76 today, down from the $832.03 exit price. Holding GS would have added another -$19.80 in losses.

### Actual vs. Forecast (March 11)

| Ticker | Forecast (from Mar 10) | Actual Close | Difference |
|--------|------------------------|--------------|------------|
| NVDA | $184.87 | $186.03 | **+$1.16** ✅ |
| BAC | $48.71 | $48.52 | -$0.19 |
| CVX | $187.96 | $191.79 | **+$3.83** ✅ |
| UNH | $285.05 | $285.25 | +$0.20 ✅ |

All four positions beat or were near forecast. The remaining Mar 12–16 forecasts from the March 10 context are now **stale** — particularly CVX, which has already surpassed its Mar 16 target of $187.54.

### Updated Forecasts: March 12–16 (from March 11 Context)

| Date | NVDA (5.605 sh) | BAC (41.017 sh) | CVX (10.498 sh) | UNH (3.460 sh) | Cash | Adj Portfolio | Return |
|------|-----------------|-----------------|-----------------|----------------|------|---------------|--------|
| Mar 12 | $185.94 (+4.2%) | $48.48 (-0.6%) | $192.82 (+1.2%) | $287.55 (-0.5%) | $4,018 | $10,068 | **+0.68%** |
| Mar 13 | $184.91 (+3.6%) | $48.79 (+0.1%) | $191.88 (+0.7%) | $288.57 (-0.2%) | $4,018 | $10,069 | **+0.69%** |
| **Mar 16** | **$185.11 (+3.7%)** | **$48.85 (+0.2%)** | **$192.40 (+1.0%)** | **$288.55 (-0.2%)** | **$4,018** | **$10,078** | **+0.78%** |

**Updated Mar 16 target: $10,078 (+0.78%).** Up from the stale Mar 10 estimate of $10,009 (+0.09%), driven by CVX and NVDA outperforming prior forecasts.

Key signals from the fresh model:
- **CVX**: Predicts continued strength at $192.40 by Mar 16 (+1.0% vs entry). Mar 11 surge has reset the context — the model now sees $192+ as the new baseline, not a reversal back to $187.
- **NVDA**: Sees a slight pullback from $186.03 to $185.11 by Mar 16. The model views $186 as near-term resistance. No action needed given partial profit already taken.
- **UNH**: Gradual recovery from -1.29% to -0.15% by Mar 16. Half-position limits impact.
- **BAC**: Essentially flat, edging to +0.2% by Mar 16.

### Mar 12 Trading Decision

**All positions: Hold.**
- CVX is the strongest signal (+1.0% by Mar 16) — hold for the close
- NVDA half-position: model sees slight pullback but still +3.7% vs entry — no reason to sell the remaining half
- BAC and UNH both flat-to-recovering — transaction cost exceeds expected gain from trading
- 40% cash provides full downside protection for the final 3 days

**Watch levels for March 12:**
- CVX below $189.00: would indicate Mar 11 surge was a single-day spike; consider tightening stop
- NVDA below $183.00: exit trigger (entry $178.49 plus meaningful buffer)
- BAC below $46.80: original stop-loss, still in plan

### Updated Portfolio (March 11 Close)

| Position | Shares | Price | Value | % Portfolio |
|----------|--------|-------|-------|-------------|
| NVDA | 5.605 | $186.03 | $1,043 | 10.4% |
| BAC | 41.017 | $48.52 | $1,990 | 19.8% |
| CVX | 10.498 | $191.79 | $2,013 | 20.0% |
| UNH | 3.460 | $285.25 | $987 | 9.8% |
| **Cash** | — | — | **$4,018** | **40.0%** |
| **Total** | | | **$10,052** | **100%** |

### Final Scenarios (March 16)

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **Model base case (Mar 11 forecasts)** | **$10,078** | **+0.78%** |
| Bull — NVDA/CVX each +2% more | ~$10,138 | +1.38% |
| Flat — no movement from Mar 11 close | ~$10,052 | +0.52% |
| Bear — all equities down 3% | ~$9,899 | -1.01% |

*Portfolio is positive and model sees further upside to +0.78% by Mar 16. With 40% cash and only 3 days remaining, maximum practical drawdown is approximately -1.0%.*

---

## March 10 Update

*Updated: March 10, 2026 — Day 6 of 10*

### Actual Performance (Full History)

| Day | Date | Portfolio (adj) | Return | Original Forecast | Gap |
|-----|------|-----------------|--------|-------------------|-----|
| 1 | Mar 3 | $10,113.85 | +1.14% | +2.94% | -1.80pp |
| 2 | Mar 4 | $10,162.33 | +1.62% | +3.06% | -1.44pp |
| 3 | Mar 5 | $10,088.09 | +0.88% | +3.19% | -2.31pp |
| 4 | Mar 6 | $9,940.51 | -0.59% | +3.20% | -3.79pp |
| 5 | Mar 9 | $9,990.52 | -0.09% | +3.92% | -4.01pp |
| **6** | **Mar 10** | **$9,992.95** | **-0.07%** | +3.70% | **-3.77pp** |

Portfolio nearly flat two days in a row (-0.09% → -0.07%). Adjusted portfolio (post-trades) at **$9,972 (-0.28%)**.

### Per-Ticker Status (March 10 Close)

| Ticker | Entry | Mar 10 Close | Return | Held Shares | Position Value |
|--------|-------|--------------|--------|-------------|----------------|
| **NVDA** | $178.49 | $184.77 | **+3.52%** ✅ | 11.205 | $2,070 |
| **BAC** | $48.76 | $48.56 | -0.41% | 41.017 | $1,992 |
| **CVX** | $190.52 | $186.29 | **-2.22%** ⚠ | 10.498 | $1,956 |
| **UNH** | $288.99 | $282.34 | **-2.30%** ⚠ | 3.460 (half) | $977 |
| **GS** | $836.00 | $833.81 | -0.26% | 0 (exited) | — |
| **Cash** | — | — | — | — | $2,977 |
| **Total (adj)** | | | | | **$9,972** |

### Two New Concerns Today

**1. NVDA exceeded its original Mar 16 target today.**
- Original target: $184.13 (+3.16%)
- Today's close: $184.77 (+3.52%) — already **past** the target with 4 days remaining
- Updated model predicts $184.58 by Mar 16 (+3.41%) — essentially flat from here
- **Signal: take partial profit.** The model sees no further upside from current levels.

**2. CVX dropped sharply (-2.22%).**
- Fell from $189.44 to $186.29 in one day — now well below entry
- Updated model predicts CVX stays depressed through Mar 16 ($187.54 = -1.56% vs. entry)
- Stop-loss at $183.00 — currently $3.29 above it
- **Signal: monitor. Below $186, thesis is failing.**

### Updated Forecasts: March 11–16 (from March 10 Context)

| Date | NVDA | BAC | CVX | UNH (half) | Cash | Adj Portfolio | Return |
|------|------|-----|-----|-----------|------|---------------|--------|
| Mar 11 | $184.87 (+3.6%) | $48.71 (-0.1%) | $187.96 (-1.3%) | $285.05 (-1.4%) | $2,977 | $10,006 | **+0.06%** |
| Mar 12 | $183.92 (+3.0%) | $48.96 (+0.4%) | $187.38 (-1.6%) | $285.87 (-1.1%) | $2,977 | $10,002 | +0.02% |
| Mar 13 | $184.25 (+3.2%) | $48.98 (+0.4%) | $187.24 (-1.7%) | $286.24 (-1.0%) | $2,977 | $10,007 | +0.07% |
| **Mar 16** | **$184.58 (+3.4%)** | **$49.15 (+0.8%)** | **$187.54 (-1.6%)** | **$283.04 (-2.1%)** | **$2,977** | **$10,009** | **+0.09%** |

**Updated Mar 16 target: $10,009 (+0.09%).** The model sees a narrow path to breakeven, with NVDA carrying the portfolio.

### March 11 Trading Decision

**Decision 1 — NVDA: Take partial profit (recommended)**
- NVDA has already hit its original Mar 16 target. Updated model sees it flat at $184.58 by Mar 16 (+$0 gain from here).
- Selling half (5.60 shares ≈ $1,035) locks in +$98 on the half sold while keeping upside exposure.
- If NVDA retraces to entry ($178.49), the remaining half limits loss to ~$35.
- **Action: sell ~5 shares at Tuesday open (~$1,035 proceeds → add to cash).**

**Decision 2 — CVX: Hold but tighten stop**
- Updated model predicts -1.6% vs. entry through Mar 16. Thesis not working.
- But the loss from here to the stop ($186.29 → $183.00) is only -1.8%, and only 4 days remain.
- Not worth the friction of another trade unless the stop is hit.
- **Action: hold. Exit if CVX drops below $184.00 (tighter than original $183 stop).**

**Decision 3 — UNH, BAC: Hold**
- UNH at -2.30% on the half-position; updated model sees marginal recovery then slight fade. Small position, low impact.
- BAC essentially flat (-0.41%); model sees +0.8% by Mar 16. Hold.
- **Action: no change.**

### Revised Portfolio After March 11 Open (if NVDA partial sell executed)

| Position | Shares | Price (est.) | Value | % Portfolio |
|----------|--------|--------------|-------|-------------|
| NVDA | ~5.60 | $184.77 | ~$1,035 | 10.4% |
| BAC | 41.017 | $48.56 | $1,992 | 20.0% |
| CVX | 10.498 | $186.29 | $1,956 | 19.6% |
| UNH | 3.460 | $282.34 | $977 | 9.8% |
| **Cash** | — | — | **~$4,012** | **40.3%** |
| **Total** | | | **~$9,972** | **100%** |

*40% cash provides strong downside protection for the final 4 days.*

### Final Scenarios (March 16)

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **Model base case (NVDA partial sell)** | **~$10,025** | **+0.25%** |
| NVDA holds, all others flat | ~$10,010 | +0.10% |
| CVX hits tightened stop ($184) | ~$9,980 | -0.20% |
| All equities down 3% | ~$9,830 | -1.70% |

*With 40% cash, maximum practical drawdown is approximately -1.7% even in a severe week-end selloff.*

---

## March 9 Checkpoint Update

*Updated: March 9, 2026 — Day 5 of 10*

### Actual Performance (Full History)

| Day | Date | Portfolio (pre-trade) | Return | Original Forecast | Gap |
|-----|------|-----------------------|--------|-------------------|-----|
| 1 | Mar 3 | $10,113.85 | +1.14% | +2.78% | -1.64pp |
| 2 | Mar 4 | $10,162.33 | +1.62% | +2.90% | -1.28pp |
| 3 | Mar 5 | $10,088.09 | +0.88% | +3.02% | -2.14pp |
| 4 | Mar 6 | $9,940.51 | -0.59% | +3.04% | -3.63pp |
| **5** | **Mar 9** | **$9,975.17** | **-0.25%** | +3.76% | **-4.01pp** |

Slight recovery vs. March 6 (-0.59% → -0.25%), driven entirely by NVDA (+2.33%). All other tickers remain below entry.

### Per-Ticker Status (March 9 Close)

| Ticker | Entry | Mar 9 Close | Return | Orig Mar 16 Target | Upd Mar 16 Forecast | Action |
|--------|-------|-------------|--------|--------------------|---------------------|--------|
| **NVDA** | $178.49 | $182.65 | **+2.33%** | $184.13 (+3.16%) | $182.63 (+2.32%) | Hold |
| **GS** | $836.00 | $832.03 | -0.47% | $878.38 (+5.07%) | — | **Exited** |
| **CVX** | $190.52 | $189.44 | -0.57% | $191.58 (+0.55%) | $192.44 (+1.01%) | Hold |
| **UNH** | $288.99 | $285.17 | -1.32% | $306.39 (+6.02%) | $293.18 (+1.45%) | **Reduced 50%** |
| **BAC** | $48.76 | $47.90 | -1.76% | $51.19 (+4.98%) | $48.39 (-0.75%) | Hold |

### Trades Executed at March 9 Close

Per the March 6 decision rules:

| Trade | Shares | Price | Proceeds | Rationale |
|-------|--------|-------|----------|-----------|
| Sell 3.46 UNH | 3.460 | $285.17 | $986.78 | Plan rule: UNH < $299 → reduce to $1,000 |
| Sell all GS | 2.392 | $832.03 | $1,990.50 | Updated model signal inverted (negative) |
| **Total cash freed** | | | **$2,977.28** | Parked in cash |

**Post-trade portfolio value: $9,964.07 (-0.36%)**

### Revised Holdings After March 9 Trades

| Position | Shares | Mar 9 Price | Value | % of Portfolio |
|----------|--------|-------------|-------|----------------|
| NVDA | 11.205 | $182.65 | $2,047 | 20.5% |
| CVX | 10.498 | $189.44 | $1,989 | 20.0% |
| BAC | 41.017 | $47.90 | $1,965 | 19.7% |
| UNH | 3.460 | $285.17 | $987 | 9.9% |
| **Cash** | — | — | **$2,977** | **29.9%** |
| **Total** | | | **$9,964** | **100%** |

### Updated Forecasts: March 10–16 (from March 9 Context)

| Date | NVDA | CVX | BAC | UNH (half) | Cash | **Adj Portfolio** | Adj Return |
|------|------|-----|-----|-----------|------|-------------------|------------|
| Mar 10 | $183.06 | $190.37 | $47.82 | $289.42 | $2,977 | $9,990 | -0.10% |
| Mar 11 | $182.14 | $190.16 | $48.08 | $290.13 | $2,977 | $9,990 | -0.10% |
| Mar 12 | $182.17 | $190.29 | $48.17 | $290.01 | $2,977 | $9,995 | -0.05% |
| Mar 13 | $182.83 | $190.92 | $48.36 | $287.17 | $2,977 | $10,007 | +0.07% |
| **Mar 16** | **$182.63** | **$192.44** | **$48.39** | **$293.18** | **$2,977** | **$10,043** | **+0.43%** |

Updated Mar 16 target: **$10,043 (+0.43%)** — a narrow positive return driven by NVDA and a late CVX recovery.

### Key Signal Changes (March 9 vs. March 6 Update)

- **NVDA**: Only winner. Model now predicts +2.32% vs. entry through Mar 16 — the signal has held. Largest single contributor to recovery.
- **BAC**: Signal has turned **negative** (-0.75% vs. entry by Mar 16). Previously the "cleanest pick," now the model sees no recovery. Not worth another trade given small magnitude.
- **CVX**: Marginal positive (+1.01%). Model sees a late-week pickup (Mar 13–16 uptick in forecast).
- **UNH**: Reduced position sees modest recovery (+1.45% vs. entry) — contributes ~+$14 on the half-position.
- **Cash (30%)**: Acts as the floor. Prevents further drawdown regardless of equity moves.

### March 10 Trading Decision

**No new trades.** Rationale:
1. The portfolio is now defensively positioned (30% cash). Adding risk into a trend-down week is not justified by model signals.
2. BAC's updated signal is -0.75% but the magnitude is small (~-$15 loss on $1,965 position) — not worth trading cost and friction to exit.
3. NVDA is the portfolio's anchor — it holds above entry and the model sees it staying there.
4. CVX and UNH both show marginal late-week recoveries that push the portfolio to +0.43%.

**Watch levels for March 10:**
- NVDA below $178.49 (entry): exit trigger — would signal breakdown of the only positive thesis.
- BAC below $46.80 (plan stop-loss): exit trigger — still in plan.
- CVX below $183.00 (plan stop-loss): exit trigger — well above.
- UNH below $277.00 (plan stop-loss): exit trigger — well above.

### Final Scenarios (March 16)

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **Model base case** | **$10,043** | **+0.43%** |
| NVDA holds, others flat | ~$10,000 | 0.00% |
| NVDA reverses to entry (-2.3%) | ~$9,940 | -0.60% |
| All equities hit stop-losses (-4%) | ~$9,800 | -2.00% |

*30% cash position limits maximum drawdown to approximately -2% even in a severe equity decline.*

---

## March 6 Mid-Plan Update

*Updated: March 6, 2026 — Day 4 of 10*

### Actual Performance Through March 6

| Day | Date | Portfolio | Return | Orig Forecast | Gap |
|-----|------|-----------|--------|---------------|-----|
| 1 | Mar 3 | $10,113.85 | +1.14% | +2.67% | -1.53pp |
| 2 | Mar 4 | $10,162.33 | +1.62% | +2.79% | -1.17pp |
| 3 | Mar 5 | $10,088.09 | +0.88% | +2.91% | -2.03pp |
| **4** | **Mar 6** | **$9,929.23** | **-0.71%** | +2.92% | **-3.63pp** |

Portfolio crossed into negative territory on day 4. All 5 tickers are below entry.

| Ticker | Entry | Mar 6 Close | Return | Orig Mar 16 Target | Upd Mar 16 Target |
|--------|-------|-------------|--------|--------------------|-------------------|
| UNH | $288.99 | $286.48 | -0.87% | $306.39 (+6.0%) | $294.67 (+1.97%) |
| BAC | $48.76 | $48.64 | -0.25% | $51.19 (+4.98%) | $49.41 (+1.33%) |
| GS | $836.00 | $821.42 | -1.74% | $878.38 (+5.07%) | $831.93 (**-0.49%**) |
| CVX | $190.52 | $189.94 | -0.30% | $191.58 (+0.55%) | $192.37 (+0.97%) |
| NVDA | $178.49 | $177.82 | -0.38% | $184.13 (+3.16%) | $178.29 (-0.11%) |

**Updated portfolio target March 16: $10,073 (+0.73%)** vs. original $10,396 (+3.96%). Expected gain has shrunk by 82%.

*Updated forecasts generated using March 6 close as context (see `scripts/reforecast_mar6.py` and `finetune_expts/forecast_data_mar6_updated.json`). Fresher context reduced day-4 MAE by 64% ($14.55 → $5.27).*

### Key Finding: GS Signal Has Inverted

The most critical change: **GS's updated forecast is negative for the entire remaining horizon.** The model anchored to $861 on March 2; seeing $821 as the new context, it now predicts sideways-to-down through March 16. The original +5.07% thesis is gone.

### March 9 Trading Decisions (Monday Open)

**Decision 1 — UNH: REDUCE (plan rule triggered)**
- Rule: if UNH < $299 at March 9 checkpoint → reduce to $1,000.
- Current: $286.48 — **$12.51 below the $299 trigger.**
- Updated forecast: only +1.97% recovery by March 16. Not worth full $2,000 exposure given persistent weakness.
- **Action: Sell ~3.46 shares at Monday open. Keep $1,000 ($500 + redeploy $1,000 below).**

**Decision 2 — GS: EXIT**
- Not in original plan rules, but the updated model now predicts GS ends March 16 at $831.93 — still *below entry*.
- Stop-loss at $803 has not been hit, but the positive signal is gone.
- Holding through March 16 for a net loss (-0.49%) is not justified when the model saw a recovery.
- **Action: Sell all 2.392 shares at Monday open. Free up ~$1,964.**

**Decision 3 — BAC, CVX, NVDA: HOLD**
- BAC: modest +1.33% updated target. Cleanest chart, no negative signal.
- CVX: essentially flat but marginally positive (+0.97%). Small position, not worth the trading cost to exit.
- NVDA: flat (-0.11%) but close to breakeven. High-beta — could recover sharply or fall further.
- **Action: Hold all three. No changes.**

**Decision 4 — Redeployment of freed capital (~$2,964 from UNH reduction + GS exit)**

The updated model shows limited upside for the remaining 6 days across all 5 tickers. Options:

| Option | Action | Rationale |
|--------|--------|-----------|
| **A — Cash** | Park $2,964 in cash | Protect against further drawdown. Accept reduced exposure. |
| **B — Double BAC** | Add ~$1,500 to BAC | Best risk-adjusted signal in the portfolio (+1.33%). |
| **C — Add CVX** | Add ~$1,500 to CVX | Energy rebound play; updated forecast shows best late-week momentum. |
| **Recommended: Option A** | Hold cash | All updated forecasts are weak. The model has limited confidence in any ticker through March 16. Avoid adding risk into a trend-down week. |

### Revised Portfolio (effective Monday March 9 open)

| Ticker | Action | New Allocation | New Shares | Updated Mar 16 Target | Expected P&L |
|--------|--------|---------------|------------|----------------------|-------------|
| UNH | Reduce ~50% | ~$1,000 | ~3.46 | $294.67 | +$20 |
| BAC | Hold | $2,000 | 41.017 | $49.41 | +$27 |
| GS | **EXIT** | $0 | 0 | — | -$35 (realized) |
| CVX | Hold | $2,000 | 10.498 | $192.37 | +$25 |
| NVDA | Hold | $2,000 | 11.205 | $178.29 | -$2 |
| Cash | — | ~$2,964 | — | — | $0 |
| **Total** | | **~$10,000** | | | **~+$35** |

### Revised Scenarios (March 16)

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **Updated model base case** | **$10,008** | **+0.08%** |
| Bull (model signals recover 50%) | $10,200 | +2.0% |
| Flat (no movement from Mar 9 open) | ~$9,964 | -0.36% |
| Bear (GS/NVDA/UNH each -3%) | ~$9,700 | -3.0% |

*With GS exit and UNH reduction, the portfolio is now defensively positioned: 50% in cash, BAC and CVX as the core holds.*

---

## How This Plan Works

There are two distinct steps:

**Step 1 — Backtest validates the strategy approach.**
A clean 12-month simulation (March 2025 – February 2026) across 18 uncontaminated tickers confirmed that the ForecastTop3 / SelectiveTop3 strategy — pick the top-predicted tickers each month — outperforms buy-and-hold and the S&P 500 across multiple universes. Set B (Large Cap Diversified) produced the best risk-adjusted result: +28.0% return, Sharpe 1.30, max drawdown -7.6%.

**Step 2 — Current forecasts determine which tickers to hold right now.**
The backtest does not tell us which tickers to buy *today*. For that, we run the model forward and rank all 18 uncontaminated tickers by their predicted 10-day return. We buy the top positively-predicted tickers regardless of which backtest set they came from.

---

## Current 10-Day Forecast Ranking (All Clean Tickers)

Forecasts generated March 2, 2026. Entry = March 3 open.

| Rank | Ticker | Set | Model | Entry (Mar 3 open) | 10-Day Target | Predicted Return | Include? |
|---|---|---|---|---|---|---|---|
| 1 | **UNH** | A | Run4 | $288.99 | $306.39 | **+6.02%** | ✓ Top 5 |
| 2 | **BAC** | B | Run4 | $48.76 | $51.19 | **+4.98%** | ✓ Top 5 |
| 3 | **GS** | A | Glia4b | $836.00 | $878.38 | **+5.07%** | ✓ Top 5 |
| 4 | **CVX** | A | Glia3a | $190.51 | $191.58 | **+0.56%** | ✓ Top 5 |
| 5 | **NVDA** | A | Glia4b | $178.48 | $184.13 | **+3.17%** | ✓ Top 5 |
| 6 | MSFT | B | Run4 | $398.55 | $402.04 | +0.88% | Option B only |
| 7 | LMT | B | Run4 | $676.70 | $673.89 | -0.42% | ✗ Negative |
| 8 | COST | B | Run4 | $1002.77 | $996.56 | -0.62% | ✗ Negative |
| 9 | MRK | B | Run4 | $121.41 | $120.37 | -0.86% | ✗ Negative |
| 10 | COP | B | Run4 | $118.24 | $117.14 | -0.93% | ✗ Negative |

*Note: Set A tickers KO and BA not shown (KO predicted -0.1%, BA predicted -0.5%).*
*Predicted returns recalculated from actual March 3 open prices, not March 2 closes.*

**Key observation:** Set B produced the best 12-month backtest result — but that was driven by LMT (+56%) and MRK (+39%) over the past year. As of today, the model predicts declines for LMT, MRK, COP, and COST. Only BAC (+4.98%) is a positive Set B pick. The strategy correctly excludes Set B's historical winners when the model sees no near-term upside.

---

## Portfolio Allocation

**Option A — Full Signal (top 5 positive-predicted tickers):**

| Ticker | Company | Set | Allocation | Amount | Entry | Shares | Model |
|---|---|---|---|---|---|---|---|
| **UNH** | UnitedHealth Group | A | 20% | $2,000 | $288.99 | 6.92 | Run4 |
| **BAC** | Bank of America | B | 20% | $2,000 | $48.76 | 41.02 | Run4 |
| **GS** | Goldman Sachs | A | 20% | $2,000 | $836.00 | 2.39 | Glia4b |
| **CVX** | Chevron | A | 20% | $2,000 | $190.51 | 10.50 | Glia3a |
| **NVDA** | NVIDIA | A | 20% | $2,000 | $178.48 | 11.21 | Glia4b |
| **Total** | | | **100%** | **$10,000** | | | |

**Option B — Conservative (swap UNH for MSFT, avoid fundamental risk):**

| Ticker | Company | Set | Allocation | Amount | Entry | Shares | Model |
|---|---|---|---|---|---|---|---|
| BAC | Bank of America | B | 20% | $2,000 | $48.76 | 41.02 | Run4 |
| GS | Goldman Sachs | A | 20% | $2,000 | $836.00 | 2.39 | Glia4b |
| CVX | Chevron | A | 20% | $2,000 | $190.51 | 10.50 | Glia3a |
| NVDA | NVIDIA | A | 20% | $2,000 | $178.48 | 11.21 | Glia4b |
| MSFT | Microsoft | B | 20% | $2,000 | $398.55 | 5.02 | Run4 |
| **Total** | | | **100%** | **$10,000** | | | |

*Entry prices are actual March 3, 2026 opening prices. Orders placed at market open.*

---

## 10-Day Price Forecasts

### Option A

| Date | Day | UNH | BAC | GS | CVX | NVDA |
|---|---|---|---|---|---|---|
| Mar 3 | 1 | $300.35 | $50.01 | $867.01 | $190.16 | $184.42 |
| Mar 4 | 2 | $301.85 | $50.30 | $866.87 | $189.66 | $183.98 |
| Mar 5 | 3 | $301.18 | $50.35 | $867.06 | $190.52 | $184.46 |
| Mar 6 | 4 | $297.70 | $50.52 | $873.56 | $190.62 | $184.64 |
| **Mar 9** | **5** | **$304.18** | **$50.56** | **$876.43** | **$192.22** | **$184.77** |
| Mar 10 | 6 | $303.22 | $50.67 | $874.76 | $191.67 | $183.91 |
| Mar 11 | 7 | $305.30 | $50.86 | $874.20 | $192.48 | $183.92 |
| Mar 12 | 8 | $305.91 | $50.88 | $878.61 | $192.59 | $185.24 |
| Mar 13 | 9 | $306.60 | $51.04 | $877.56 | $192.78 | $184.36 |
| **Mar 16** | **10** | **$306.39** | **$51.19** | **$878.38** | **$191.58** | **$184.13** |

*Forecasts are model outputs from March 2 context. They are not adjusted for the actual March 3 open gap-down.*

### Option B

| Date | Day | BAC | GS | CVX | NVDA | MSFT |
|---|---|---|---|---|---|---|
| Mar 3 | 1 | $50.01 | $867.01 | $190.16 | $184.42 | $400.98 |
| Mar 4 | 2 | $50.30 | $866.87 | $189.66 | $183.98 | $401.50 |
| Mar 5 | 3 | $50.35 | $867.06 | $190.52 | $184.46 | $400.11 |
| Mar 6 | 4 | $50.52 | $873.56 | $190.62 | $184.64 | $399.69 |
| **Mar 9** | **5** | **$50.56** | **$876.43** | **$192.22** | **$184.77** | **$401.73** |
| Mar 10 | 6 | $50.67 | $874.76 | $191.67 | $183.91 | $402.55 |
| Mar 11 | 7 | $50.86 | $874.20 | $192.48 | $183.92 | $402.06 |
| Mar 12 | 8 | $50.88 | $878.61 | $192.59 | $185.24 | $400.88 |
| Mar 13 | 9 | $51.04 | $877.56 | $192.78 | $184.36 | $401.82 |
| **Mar 16** | **10** | **$51.19** | **$878.38** | **$191.58** | **$184.13** | **$402.04** |

---

## Predicted Returns by Ticker

### Option A

| Ticker | Entry | 5-Day Target | 5-Day Return | 10-Day Target | 10-Day Return | Predicted $ Gain |
|---|---|---|---|---|---|---|
| **UNH** | $288.99 | $304.18 | +5.26% | $306.39 | **+6.02%** | +$120.40 |
| **GS** | $836.00 | $876.43 | +4.84% | $878.38 | **+5.07%** | +$101.40 |
| **BAC** | $48.76 | $50.56 | +3.69% | $51.19 | **+4.98%** | +$99.60 |
| **NVDA** | $178.48 | $184.77 | +3.52% | $184.13 | **+3.17%** | +$63.40 |
| **CVX** | $190.51 | $192.22 | +0.90% | $191.58 | **+0.56%** | +$11.20 |
| **Portfolio** | $10,000 | | **+3.64%** | | **+3.96%** | **+$396.00** |

### Option B

| Ticker | Entry | 5-Day Target | 5-Day Return | 10-Day Target | 10-Day Return | Predicted $ Gain |
|---|---|---|---|---|---|---|
| **GS** | $836.00 | $876.43 | +4.84% | $878.38 | **+5.07%** | +$101.40 |
| **BAC** | $48.76 | $50.56 | +3.69% | $51.19 | **+4.98%** | +$99.60 |
| **NVDA** | $178.48 | $184.77 | +3.52% | $184.13 | **+3.17%** | +$63.40 |
| **MSFT** | $398.55 | $401.73 | +0.80% | $402.04 | **+0.88%** | +$17.60 |
| **CVX** | $190.51 | $192.22 | +0.90% | $191.58 | **+0.56%** | +$11.20 |
| **Portfolio** | $10,000 | | **+2.75%** | | **+2.93%** | **+$293.20** |

*Dollar gains based on $2,000 allocation per ticker. Returns recalculated from actual March 3 open prices.*

---

## Per-Ticker Analysis

### UNH — Strongest Signal (+6.02%) ⚠ Fundamental Risk
The model predicts a rebound from $288.99 to $306.39. The March 3 open gap-down (from $294.93 to $288.99) amplified the predicted return — the model is calling a recovery from a depressed price.

> ⚠ **Warning:** UNH declined -35.5% over the past 12 months and has a -60% max drawdown. The model cannot see ongoing regulatory pressure or margin compression. This is the highest-risk position. Consider capping at $1,000 if you are concerned, and redirect the other $1,000 to GS or BAC.

**Key level:** $300 by March 9. Failure to break above $295 in week 1 weakens the thesis.

### GS — High Signal, Tracking Well (+5.07%)
Already +3.18% on day 1. Glia4b (the per-ticker best model for GS) predicts continued climb to $878. The trajectory accelerates in week 2 (March 11–13).

**Key level:** $870 by end of week 1. Above $880, consider partial profit — that is 2× the predicted move.

### BAC — Best Clean Pick from Set B (+4.98%)
The only Set B ticker with a strong positive current forecast. BAC has no training data overlap, making it the highest-confidence signal with the cleanest track record. Forecast trajectory is near-monotonic — the smoothest path of all five tickers.

**Key level:** $50.50 by March 6.

### NVDA — Strong Signal (+3.17%)
Opened at $178.48 (down from $182.48 close), which increased the predicted return. The March 12 peak ($185.24) is the highest point in the forecast before a slight pullback to $184.13 on March 16.

**Key level:** $184.50 by March 12. NVDA is the highest-beta stock here — any AI or macro news overrides the model signal.

### CVX — Weakest Signal (+0.56%)
The smallest predicted gain in the portfolio. Oil price exposure dominates the model's statistical signal. CVX provides diversification into energy but contributes minimally to expected return.

**Key level:** $192 by March 9. If CVX fails to recover above entry ($190.51) by March 5, the thesis is not working.

---

## Suggested Order Types

### Option A

| Ticker | Order Type | Entry | Stop-Loss | Take-Profit |
|---|---|---|---|---|
| UNH | Limit buy | $288.99 | $277.00 (-4.1%) | $324.00 (+12% = 2× signal) |
| BAC | Limit buy | $48.76 | $46.80 (-4.0%) | $53.60 (+9.9% = 2× signal) |
| GS | Limit buy | $836.00 | $803.00 (-3.9%) | $920.50 (+10.1% = 2× signal) |
| CVX | Limit buy | $190.51 | $183.00 (-3.9%) | $191.62 (+0.6% ≈ target) |
| NVDA | Limit buy | $178.48 | $171.00 (-4.2%) | $189.78 (+6.3% = 2× signal) |

### Option B

| Ticker | Order Type | Entry | Stop-Loss | Take-Profit |
|---|---|---|---|---|
| BAC | Limit buy | $48.76 | $46.80 (-4.0%) | $53.60 (+9.9% = 2× signal) |
| GS | Limit buy | $836.00 | $803.00 (-3.9%) | $920.50 (+10.1% = 2× signal) |
| CVX | Limit buy | $190.51 | $183.00 (-3.9%) | $191.62 (+0.6% ≈ target) |
| NVDA | Limit buy | $178.48 | $171.00 (-4.2%) | $189.78 (+6.3% = 2× signal) |
| MSFT | Limit buy | $398.55 | $383.00 (-3.9%) | $405.60 (+1.8% = 2× signal) |

---

## Execution Plan

**March 3 (today — already executed):**
- 5 limit buy orders placed at market open
- Actual entry prices confirmed (all gap-down from March 2 close)
- Portfolio at +1.14% end of day 1

**March 9 (mid-week review — 5-day checkpoint):**
- Compare actual prices against 5-day targets in the table above
- UNH: if < $299, reduce to $1,000, rotate into BAC or GS
- GS: if > $880, consider taking 50% profit
- CVX: if still below $190, evaluate whether to hold or redeploy

**March 16 (end of horizon):**
- Evaluate actual vs forecast performance per ticker
- Re-run `scripts/investment_sim.py` for updated April allocations
- The model will re-rank all 18 clean tickers — picks will likely differ from today

---

## Expected Portfolio Value at End of 2 Weeks

### Option A

| Scenario | Portfolio Value | Return |
|---|---|---|
| **Model forecast (base case)** | **$10,396** | **+3.96%** |
| Bull case (+2× model signal) | $10,792 | +7.92% |
| Flat (no movement) | $10,000 | 0.00% |
| Bear case (−4% drawdown) | $9,600 | -4.00% |

### Option B

| Scenario | Portfolio Value | Return |
|---|---|---|
| **Model forecast (base case)** | **$10,293** | **+2.93%** |
| Bull case (+2× model signal) | $10,586 | +5.86% |
| Flat (no movement) | $10,000 | 0.00% |
| Bear case (−4% drawdown) | $9,600 | -4.00% |

*Annualized equivalent: Option A base case ~49%, Option B ~36%*

---

## Risk Factors

1. **UNH fundamental risk (Option A):** The time-series model sees price momentum, not business fundamentals. The +6.02% signal is partly a gap-down artifact — the stock opened $6 below yesterday's close. Treat UNH as a speculative recovery bet.
2. **Gap-down amplification:** All predicted returns are higher than the original plan because the March 3 open was significantly lower than March 2 close (GS -3%, UNH -2%, NVDA -2.2%). The model forecasts the same dollar targets — the returns look larger because we bought cheaper.
3. **Set B LMT/MRK exclusion:** The model currently predicts declines for Set B's historical winners. This could reverse mid-month. If you want Set B exposure, monitor LMT and MRK — if the model's signal turns positive, they could be added in a next rebalance.
4. **CVX oil exposure:** WTI crude dominates CVX more than any model signal.
5. **NVDA event risk:** Single news events move NVDA ±5–10%, overwhelming the model's statistical forecast.
6. **Short horizon noise:** Models validated at 120-day horizon. 10-day forecasts carry more noise.

---

*Report generated: March 3, 2026*
*Forecast model: TimesFM 2.5 (200M) — Run4 + Glia3a + Glia4b selective routing*
*Forecast data: finetune_expts/forecast_data_v2.json | Entry prices: actual March 3 open*
*Live simulation: scripts/sim_2week_live.py | Reference: finetune_expts/report_clean_simulation.md*
