# 2-Week Global Investment Plan — $10,000

**Date:** March 14, 2026 (forecasts from March 13 close context)
**Horizon:** 10 trading days (March 16 – March 27, 2026)
**Universe:** 15 global tickers across USA, China/HK, Europe, Japan
**Models:** TimesFM 2.5 — regional finetuned checkpoints

> ⚠ **Disclaimer:** This plan is based on machine-learning time-series models. It is not professional financial advice. Models have no knowledge of upcoming earnings, macro events, or news. Currency risk applies to HKD (pegged, minimal) and JPY (floating, material). Past simulation performance does not guarantee future results.

---

## Context: What Happened in the Previous Plan (Mar 3–16)

The previous 2-week plan (USA tickers only) ended near breakeven (~+0.04%). Key lessons:
- **CVX surged +3.4%** on an oil price shock — the model caught the directional trend
- **GS fell -6.3%** from entry — model signal inverted by Day 4, triggering an early exit that saved ~$106
- **UNH recovered to $282** after touching the $277 stop-loss on March 12 — model now sees further upside
- **NVDA** fell to $180.25 as of March 13, sitting below the $183 watch level from the previous plan

This plan expands the universe to all 4 regions to access the China model's strong signal on 2382.HK.

---

## Full Ranking — All 15 Global Tickers (10-Day Predicted Return)

Context date: March 13, 2026 close. Forecast horizon: March 16–27.

| Rank | Ticker | Region | Model | Mar 13 Close | 10d Target | Pred Return | Include? |
|------|--------|--------|-------|--------------|------------|-------------|----------|
| 1 | **2382.HK** | China | Glia_China | HKD 56.40 (~$7.25) | HKD 57.94 | **+2.73%** | ✓ |
| 2 | **UNH** | USA | Run4 | $282.09 | $288.74 | **+2.36%** | ✓ |
| 3 | **CVX** | USA | Glia3a | $196.82 | $199.46 | **+1.34%** | ✓ |
| 4 | 6902.T | Japan | Pretrained | ¥1,937.50 (~$13.25) | ¥1,961.61 | +1.24% | Option B |
| 5 | 8035.T | Japan | Pretrained | ¥38,340 (~$262) | ¥38,757 | +1.09% | Option B |
| 6 | **VWAGY** | Europe | Run1_EU | $10.30 | $10.39 | **+0.87%** | ✓ |
| 7 | **0388.HK** | China | Pretrained | HKD 401.40 (~$51.60) | HKD 403.02 | **+0.40%** | ✓ |
| 8 | EADSF | Europe | Run1_EU | $192.00 | $192.33 | +0.17% | ✗ Marginal |
| 9 | NVDA | USA | Glia4b | $180.25 | $180.52 | +0.15% | ✗ Marginal |
| 10 | 4661.T | Japan | Pretrained | ¥2,787.00 | ¥2,789.72 | +0.10% | ✗ Marginal |
| 11 | 1177.HK | China | Glia_China | HKD 5.95 | HKD 5.94 | -0.17% | ✗ Negative |
| 12 | UL | Europe | Run1_EU | $64.05 | $63.93 | -0.19% | ✗ Negative |
| 13 | BA | USA | Run4 | $209.89 | $209.35 | -0.26% | ✗ Negative |
| 14 | GS | USA | Glia4b | $782.21 | $776.61 | -0.72% | ✗ Negative |
| 15 | KO | USA | Glia4b | $77.34 | $76.58 | -0.99% | ✗ Negative |

**Key observation:** 6 of 15 tickers show positive signals. Japan tickers 6902.T and 8035.T are positive but require JPY and Tokyo Stock Exchange access. Option A avoids JPY exposure.

---

## Portfolio Allocation

### Option A — Global USD/HKD (no JPY risk, recommended for most investors)

Five tickers with positive signals accessible via US or HK exchanges:

| Ticker | Company | Region | Allocation | Amount | Entry (Mar 16 open) | Approx Shares | Model |
|--------|---------|--------|-----------|--------|---------------------|---------------|-------|
| **2382.HK** | Sunny Optical Technology | China/HK | 20% | $2,000 | HKD ~56.40 | ~276 shares | Glia_China |
| **UNH** | UnitedHealth Group | USA | 20% | $2,000 | ~$282.09 | ~7.09 | Run4 |
| **CVX** | Chevron | USA | 20% | $2,000 | ~$196.82 | ~10.16 | Glia3a |
| **VWAGY** | Volkswagen ADR | Europe | 20% | $2,000 | ~$10.30 | ~194 | Run1_EU |
| **0388.HK** | HKEX | China/HK | 20% | $2,000 | HKD ~401.40 | ~38.8 shares | Pretrained |
| **Total** | | | **100%** | **$10,000** | | | |

*Entry prices are March 13 closes. Actual entries will be March 16 opens — use limit orders within 0.5% of these levels.*
*HKD allocation: 2382.HK ≈ HKD 15,566; 0388.HK ≈ HKD 15,574 (1 USD ≈ 7.78 HKD)*

### Option B — Full Global (includes JPY exposure, higher expected return)

Replace VWAGY and 0388.HK with the top Japan signals:

| Ticker | Company | Region | Allocation | Amount | Entry Est. | Model |
|--------|---------|--------|-----------|--------|------------|-------|
| **2382.HK** | Sunny Optical Technology | China | 20% | $2,000 | HKD ~56.40 | Glia_China |
| **UNH** | UnitedHealth Group | USA | 20% | $2,000 | ~$282.09 | Run4 |
| **CVX** | Chevron | USA | 20% | $2,000 | ~$196.82 | Glia3a |
| **6902.T** | Denso Corp | Japan | 20% | $2,000 | ¥~1,937 (~$13.25) | Pretrained |
| **8035.T** | Tokyo Electron | Japan | 20% | $2,000 | ¥~38,340 (~$262) | Pretrained |

*Note: 6902.T and 8035.T require direct TSE access or Japan ADRs. JPY has been appreciating against USD — a JPY strengthening during the period would add to USD-equivalent returns.*

---

## 10-Day Price Forecasts

### Option A Tickers

| Date | Day | 2382.HK (HKD) | UNH | CVX | VWAGY | 0388.HK (HKD) | Portf. Return |
|------|-----|---------------|-----|-----|-------|----------------|---------------|
| Mar 16 | 1 | 56.23 | $286.21 | $197.05 | $10.40 | 399.62 | +0.30% |
| Mar 17 | 2 | 56.12 | $287.18 | $196.71 | $10.44 | 399.11 | +0.37% |
| Mar 18 | 3 | 56.28 | $286.57 | $197.43 | $10.40 | 398.42 | +0.43% |
| Mar 19 | 4 | 56.45 | $283.01 | $197.90 | $10.41 | 398.53 | +0.31% |
| **Mar 20** | **5** | **56.58** | **$288.55** | **$199.36** | **$10.33** | **398.80** | **+0.95%** |
| Mar 23 | 6 | 56.84 | $287.52 | $198.45 | $10.27 | 400.29 | +1.18% |
| Mar 24 | 7 | 57.05 | $288.40 | $199.53 | $10.29 | 400.04 | +1.48% |
| Mar 25 | 8 | 57.08 | $288.69 | $201.05 | $10.20 | 400.69 | +1.57% |
| Mar 26 | 9 | 57.47 | $289.02 | $200.98 | $10.18 | 402.50 | +1.85% |
| **Mar 27** | **10** | **57.94** | **$288.74** | **$199.46** | **$10.39** | **403.02** | **+1.94%** |

*Portfolio return = equal-weighted average of 5 ticker returns (USD-equivalent)*

### Option B Tickers (Japan substitution)

| Date | Day | 2382.HK (HKD) | UNH | CVX | 6902.T (JPY) | 8035.T (JPY) | Portf. Return |
|------|-----|---------------|-----|-----|--------------|--------------|---------------|
| Mar 16 | 1 | 56.23 | $286.21 | $197.05 | ¥1,943 | ¥38,548 | +0.59% |
| Mar 17 | 2 | 56.12 | $287.18 | $196.71 | ¥1,944 | ¥38,422 | +0.55% |
| Mar 18 | 3 | 56.28 | $286.57 | $197.43 | ¥1,948 | ¥38,341 | +0.58% |
| Mar 19 | 4 | 56.45 | $283.01 | $197.90 | ¥1,950 | ¥38,229 | +0.42% |
| **Mar 20** | **5** | **56.58** | **$288.55** | **$199.36** | **¥1,951** | **¥38,223** | **+0.77%** |
| Mar 23 | 6 | 56.84 | $287.52 | $198.45 | ¥1,953 | ¥38,395 | +1.07% |
| Mar 24 | 7 | 57.05 | $288.40 | $199.53 | ¥1,955 | ¥38,561 | +1.37% |
| Mar 25 | 8 | 57.08 | $288.69 | $201.05 | ¥1,959 | ¥38,660 | +1.50% |
| Mar 26 | 9 | 57.47 | $289.02 | $200.98 | ¥1,959 | ¥38,718 | +1.71% |
| **Mar 27** | **10** | **57.94** | **$288.74** | **$199.46** | **¥1,962** | **¥38,757** | **+1.88%** |

*Japan returns are in JPY; USD-equivalent depends on JPY/USD rate during the period.*

---

## Per-Ticker Analysis

### 2382.HK — Strongest Signal (+2.73%) — Sunny Optical Technology
China's finetuned model (Glia_China) has the highest conviction on this ticker. Rolling evaluation showed it improved 2382.HK by **-8.9pp MAPE** — the largest improvement of any ticker across all four regions. The model predicts a steady climb from HKD 56.40 to 57.94, with acceleration in week 2 (March 23–27).

- **Note on 0388.HK**: Our rolling eval showed the China model is less reliable for HKEX (exchange operator) — baseline preferred. The +0.40% signal is from the finetuned model; treat it as a weaker pick. Consider reducing its allocation to 10% and adding to 2382.HK.
- **Key level:** HKD 57.00 by March 20. Failure to reach this level would weaken the week-2 outlook.
- **Stop-loss:** HKD 54.14 (~-4% from entry)

### UNH — Recovery Signal (+2.36%) — UnitedHealth Group
UNH touched its previous plan's stop-loss ($277.05 on March 12) then recovered to $282.09. The model now sees a recovery path to $288.74, reaching near-target by March 20 ($288.55) and holding through March 27.

- Trajectory: mostly front-loaded — strong first week, slight dip on March 19, then steady.
- The March 12 oil-shock-driven selloff appears to have passed; model context anchored at $282 sees recovery.
- **Key level:** $288 by March 20. If UNH is still below $280 on March 20, exit.
- **Stop-loss:** $271.00 (~-4% from $282.09 entry)

### CVX — Energy Continuation (+1.34%) — Chevron
CVX is now *above* its original March 3 plan's 10-day target ($191.58) having surged to $196.82 on the oil spike. The model sees further upside to $199.46, driven by week-2 momentum (Mar 25 peak of $201.05).

- Context reset: the model now anchors at $196–197 as the new base, not reverting to pre-spike levels.
- Pattern: gradual climb with a slight mid-week consolidation (Mar 23: $198.45) before resuming.
- **Key level:** $199 by March 25. Below $195 would indicate oil retreat.
- **Stop-loss:** $189.00 (~-4% from entry). Tight given oil market volatility.

### VWAGY — European Pick (+0.87%) — Volkswagen ADR
Europe's model improvement was small (+0.20pp rolling avg) but consistent. VWAGY is Europe's top-ranked ticker this cycle. The ADR is USD-denominated, avoiding FX complexity.

- Trajectory: gradual rise, slight pullback in week 2 (Mar 25–26), recovering by March 27.
- **Caution:** VWAGY was the problem ticker in Europe rolling eval (+1.2pp regression on some windows). Consider capping at $1,000 if risk appetite is limited.
- **Key level:** $10.39 by March 27 (essentially the target). Stop at $9.90 (-4%).

### 6902.T / 8035.T — Japan Tickers (Option B)
Japan finetuning uniformly hurt all tickers; we use the pretrained baseline here.
- **6902.T (Denso)**: Steady climb ¥1,937→¥1,962, +1.24%. Most stable Japan signal.
- **8035.T (Tokyo Electron)**: Dips in week 1 (¥38,229 on Mar 19) before recovering to ¥38,757 by Mar 27. Week-1 weakness is a risk; hold if above ¥37,000.
- **JPY risk**: A 2% JPY weakening against USD would erase the +1% equity gain. Monitor USD/JPY.

---

## Suggested Order Types

### Option A

| Ticker | Order | Entry | Stop-Loss | Mid-Plan Target | 10d Target |
|--------|-------|-------|-----------|-----------------|------------|
| 2382.HK | Limit buy | HKD 56.40 | HKD 54.14 (-4%) | HKD 56.58 | HKD 57.94 |
| UNH | Limit buy | $282.09 | $271.00 (-4%) | $288.55 | $288.74 |
| CVX | Limit buy | $196.82 | $189.00 (-4%) | $199.36 | $199.46 |
| VWAGY | Limit buy | $10.30 | $9.90 (-4%) | $10.33 | $10.39 |
| 0388.HK | Limit buy | HKD 401.40 | HKD 385.34 (-4%) | HKD 398.80 | HKD 403.02 |

---

## Execution Plan

**March 16 (Monday open — Entry day):**
- Place 5 limit buy orders at or near March 13 close prices
- If 2382.HK opens above HKD 57.50, reduce size or skip (signal diminished)
- If UNH gaps up above $290 at open, skip (already near target)

**March 20 (5-day checkpoint):**
- Compare actuals vs 5-day targets above
- **2382.HK**: if below HKD 56.00, consider reducing to half-position
- **0388.HK**: if below HKD 397.00, consider exiting (weakest signal, most volatile)
- **UNH**: if above $289, consider taking 50% profit (near 10-day target by day 5)
- Re-run forecast from March 20 context to get fresh signals for week 2

**March 27 (End of horizon):**
- Close all remaining positions
- Re-run global ranking for the following 2-week plan

---

## Expected Portfolio Value

### Option A

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **Model base case (equal-weighted)** | **$10,194** | **+1.94%** |
| Bull — all signals × 2 | $10,388 | +3.88% |
| Mid — only 2382.HK + UNH + CVX work | $10,129 | +1.29% |
| Flat — no movement from entry | $10,000 | 0.00% |
| Bear — 4% drawdown across all | $9,600 | -4.00% |

### Option B

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **Model base case (JPY-neutral)** | **$10,188** | **+1.88%** |
| Bull — all signals × 2 + JPY +2% | $10,492 | +4.92% |
| Bear — JPY weakens 3% + equities flat | $9,740 | -2.60% |

*Annualized equivalent: Option A base case ~+50%. Higher than simple extrapolation due to compound effect.*

---

## Risk Factors

1. **2382.HK concentration risk:** The strongest signal is also a high-volatility HK technology stock. Sunny Optical has a 30-day historical vol of ~25% annualized — a 1-sigma adverse move over 10 days is ~3.5%.
2. **0388.HK model reliability:** Rolling eval showed the China finetuned model is less reliable for HKEX than for 2382.HK/1177.HK. Consider routing 0388.HK to baseline model (signal still +0.40%) or reducing position size.
3. **CVX oil dependence:** Model signal can be overridden by a sudden oil price reversal. A $5/bbl drop in WTI would likely push CVX below the $189 stop.
4. **UNH regulatory overhang:** Same as previous plan — the model sees price momentum, not regulatory news. Treat UNH as a momentum recovery bet, not a fundamental long.
5. **Japan JPY risk (Option B):** If JPY depreciates 2%+ against USD during the period, the Japan equity gains are fully offset.
6. **Short horizon model noise:** TimesFM validated at 120-day horizon. 10-day forecasts are noisier. Previous plan showed ~3–4pp gap between model predictions and actual outcomes.
7. **Easter holiday (March 28–30):** Good Friday (March 28) — US and European markets closed. TSE closes for Golden Week the following week. Plan ends on March 27 to avoid holiday disruption.

---

## March 27 Update — Day 10 — FINAL

*Plan complete. UNH crashed further despite not being exited. CVX surged to +7.40%. Script tracks all 5 original positions.*

### Full 10-Day Performance

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | — | +0.65% | +0.37% |
| 2 | Mar 17 | — | +0.97% | +0.40% |
| 3 | Mar 18 | — | +0.28% | +0.38% |
| 4 | Mar 19 | — | -0.21% | +0.26% |
| 5 | Mar 20 | $9,827 | -1.73% | +0.71% |
| 6 | Mar 23 | $9,745 | -2.55% | +0.60% |
| 7 | Mar 24 | $9,858 | -1.42% | +0.87% |
| 8 | Mar 25 | $9,894 | -1.06% | +0.91% |
| 9 | Mar 26 | $9,716 | -2.84% | +1.12% |
| **10** | **Mar 27** | **$9,781** | **-2.19%** | +1.54% |

*Script tracks all 5 original positions. VWAGY exited Mar 20 in practice (cash ~$1,984).*

### Final Per-Ticker Outcomes

| Ticker | Entry | Final | Return | Pred Return | Delta | Held to end? |
|--------|-------|-------|--------|-------------|-------|--------------|
| **CVX** | $196.60 | $211.15 | **+7.40%** | +1.34% | +6.06pp | Yes |
| **VWAGY** | $10.30 | $10.08 | -2.14% | +0.87% | -3.01pp | Exited Mar 20 |
| **0388.HK** | HKD 401.40 | HKD 390.40 | -2.74% | +0.40% | -3.14pp | Yes (not exited Mar 24) |
| **2382.HK** | HKD 56.40 | HKD 53.75 | **-4.70%** | +2.73% | -7.43pp | Below stop throughout |
| **UNH** | $283.98 | $259.02 | **-8.79%** | +2.36% | -11.15pp | Yes (not exited Mar 24) |

### Scenario Comparison — Impact of March 24 Decision

| Scenario | Final Value | Return |
|----------|------------|--------|
| All 5 held (script) | $9,781 | -2.19% |
| VWAGY exited Mar 20 only | ~$9,797 | -2.03% |
| VWAGY + 2382.HK exited at stops | ~$9,856 | -1.44% |
| VWAGY + 2382.HK + UNH exited at stops | ~$9,953 | **-0.47%** |

Not executing the UNH exit on March 24 (when it was above the global stop at $272.28) cost ~$93 as UNH fell to $259.02 by end of plan. 2382.HK continued declining from HKD 52.90 to HKD 53.75 — slight recovery but never cleared the stop.

**CVX was the sole winner** — +7.40% vs +1.34% predicted. Dramatically underpredicted by the model for the third consecutive plan.

---

## March 24 Update — Day 7 — Partial Recovery

*Day 7 of 10. Market bounced. UNH and 0388.HK recovered above their stops. 2382.HK still below stop.*
*Note: Plan recommended exiting 2382.HK, UNH, 0388.HK at today's open — if executed, recoveries were missed.*

### Actual Performance (Days 1–7)

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | — | +0.65% | +0.37% |
| 2 | Mar 17 | — | +0.97% | +0.40% |
| 3 | Mar 18 | — | +0.28% | +0.38% |
| 4 | Mar 19 | — | -0.21% | +0.26% |
| 5 | Mar 20 | $9,827 | -1.73% | +0.71% |
| 6 | Mar 23 | $9,745 | -2.55% | +0.60% |
| **7** | **Mar 24** | **$9,858** | **-1.42%** | +0.87% |

*Script tracks all 5 original positions (including VWAGY and positions recommended for exit).*

### Per-Ticker Status (March 24 Close)

| Ticker | Entry | Mar 24 Close | Return | FC 10d target | Stop | Status |
|--------|-------|--------------|--------|---------------|------|--------|
| **CVX** | $196.60 | $206.79 | **+5.18%** | $199.46 | $196.60 | ✅ Well past target |
| **VWAGY** | $10.30 | $10.35 | **+0.49%** | $10.39 | $9.89 | Recovered above entry (exited Mar 20) |
| **0388.HK** | HKD 401.40 | HKD 391.60 | -2.44% | HKD 403.02 | HKD 385.34 | Recovered above stop — if still held |
| **UNH** | $283.98 | $272.28 | -4.12% | $288.74 | $270.81 | Recovered above global stop — if still held |
| **2382.HK** | HKD 56.40 | HKD 52.90 | **-6.21%** | HKD 57.94 | HKD 54.14 | STOP!! Still below stop |

**Decision update:** UNH and 0388.HK exits were NOT executed. Both recovered above their stop-loss levels by the March 24 close — UNH to $272.28 (above global stop $270.81) and 0388.HK to HKD 391.60 (above stop HKD 385.34). Continuing to hold.

**2382.HK:** Still below stop at HKD 52.90 vs HKD 54.14 — exit at March 25 open.

**VWAGY:** Already exited March 20 — held as cash.

**CVX:** +5.18% and still climbing past its $199.46 10-day target.

### Active positions going into March 25

| Position | Shares | Mar 24 Close | Value | Status |
|----------|--------|--------------|-------|--------|
| UNH | 7.043 | $272.28 | ~$1,917 | Hold — above global stop |
| CVX | 10.173 | $206.79 | ~$2,104 | Hold — stop at entry |
| 0388.HK | 38.8 sh | HKD 391.60 | ~$1,950 USD | Hold — above stop |
| VWAGY (cash) | — | — | ~$1,984 | Exited Mar 20 |
| 2382.HK | 276 sh | HKD 52.90 | — | Exit Mar 25 open |

3 days remain (Mar 25, 26, 27).

---

## March 23 Update — Day 6 ⚠ THREE STOPS HIT

*Day 6 of 10. Severe drawdown. 2382.HK, UNH, and 0388.HK have all breached their stop-losses.*

### Actual Performance (Days 1–6)

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | — | +0.65% | +0.37% |
| 2 | Mar 17 | — | +0.97% | +0.40% |
| 3 | Mar 18 | — | +0.28% | +0.38% |
| 4 | Mar 19 | — | -0.21% | +0.26% |
| 5 | Mar 20 | $9,827 | -1.73% | +0.71% |
| **6** | **Mar 23** | **$9,745** | **-2.55%** | +0.60% |

*Script tracks all 5 original positions (including VWAGY, which was recommended for exit at Mar 20 open).*

### Per-Ticker Status (March 23 Close)

| Ticker | Entry | Mar 23 Close | Return | FC 10d target | Stop | Status |
|--------|-------|--------------|--------|---------------|------|--------|
| **CVX** | $196.60 | $205.21 | **+4.38%** | $199.46 | $196.60 | ✅ Well past target |
| **VWAGY** | $10.30 | $10.22 | -0.78% | $10.39 | $9.89 | OK (if still held; should have been exited Mar 20) |
| **2382.HK** | HKD 56.40 | HKD 52.70 | **-6.56%** | HKD 57.94 | HKD 54.14 | ⚠ **STOP HIT** |
| **UNH** | $283.98 | $269.54 | **-5.08%** | $288.74 | $270.81 | ⚠ **STOP HIT** |
| **0388.HK** | HKD 401.40 | HKD 382.60 | **-4.68%** | HKD 403.02 | HKD 385.34 | ⚠ **STOP HIT** |

### ⚠ Trading Decisions — March 24 Open

**2382.HK: EXIT.** HKD 52.70 is 2.68% below the HKD 54.14 stop. The week-2 thesis (HKD 56.84→57.94) has completely failed. Exit at March 24 HK open. Loss: ~-6.56% on $2,000 position = ~-$131.

**UNH: EXIT.** $269.54 is well below the $270.81 stop. Three consecutive plans of UNH underperformance. Exit at March 24 US open. Loss: ~-5.08% on $2,000 = ~-$102.

**0388.HK: EXIT.** HKD 382.60 is below the HKD 385.34 stop. Weakest signal from the start (+0.40%) — confirmed unreliable. Exit at March 24 HK open. Loss: ~-4.68% on $2,000 = ~-$94.

**CVX: Hold, stop at entry ($196.60).** CVX at $205.21 is the only position working. Same pattern as US plan — CVX dramatically outperforming model targets.

**VWAGY: Exit if not already done.** Was recommended for exit at Mar 20 open. Still marginally above stop but thesis never materialized. Exit at March 24 US open.

### Post-Exit Portfolio Estimate (March 24 open)

| Position | Value | % |
|----------|-------|---|
| CVX | ~$2,088 | 21.4% |
| Cash (2382.HK + UNH + 0388.HK + VWAGY exits) | ~$7,657 | 78.6% |
| **Total** | **~$9,745** | |

*Estimated losses: 2382.HK ~$131, UNH ~$102, 0388.HK ~$94, VWAGY ~$16. Total realized ~$343 loss on $8,000 deployed. CVX gain ~$88 offsets partially. Net: ~-$255 on $10,000.*

---

## March 20 Update — Day 5

*Day 5 of 10 — planned checkpoint. 2382.HK narrowly above stop. VWAGY exit executed.*

### Actual Performance (Days 1–5)

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | — | +0.65% | +0.37% |
| 2 | Mar 17 | — | +0.97% | +0.40% |
| 3 | Mar 18 | — | +0.28% | +0.38% |
| 4 | Mar 19 | — | -0.21% | +0.26% |
| **5** | **Mar 20** | **$9,827** | **-1.73%** | +0.71% |

### Per-Ticker Status (March 20 Close)

| Ticker | Entry | Mar 20 Close | Return | 5d FC (target) | Stop | Status |
|--------|-------|--------------|--------|----------------|------|--------|
| CVX | $196.60 | $201.73 | **+2.61%** | $199.36 | $196.60 | ✅ Past 10d target |
| 2382.HK | HKD 56.40 | HKD 54.45 | **-3.46%** | HKD 56.58 | HKD 54.14 | ⚠ Near stop (0.57% above) |
| UNH | $283.98 | $275.59 | **-2.95%** | $288.55 | $270.81 | Watch — well below target |
| 0388.HK | HKD 401.40 | HKD 396.00 | -1.35% | HKD 398.80 | HKD 385.34 | Hold — above stop |
| VWAGY | $10.30 | $9.94 | -3.50% | $10.33 | $9.89 | Exited at open per plan |

**VWAGY exit:** Exited at March 20 open per March 20 checkpoint plan. Closed position at ~$9.94 close equivalent. Proceeds ~$1,984 held as cash.

**2382.HK critical:** HKD 54.45 is only HKD 0.31 above the HKD 54.14 stop. Week-2 thesis requires recovery immediately. Failed to reach the HKD 56.58 Day-5 target by a wide margin. If 2382.HK opens below HKD 54.14 on March 23, exit.

**UNH:** $275.59 vs $288.55 5-day target — missed by $13. Pattern of persistent underperformance across 3 consecutive plans. Stop at $270.81 is the last line.

---

## March 19 Update — Day 4 (5-Day Checkpoint)

**Portfolio: -0.19%** ($9,981) — recovered from -0.30% yesterday; tomorrow is the planned Mar 20 checkpoint.

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | — | +0.65% | +0.37% |
| 2 | Mar 17 | — | +0.97% | +0.40% |
| 3 | Mar 18 | — | +0.28% | +0.38% |
| **4** | **Mar 19** | **$9,981** | **-0.19%** | +0.26% |

| Ticker | Entry | Mar 19 Close | Return | 5d FC (target) | Stop | Status |
|--------|-------|--------------|--------|----------------|------|--------|
| CVX | $196.60 | $201.44 | **+2.46%** | $199.36 | $188.95 | ✅ Past 10d target |
| 2382.HK | HKD 56.40 | HKD 56.40 | **0.00%** | HKD 56.58 | HKD 54.14 | OK — flat |
| 0388.HK | HKD 401.40 | HKD 398.60 | -0.70% | HKD 398.80 | HKD 385.34 | Near 5d target |
| UNH | $283.98 | $280.44 | **-1.25%** | $288.55 | $270.81 | Lagging |
| VWAGY | $10.30 | $10.15 | **-1.46%** | $10.33 | $9.89 | ⚠ Watch |

### March 20 Checkpoint Decisions

**CVX: Consider partial profit.** At +2.46%, CVX has already exceeded its 10-day model target ($199.46) by Day 4. Tighten stop to entry ($196.60). Same pattern as previous plan — CVX front-loads its gain.

**VWAGY: Exit or reduce.** -1.46% over 4 days, model forecast still sees +0.87% by March 27 but signal was never strong (only +0.87% predicted from context). The plan stated: *"if VWAGY not recovered above $10.30 by March 20, consider exiting."* It hasn't. Recommended: exit VWAGY at March 20 open, redeploy into cash or add to CVX.

**2382.HK: Hold — week-2 thesis intact.** Recovered from -0.80% to flat. Model sees the main move in week 2 (HKD 56.84→57.94 from March 23–27). No action.

**0388.HK: Hold.** At -0.70%, essentially tracking the model's 5-day path ($398.60 actual vs $398.80 target). Weakest signal (+0.40% predicted) but still above stop.

**UNH: Hold with caution.** -1.25% vs +2.36% 10-day forecast — significantly lagging. Stop at $270.81 provides $9.63 cushion. Both plans show UNH underperforming. Watch for recovery above $283 (entry) by March 23.

---

## March 18 Update — Day 3

**Portfolio: -0.30%** ($9,970) — turned negative after peaking at +0.97% on Day 2

*Note: HK tickers close ~12h ahead of US; global tracker shows HK Day 3 data alongside US Day 3 data.*

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | — | +0.65% | +0.37% |
| 2 | Mar 17 | — | **+0.97%** | +0.40% |
| **3** | **Mar 18** | **$9,970** | **-0.30%** | +0.38% |

| Ticker | Entry | Mar 18 Close | Return | vs 10d FC | Stop | Status |
|--------|-------|--------------|--------|-----------|------|--------|
| CVX | $196.60 | $198.61 | **+1.02%** | ahead | $188.95 | OK |
| UNH | $283.98 | $284.33 | +0.12% | lagging | $270.81 | OK |
| 2382.HK | HKD 56.40 | HKD 55.95 | **-0.80%** | below | HKD 54.14 | OK |
| 0388.HK | HKD 401.40 | HKD 398.20 | **-0.80%** | below | HKD 385.34 | OK |
| VWAGY | $10.30 | $10.19 | **-1.07%** | lagging | $9.89 | OK |

**Key reversal — HK tickers:** 2382.HK fell from +1.86% peak (Day 2) to -0.80% (Day 3). A 2.7% intraday reversal. Still 1.81 HKD above stop-loss — no action triggered. The model's 10-day target remains HKD 57.94 (+3.56% from current $55.95).

**VWAGY continues weak:** -1.07% over 3 days vs model forecast of +0.87% by March 27. Consider reducing if VWAGY doesn't recover above $10.30 (entry) by March 20 checkpoint.

**CVX is the anchor:** +1.02% and pulling ahead of model path — the one consistent positive across both plans.

---

## March 17 Update — Day 2

**Portfolio: +0.86%** ($10,086) — beating forecast (+0.40%) by **+0.45pp**

| Ticker | Entry | Mar 17 Close | Return | vs 10d FC |
|--------|-------|--------------|--------|-----------|
| 2382.HK | HKD 56.40 | HKD 57.45 | **+1.86%** | ahead (FC 10d: +2.73%) |
| UNH | $283.98 | $287.57 | **+1.26%** | ahead (FC 10d: +2.36%) |
| CVX | $196.60 | $197.97 | +0.70% | ahead (FC 10d: +1.34%) |
| 0388.HK | HKD 401.40 | HKD 404.40 | +0.75% | above 10d FC (+0.40%) already ✓ |
| VWAGY | $10.30 | $10.27 | **-0.29%** | lagging (FC 10d: +0.87%) |

- All positions **well above stop-losses** (>5% cushion each)
- 2382.HK and 0388.HK continue to outrun the day-by-day model path (HK market strong)
- 0388.HK already exceeds its 10-day price target of HKD 403.02 — consider taking partial profit or tightening stop to HKD 401.40 (entry)
- VWAGY is the only negative position; -0.29% over 2 days, still above the $9.89 stop

---

## March 16 Update — Day 1

**Portfolio: +0.65%** ($10,065) — beat forecast (+0.37%) by **+0.28pp**

| Ticker | Entry | Mar 16 Close | Return | Note |
|--------|-------|--------------|--------|------|
| 2382.HK | HKD 56.40 | HKD 57.35 | +1.68% | Strong HK open |
| 0388.HK | HKD 401.40 | HKD 406.20 | +1.20% | Above Day-1 forecast |
| UNH | $283.98 | $285.49 | +0.53% | Below Day-1 forecast |
| CVX | $196.60 | $196.84 | +0.12% | On forecast |
| VWAGY | $10.32 | $10.29 | -0.29% | Soft open |

---

## Daily Update Procedure

Run each trading day after market close:

```shell
cd /c/Users/ylchen/workspace/timesfm
source .venv/Scripts/activate
# Re-forecast from latest context (adapt sim_2week_live.py for global tickers)
python global_sim/track_2week.py
```

After running, add a `## March XX Update` section above this section with:
- Actual prices vs model forecasts
- Stop-loss status per ticker
- Any trade decisions for next day

---

## Notes on Model Selection

| Ticker | Model Used | Why |
|--------|-----------|-----|
| 2382.HK | Glia_China (run_glia_2a) | +5.75pp rolling improvement for China, -8.9pp on 2382.HK specifically |
| 0388.HK | Pretrained baseline | Rolling eval: baseline preferred for HKEX (exchange operator) |
| UNH, BA | Run4 (US default) | Standard US finetuned checkpoint |
| CVX | Glia3a | Per-ticker selective routing from US experiments |
| NVDA, GS, KO | Glia4b | Per-ticker selective routing from US experiments |
| UL, EADSF, VWAGY | Run1_EU | +0.20pp Europe rolling improvement |
| All Japan | Pretrained | Finetuning uniformly hurts all Japan tickers (+1.95pp regression) |

---

*Report generated: March 14, 2026*
*Forecast model: TimesFM 2.5 (200M) — regional finetuned checkpoints*
*Forecast data: global_sim/forecast_mar14.json*
*Entry prices: estimated from March 13, 2026 close (actual entries = March 16 open)*
