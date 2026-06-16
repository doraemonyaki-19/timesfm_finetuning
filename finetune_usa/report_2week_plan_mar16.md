# 2-Week US Investment Plan — $10,000

**Date:** March 17, 2026 (forecasts from March 13 close; entry March 16 open)
**Horizon:** 10 trading days (March 16 – March 27, 2026)
**Universe:** 18 clean tickers — Sets A, B, C (zero training-data overlap)
**Strategy:** SelectiveTop5 — top 5 ranked tickers by predicted 10-day return
**Models:** Run4 (checkpoints/best), Glia3a (CVX), Glia4b (NVDA/GS/KO)

> ⚠ **Disclaimer:** This plan is based on a machine-learning time-series model. It is not professional financial advice. The model has no knowledge of upcoming earnings, macro events, or news. All predicted price levels carry uncertainty. Past simulation performance does not guarantee future results.

---

## Daily Update Procedure

```shell
cd /c/Users/ylchen/workspace/timesfm
source .venv/Scripts/activate
python scripts/sim_2week_live_mar16.py
```

Outputs:
- **`finetune_expts/sim_2week_live_mar16.json`** — daily portfolio values, per-ticker breakdown
- **`finetune_expts/sim_2week_live_mar16.png`** — portfolio chart (actual vs forecast)

---

## March 27 Update — Day 10 — FINAL ⚠ META Crash, CVX Surge

*Plan complete. Massive divergence on final two days — META crashed -16.82%, CVX surged +7.40%.*

### Full 10-Day Performance

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | $9,986 | -0.14% | -0.69% |
| 2 | Mar 17 | $10,034 | +0.34% | -0.58% |
| 3 | Mar 18 | $9,991 | -0.09% | -0.56% |
| 4 | Mar 19 | $9,966 | -0.34% | -0.56% |
| 5 | Mar 20 | $9,938 | -0.62% | -0.01% |
| 6 | Mar 23 | $10,015 | +0.15% | +0.01% |
| 7 | Mar 24 | $10,061 | +0.61% | +0.24% |
| 8 | Mar 25 | $10,062 | +0.62% | +0.49% |
| 9 | Mar 26 | $9,868 | -1.32% | +0.63% |
| **10** | **Mar 27** | **$9,654** | **-3.46%** | +0.52% |

*Script baseline — all 5 original positions held throughout (META not modeled as exited).*

### Final Per-Ticker Outcomes

| Ticker | Entry | Final | Return | Pred Return | Delta | Stop executed? |
|--------|-------|-------|--------|-------------|-------|----------------|
| **CVX** | $196.60 | $211.15 | **+7.40%** | +1.45% | +5.95pp | No — winner |
| **MS** | $156.46 | $158.39 | **+1.23%** | -0.43% | +1.66pp | No — held |
| **BAC** | $47.12 | $46.97 | -0.32% | +0.96% | -1.28pp | No — near flat |
| **UNH** | $283.98 | $259.02 | **-8.79%** | +1.68% | -10.47pp | Stop hit Mar 23, exit Mar 25 |
| **META** | $632.00 | $525.72 | **-16.82%** | -1.05% | -15.76pp | Stop hit Mar 19, exit Mar 20 |

### Stop-Loss Value Analysis

The stop-loss discipline was the defining factor:

| Scenario | Final Value | Return |
|----------|------------|--------|
| Script baseline (nothing exited) | $9,654 | **-3.46%** |
| META exited at stop (~$1,878), UNH held | $9,869 | -1.31% |
| META + UNH exited at stops | $9,961 | **-0.39%** |
| If META never entered (gap-up avoided) | ~$10,120 | ~+1.20% |

Exiting META at its $607 stop on March 19 **saved ~$337** vs holding to $525. Exiting UNH at $272 on March 25 **saved ~$90** vs $259 final. Stop-loss rules outperformed in both cases.

---

## March 24 Update — Day 7 — Recovery

*Day 7 of 10. Market-wide recovery. CVX and MS continue to run. UNH recovered but still below stop.*

### Actual Performance (Days 1–7)

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | $9,986 | -0.14% | -0.69% |
| 2 | Mar 17 | $10,034 | +0.34% | -0.58% |
| 3 | Mar 18 | $9,991 | -0.09% | -0.56% |
| 4 | Mar 19 | $9,966 | -0.34% | -0.56% |
| 5 | Mar 20 | $9,938 | -0.62% | -0.01% |
| 6 | Mar 23 | $10,015 | +0.15% | +0.01% |
| **7** | **Mar 24** | **$10,061** | **+0.61%** | +0.24% |

*Script baseline includes META and UNH as if still held.*

### Per-Ticker Status (March 24 Close)

| Ticker | Entry | Mar 24 Close | Return | FC 10d | Stop | Status |
|--------|-------|--------------|--------|--------|------|--------|
| **MS** | $156.46 | $165.87 | **+6.01%** | -0.43% | $150.20 | ✅ Far above model — exit or trail stop |
| **CVX** | $196.60 | $206.79 | **+5.18%** | +1.45% | $196.60 | ✅ Still climbing — stop at entry |
| **BAC** | $47.12 | $48.14 | **+2.16%** | +0.96% | $45.24 | ✅ Above 10d target already |
| **UNH** | $283.98 | $272.28 | -4.12% | +1.68% | $272.62 | ⚠ Still below stop — exit at open |
| **META** | $632.00 | $592.92 | -6.18% | -1.05% | $607.10 | Exited Mar 20 (cash) |

**UNH note:** Recovered from $269.54 (Day 6) to $272.28 — but still $0.34 below the $272.62 stop. Exit rule stands: exit at March 25 open. Three plans in a row of UNH underperformance.

**CVX + MS note:** Both continuing well past model targets. CVX model peak was $201.05 (Day 8 forecast); actual is $206.79 on Day 7. MS model target was -0.43% from entry; actual is +6.01%. Energy and financials outperforming significantly.

### Adjusted Portfolio (META + UNH exited, Mar 25 open)

| Position | Value (Mar 24 close) | % |
|----------|----------------------|---|
| CVX | ~$2,104 | 26.6% |
| MS | ~$2,120 | 26.8% |
| BAC | ~$2,043 | 25.8% |
| Cash (META ~$1,878 + UNH ~$1,917) | ~$3,795 | — |
| **Total** | **~$10,062** | |

3 days remain (Mar 25, 26, 27). With UNH exited, CVX/MS/BAC carry the portfolio.

---

## March 23 Update — Day 6 ⚠ UNH STOP TRIGGERED

*Day 6 of 10. Script baseline tracks all 5 original positions (META not modeled as exited). UNH has now hit its stop-loss.*

### Actual Performance (Days 1–6)

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | $9,986 | -0.14% | -0.69% |
| 2 | Mar 17 | $10,034 | +0.34% | -0.58% |
| 3 | Mar 18 | $9,991 | -0.09% | -0.56% |
| 4 | Mar 19 | $9,966 | -0.34% | -0.56% |
| 5 | Mar 20 | $9,938 | -0.62% | -0.01% |
| **6** | **Mar 23** | **$10,015** | **+0.15%** | +0.01% |

*Script baseline includes META as if still held. Adjusted portfolio (META → cash after Mar 20 stop) ≈ $9,982 (-0.18%).*

### Per-Ticker Status (March 23 Close)

| Ticker | Entry | Mar 23 Close | Return | FC 10d (from entry) | Stop | Status |
|--------|-------|--------------|--------|---------------------|------|--------|
| **CVX** | $196.60 | $205.21 | **+4.38%** | +1.45% | $196.60 | ✅ Well past target — stop tightened to entry |
| **MS** | $156.46 | $164.32 | **+5.02%** | -0.43% | $150.20 | ✅ Far above model — take profit |
| **BAC** | $47.12 | $47.52 | +0.85% | +0.96% | $45.24 | OK — tracking model |
| **UNH** | $283.98 | $269.54 | **-5.08%** | +1.68% | **$272.62** | ⚠ **STOP HIT** |
| **META** | $632.00 | $604.06 | -4.42% | -1.05% | $607.10 | Exited Mar 20 (stop triggered Mar 19) |

### ⚠ Trading Decision — March 24 Open

**UNH: EXIT at market open.** Close at $269.54 is well below the $272.62 stop-loss. Exit UNH at March 24 open. Proceeds ≈ $1,898 (loss of ~$102 on the $2,000 position).

**MS: Consider full profit take.** At +5.02%, MS has run far beyond the model's target (-0.43% from entry). Momentum has front-loaded all expected return and then some. Consider closing the position at March 24 open near $164.

**CVX: Hold, stop at entry ($196.60).** Week-2 surge is materializing — $205.21 vs peak target $201.05. Model was right directionally but underestimated magnitude. Stop tightened to entry protects the full +4.38% gain.

**BAC: Hold.** Tracking model path cleanly — $47.52 vs $47.25 Day-6 forecast.

### Updated Portfolio (post-UNH exit estimate, March 24 open)

| Position | Shares | Est. Price | Value | % |
|----------|--------|------------|-------|---|
| CVX | 10.173 | ~$205 | ~$2,088 | 26.3% |
| MS | 12.783 | ~$164 | ~$2,096 | 26.4% |
| BAC | 42.445 | ~$47.52 | ~$2,017 | 25.4% |
| Cash (META exit) | — | — | ~$1,878 | 23.6% |
| UNH exit proceeds | — | — | ~$1,898 | — |
| **Total (3 pos + 2 cash)** | | | **~$9,977** | |

*If MS also exited: ~$9,977 → 2 positions + 3 cash tranches, heavily concentrated in CVX.*

---

## March 20 Update — Day 5 — META Exited

*Day 5 of 10. META stop executed at open. CVX and MS both well above model targets.*

### Actual Performance (Days 1–5)

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | $9,986 | -0.14% | -0.69% |
| 2 | Mar 17 | $10,034 | +0.34% | -0.58% |
| 3 | Mar 18 | $9,991 | -0.09% | -0.56% |
| 4 | Mar 19 | $9,966 | -0.34% | -0.56% |
| **5** | **Mar 20** | **$9,938** | **-0.62%** | -0.01% |

### Per-Ticker Status (March 20 Close)

| Ticker | Entry | Mar 20 Close | Return | FC 5d (target) | Stop | Status |
|--------|-------|--------------|--------|----------------|------|--------|
| **CVX** | $196.60 | $201.73 | **+2.61%** | $199.36 | $196.60 | ✅ Past 10d target — stop moved to entry |
| **MS** | $156.46 | $161.47 | **+3.20%** | $155.16 | $150.20 | ✅ Far above model target |
| **BAC** | $47.12 | $47.16 | +0.08% | $47.11 | $45.24 | OK — on plan |
| **UNH** | $283.98 | $275.59 | **-2.95%** | $288.55 | $272.62 | Watch — lagging |
| **META** | $632.00 | $593.66 | -6.07% | $617.87 | $607.10 | Exited at open (stop triggered Mar 19) |

**META exit note:** META fell to $593.66 on Day 5 — well below the $607.10 stop. The stop was triggered by March 19's $606.70 close; exit at March 20 open was the correct action. Estimated proceeds ~$1,878 → held as cash.

**UNH note:** At -2.95% and $275.59, UNH is above the $272.62 stop but lagging the model's $288.55 Day-5 target by $13. Watch closely — model now sees recovery from here through March 27.

---

## March 19 Update — Day 4 ⚠ STOP TRIGGERED on META

*Day 4 of 10. META has hit its stop-loss. CVX has exceeded its 10-day target.*

### Actual Performance (Days 1–4)

| Day | Date | Portfolio | Return | Forecast |
|-----|------|-----------|--------|----------|
| 1 | Mar 16 | $9,986 | -0.14% | -0.69% |
| 2 | Mar 17 | $10,034 | +0.34% | -0.58% |
| 3 | Mar 18 | $9,991 | -0.09% | -0.56% |
| **4** | **Mar 19** | **$9,966** | **-0.34%** | -0.56% |

### Per-Ticker Status (March 19 Close)

| Ticker | Entry | Mar 19 Close | Return | FC 10d (from entry) | Stop | Status |
|--------|-------|--------------|--------|---------------------|------|--------|
| **CVX** | $196.60 | $201.44 | **+2.46%** | +1.45% | $188.95 | ✅ Past 10d target |
| **MS** | $156.46 | $158.55 | **+1.34%** | -0.43% | $150.20 | ✅ Outperforming |
| **BAC** | $47.12 | $47.01 | -0.23% | +0.96% | $45.24 | Hold |
| **UNH** | $283.98 | $280.44 | **-1.25%** | +1.68% | $272.62 | Watch |
| **META** | $632.00 | $606.70 | **-4.00%** | -1.05% | **$607.10** | ⚠ **STOP HIT** |

### ⚠ Trading Decision — March 20 Open

**META: EXIT at market open.** Close at $606.70 is below the $607.10 stop-loss. The plan rule is clear — exit META at March 20 open. Proceeds ≈ $1,920 (loss of ~$80 on the $2,000 position).

**CVX: Hold, tighten stop.** CVX at $201.44 has already exceeded the 10-day target of $199.46 by Day 4 — same pattern as the previous plan where CVX surged past target early. Tighten stop to $196.60 (entry) to protect gains. Consider partial profit if CVX reaches $203.

**Redeployment of META proceeds (~$1,920):**
- Option 1 — Add to UNH: UNH at -1.25% is lagging but stop ($272.62) not threatened. Model still sees recovery to $288. Add $1,000 → ~3.6 shares. Risky given UNH's structural weakness.
- Option 2 — Add to CVX: Already past target, but energy momentum may continue. Add $960 → ~4.8 shares. Doubles down on the one clear winner.
- **Option 3 — Cash (recommended):** With 6 days remaining and only CVX/MS generating real gains, park the $1,920 in cash. Reduces deployed capital to $6,080 but protects against further drawdown.

### Updated Portfolio (post-META exit, March 20 open estimate)

| Position | Shares | Price (est.) | Value | % |
|----------|--------|--------------|-------|---|
| UNH | 7.043 | ~$280 | ~$1,972 | 19.8% |
| BAC | 42.445 | ~$47.01 | ~$1,994 | 20.0% |
| CVX | 10.173 | ~$201 | ~$2,045 | 20.5% |
| MS | 12.783 | ~$158 | ~$2,020 | 20.3% |
| **Cash** | — | — | **~$1,920** | **19.3%** |
| **Total** | | | **~$9,951** | |

---

## March 18 Update — Day 3

*Day 3 of 10. Portfolio slipped back below breakeven.*

### Actual Performance (Days 1–3)

| Day | Date | Portfolio | Return | Forecast | Gap |
|-----|------|-----------|--------|----------|-----|
| 1 | Mar 16 | $9,986 | -0.14% | -0.69% | +0.55pp |
| 2 | Mar 17 | $10,034 | +0.34% | -0.58% | +0.92pp |
| **3** | **Mar 18** | **$9,991** | **-0.09%** | -0.56% | **+0.47pp** |

### Per-Ticker Status (March 18 Close)

| Ticker | Entry | Mar 18 Close | Return | FC path (Day 3) | Status |
|--------|-------|--------------|--------|-----------------|--------|
| **MS** | $156.46 | $158.93 | **+1.58%** | $154.88 | Far ahead of forecast ↑↑ |
| **CVX** | $196.60 | $198.61 | **+1.02%** | $197.43 | Ahead of path ↑ |
| **UNH** | $283.98 | $284.33 | +0.12% | $286.57 | Lagging — model expected more |
| **BAC** | $47.12 | $46.83 | -0.62% | $46.99 | Slightly below path |
| **META** | $632.00 | $615.68 | **-2.58%** | $614.20 | On model path, but entry was wrong |

**META note:** Actual ($615.68) is tracking almost exactly on the Day-3 model path ($614.20). The model is not wrong about META's direction — the problem was the $632 gap-up entry. From current levels the model's 10-day target ($625.34) now implies **+1.57% upside**. Thesis is partially restored if willing to hold from current levels rather than entry.

**MS note:** At $158.93 vs Day-3 forecast $154.88 (+$4.05 above path). Significantly outrunning the model. Consider taking partial profit if MS reaches $161.

### Mar 19 Trading Decision
- **META**: Hold — current price $615.68 is above the model path and the 10-day target ($625.34) now shows +1.57% upside from here. Stop remains $607.10.
- **MS**: Partial profit consideration at $161 (+3% from entry).
- **CVX**: Hold — week-2 surge (peak $201.05 on March 25) not yet realized.
- **UNH, BAC**: Hold — below model path but stop-losses not threatened.

---

## March 17 Update — Day 2

*Day 2 of 10. Positions entered March 16 open.*

### Actual Performance (Days 1–2)

| Day | Date | Portfolio | Return | Forecast | Gap |
|-----|------|-----------|--------|----------|-----|
| 1 | Mar 16 | $9,986 | **-0.14%** | +0.39% | -0.53pp |
| **2** | **Mar 17** | **$10,034** | **+0.34%** | +0.51% | **-0.17pp** |

Portfolio recovered through Day 2 as UNH, CVX, and MS all gained. META's Day 1 gap-up reversal (-0.72% on Day 1) has been the primary drag; it partially recovered Day 2.

### Per-Ticker Status (March 17 Close)

| Ticker | Entry (Mar 16 open) | Mar 17 Close | Return | FC 10d (from entry) | Stop-Loss | Status |
|--------|---------------------|--------------|--------|---------------------|-----------|--------|
| **UNH** | $283.98 | $287.57 | **+1.26%** | +1.68% | $272.62 | OK — on track |
| **META** | $632.00 | $622.66 | **-1.48%** | -1.05% | $607.10 | ⚠ gap-up trap |
| **BAC** | $47.12 | $47.28 | +0.34% | +0.95% | $45.24 | OK |
| **CVX** | $196.60 | $197.97 | +0.70% | +1.45% | $188.74 | OK |
| **MS** | $156.46 | $157.83 | **+0.88%** | -0.43% | $150.20 | Ahead of FC ✓ |

**Key finding — META gap-up:** META opened March 16 at $632.00 (+3.1% above the March 13 context price of $613.18). The model's $625 target is now *below* our entry price. This converts a +1.98% forecast (from context) into a **-1.05% expected return from entry**. The thesis has inverted. See trading decisions below.

**MS outperformance:** MS at +0.88% in 2 days already exceeds its full 10-day model target (+0.59% from Mar 13 context). The model systematically underestimated MS this cycle.

---

## March 16 Update — Day 1

| Ticker | Entry (Mar 16 open) | Mar 16 Close | Return |
|--------|---------------------|--------------|--------|
| UNH | $283.98 | $285.49 | +0.53% |
| META | $632.00 | $627.45 | -0.72% ⚠ |
| BAC | $47.12 | $47.06 | -0.13% |
| CVX | $196.60 | $196.84 | +0.12% |
| MS | $156.46 | $155.70 | -0.49% |
| **Portfolio** | | | **-0.14%** |

META opened sharply above the model's reference context (+3.1% gap), immediately invalidating the +1.98% forecast from entry.

---

## Previous Plan Summary (Mar 3–16 Final: -0.47%)

The March 3–16 plan ended at -0.47% ($9,953). Key lessons carried forward:

- **GS, BAC, UNH all failed** — financial sector took macro/tariff shock averaging -8pp vs forecast
- **CVX (+3.32%) and NVDA (+2.66%) were the only winners** — energy and semiconductors outperformed
- **Model's 10-day financial sector signals were overconfident** in a volatile macro regime
- **Gap-up entries are dangerous** — META's Day 1 situation repeats the structural problem where a stock opens above the model's context, immediately reducing expected returns

---

## Full 18-Ticker Ranking (from March 13 context)

| Rank | Ticker | Set | Model | Mar 13 Close | 10d Target | Pred Ret | Mar 16 Open | From Entry |
|------|--------|-----|-------|--------------|------------|----------|-------------|------------|
| 1 | **UNH** | A | Run4 | $282.09 | $288.74 | +2.36% | $283.98 | **+1.68%** ✓ |
| 2 | **META** | C | Run4 | $613.18 | $625.34 | +1.98% | $632.00 | **-1.05%** ✗ gap |
| 3 | **BAC** | B | Run4 | $46.72 | $47.57 | +1.82% | $47.12 | **+0.95%** ✓ |
| 4 | **CVX** | A | Glia3a | $196.82 | $199.46 | +1.34% | $196.60 | **+1.45%** ✓ |
| 5 | **MS** | C | Run4 | $154.87 | $155.79 | +0.59% | $156.46 | **-0.43%** ✗ gap |
| 6 | MSFT | B | Run4 | $395.55 | $397.87 | +0.59% | $398.07 | **-0.05%** flat |
| 7 | TGT | C | Run4 | $117.34 | $117.53 | +0.17% | — | — |
| 8 | NVDA | A | Glia4b | $180.25 | $180.52 | +0.15% | $182.97 | **-1.34%** ✗ gap |
| 9 | MRK | B | Run4 | $114.76 | $114.73 | -0.02% | — | — |
| 10 | LLY | C | Run4 | $985.08 | $983.34 | -0.18% | — | — |
| 11 | BA | A | Run4 | $209.89 | $209.35 | -0.26% | — | — |
| 12 | GS | A | Glia4b | $782.21 | $776.61 | -0.72% | — | — |
| 13 | COP | B | Run4 | $121.89 | $120.92 | -0.80% | — | — |
| 14 | KO | A | Glia4b | $77.34 | $76.58 | -0.99% | — | — |
| 15 | COST | B | Run4 | $1008.43 | $998.41 | -0.99% | — | — |
| 16 | SLB | C | Run4 | $44.72 | $44.24 | -1.07% | — | — |
| 17 | HON | C | Run4 | $234.50 | $231.78 | -1.16% | — | — |
| 18 | LMT | B | Run4 | $646.00 | $633.92 | -1.87% | — | — |

**Note on LMT:** LMT was the #1 performer in the 12-month backtest (+56%). The model now predicts -1.87% — do not fight the model signal in the short run.

---

## Portfolio Allocation

### Option A — Full Signal (Top 5 from March 13 context)

| Ticker | Company | Set | Amount | Entry (Mar 16 open) | Shares | 10d Target (from entry) | Model |
|--------|---------|-----|--------|---------------------|--------|-------------------------|-------|
| **UNH** | UnitedHealth Group | A | $2,000 | $283.98 | 7.042 | $288.74 (+1.68%) | Run4 |
| **META** | Meta Platforms | C | $2,000 | $632.00 | 3.165 | $625.34 (-1.05%) ⚠ | Run4 |
| **BAC** | Bank of America | B | $2,000 | $47.12 | 42.444 | $47.57 (+0.95%) | Run4 |
| **CVX** | Chevron | A | $2,000 | $196.60 | 10.173 | $199.46 (+1.45%) | Glia3a |
| **MS** | Morgan Stanley | C | $2,000 | $156.46 | 12.784 | $155.79 (-0.43%) ⚠ | Run4 |
| **Total** | | | **$10,000** | | | **+0.52% base case** | |

⚠ META and MS both opened above their model context price — adjusted expected returns from actual entry are negative.

### Option A (Adjusted) — Swap Gap-Up Tickers

Given META and MS gap-ups that inverted their forecasts, the cleaner composition is:

| Ticker | Amount | Entry | 10d from Entry | Note |
|--------|--------|-------|----------------|------|
| **UNH** | $2,500 | $283.98 | +1.68% | Top signal, overweight |
| **CVX** | $2,500 | $196.60 | +1.45% | Energy winner (2 plans running) |
| **BAC** | $2,000 | $47.12 | +0.95% | Clean signal, no gap |
| **META** | $2,000 | $632.00 | -1.05% | Hold if already entered, reduce if not |
| **MS** | $1,000 | $156.46 | -0.43% | Reduce; already ahead of target |

### Option B — Conservative (No gap-up risk)

Replace META and NVDA gap-up entries with just 3 validated positive signals + cash:

| Ticker | Amount | Entry | 10d from Entry | Note |
|--------|--------|-------|----------------|------|
| **UNH** | $2,500 | $283.98 | +1.68% | Largest remaining upside |
| **CVX** | $2,500 | $196.60 | +1.45% | Energy momentum confirmed |
| **BAC** | $2,000 | $47.12 | +0.95% | Marginal but positive |
| **Cash** | $3,000 | — | 0% | Defensive buffer |
| **Expected** | | | **+0.92% base** | on $7k deployed |

---

## 10-Day Price Forecasts (from March 13 context)

*These forecasts are anchored at March 13 close, not March 16 open. Return percentages in the table are from March 13 context.*

| Date | Day | UNH | META | BAC | CVX | MS | Portf (from ctx) |
|------|-----|-----|------|-----|-----|----|-----------------|
| Mar 16 | 1 | $286.21 | $615.84 | $46.76 | $197.05 | $154.66 | +0.39% |
| Mar 17 | 2 | $287.18 | $615.72 | $46.95 | $196.71 | $154.69 | +0.51% |
| Mar 18 | 3 | $286.57 | $614.20 | $46.99 | $197.43 | $154.88 | +0.53% |
| Mar 19 | 4 | $283.01 | $616.33 | $47.09 | $197.90 | $155.56 | +0.52% |
| **Mar 20** | **5** | **$288.55** | **$617.87** | **$47.11** | **$199.36** | **$155.16** | **+1.07%** |
| Mar 23 | 6 | $287.52 | $620.50 | $47.25 | $198.45 | $155.51 | +1.10% |
| Mar 24 | 7 | $288.40 | $621.34 | $47.32 | $199.53 | $155.55 | +1.33% |
| Mar 25 | 8 | $288.69 | $623.38 | $47.33 | $201.05 | $155.59 | +1.59% |
| Mar 26 | 9 | $289.02 | $624.77 | $47.47 | $200.98 | $155.76 | +1.73% |
| **Mar 27** | **10** | **$288.74** | **$625.34** | **$47.57** | **$199.46** | **$155.79** | **+1.62%** |

*Portfolio return = equal-weighted average of the 5 returns relative to March 13 closes.*
*Actual entry was at March 16 open — actual returns will differ from these figures (see Option A adjusted).*

---

## Per-Ticker Analysis

### UNH — +1.68% from entry (top signal for third consecutive plan)
The model predicted +6.85% last plan and got -0.44%. It now predicts +1.68% — a far more conservative signal. In the global plan (started same day), UNH is already +1.26% after 2 days. The model's current context reflects UNH's recovery from its March 12 floor (~$277) back to $282-288.

- **Key level:** $288.55 by March 20. The model expects 90% of the gain by Day 5.
- **Stop-loss:** $272.62 (~4% below entry $283.98)

### META — Caution (gap-up inverted the thesis)
The model predicted +1.98% from the March 13 context of $613. META opened at $632 (+3.1%), presumably on news or momentum not visible to the model. From our $632 entry, the model's $625 target is now a -1.05% loss.

- **If already entered:** Hold unless META closes below $624 (below 10-day model target). The gap may partially mean-revert back toward the target zone.
- **If not yet entered:** Do not enter. Use the $2,000 as cash or add to UNH/CVX.
- **Stop-loss:** $607.10 (~4% below entry)

### BAC — Clean +0.95% signal
BAC is the cleanest pick: no gap at open ($47.12 vs $46.72 context = only +0.85% move), positive 10-day signal, and Set B's best-performing strategy in the 12-month backtest. The model sees gradual climb from $47.12 to $47.57 over 10 days.

- **Key level:** $47.25 by March 20. If BAC is still below $47.00 on March 20, exit.
- **Stop-loss:** $45.24 (~4% below entry)

### CVX — Confirmed energy momentum (+1.45% from entry)
CVX was the biggest winner in the March 3 plan (+3.32%) and is now the most reliable signal in this plan (+1.45% from entry). The Glia3a model sees a two-wave pattern: flat in week 1 then surge week 2 (peak $201.05 on March 25).

- **Key level:** $199 by March 24–25. The week-2 surge is the main return driver.
- **Stop-loss:** $188.74 (~4% below entry $196.60). Given CVX's prior performance, consider tightening to $192 if oil retreats.

### MS — Already ahead of forecast (+0.88% in 2 days vs +0.59% 10-day target)
MS has already exceeded the model's 10-day target from context — it opened above the context price and has continued climbing. From our actual entry of $156.46, the remaining forecast is slightly negative (-0.43%). The momentum has front-loaded the expected return.

- **If already entered:** Consider taking partial profit if MS reaches $159 (overshot territory). Otherwise, hold — momentum stocks sometimes continue well past model targets.
- **Stop-loss:** $150.20 (~4% below entry)

---

## Trading Decisions (as of March 17 close)

### Immediate Actions

| Action | Ticker | Rationale |
|--------|--------|-----------|
| **Consider exiting** | META | 10-day target ($625) now below current price ($622.66). Remaining expected: -1.05%. Asymmetric risk. |
| **Hold — strong** | UNH | +1.26% in 2 days, tracking perfectly toward $288 target. |
| **Hold** | CVX | +0.70% in 2 days, week-2 surge expected. Don't exit before March 24–25 peak. |
| **Hold** | BAC | +0.34% in 2 days, gradual climb expected through March 27. |
| **Consider partial profit** | MS | Already +0.88% vs 10-day target of -0.43% from entry. Model was conservative. |

### If META Exited ($622.66 proceeds = ~$1,971)

Redeployment options:
- **Add to UNH** (best signal, +1.68% remaining): redeploy $1,000 → add ~3.5 shares
- **Add to CVX** (energy momentum, +1.45% remaining): redeploy $1,000 → add ~5 shares
- **Cash**: conservative approach, keep as buffer

---

## Stop-Loss Levels

| Ticker | Entry | Stop-Loss | Cushion (Day 2) | Trigger |
|--------|-------|-----------|-----------------|---------|
| UNH | $283.98 | $272.62 | +5.5% above stop | If UNH < $272 |
| META | $632.00 | $607.10 | -2.4% above stop | If META < $607 |
| BAC | $47.12 | $45.24 | +0.1% above stop | If BAC < $45.24 |
| CVX | $196.60 | $188.74 | +4.9% above stop | If CVX < $189 |
| MS | $156.46 | $150.20 | +5.1% above stop | If MS < $150 |

---

## Expected Portfolio Value

### Option A (UNH, META, BAC, CVX, MS from actual entries)

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **Base case (model targets from entry)** | **$10,052** | **+0.52%** |
| Bull — META recovers to $635, all others hit targets | $10,200 | +2.00% |
| Mid — META exits at $622, UNH/CVX/BAC hit targets | $10,108 | +1.08% |
| Flat — no movement from March 17 close | $10,034 | +0.34% |
| Bear — all equities down 4% from entry | $9,600 | -4.00% |

### Option B (UNH, CVX, BAC + cash from actual entries)

| Scenario | Portfolio Value | Return |
|----------|----------------|--------|
| **Base case (model targets from entry)** | **$10,092** | **+0.92%** |
| Bull — all three up 2× model signal | $10,184 | +1.84% |
| Flat | $10,000 | 0.00% |
| Bear — all equities down 4% | $9,728 | -2.72% |

*Option B's narrower deployed capital (70%) limits both upside and downside.*

---

## March 20 Checkpoint Plan

By end of Day 5 (March 20 close), compare against model's 5-day targets:

| Ticker | 5d Target (from ctx) | Decision if below | Decision if above |
|--------|----------------------|-------------------|-------------------|
| UNH | $288.55 | Exit if below $283 | Hold, target $288.74 |
| META | $617.87 (context) | Exit — confirms thesis failure | Partial profit if >$628 |
| BAC | $47.11 | Exit if below $46.50 | Hold through March 27 |
| CVX | $199.36 | Hold — week-2 surge expected | Consider tightening stop to entry |
| MS | $155.16 (context) | N/A — already exceeded | Consider full profit take if >$160 |

---

## Risk Factors

1. **META gap-up hazard:** Model's forecast was anchored at $613. The $632 entry has already consumed most of the margin. Any macro catalyst that sends META below $615 puts the position in significant loss territory.
2. **UNH recurrence risk:** Two consecutive plans have featured UNH as the top model pick. The 12-month backtest shows UNH at -35.5%. The short-term signal is real but the structural headwinds remain. Cap position size if uncomfortable.
3. **CVX oil reversal:** If WTI crude drops meaningfully, CVX can give back week-1 gains before the week-2 surge materializes. The stop at $188.74 is $8 below entry.
4. **BAC financial-sector contagion:** Last plan, BAC ended -2.94% due to the macro shock that hit all financials. Same risk applies this plan.
5. **MS momentum overshoot:** MS has already exceeded the model's target in 2 days. Momentum can reverse sharply when positioning is crowded.
6. **Short horizon noise:** TimesFM validated at 120-day horizon. 10-day forecasts carry more noise — the previous plan showed ~8pp gaps between predicted and actual returns in financial stocks.

---

## Notes on Model Selection

| Ticker | Model | Reason |
|--------|-------|--------|
| UNH, BA | Run4 (checkpoints/best) | Standard US finetuned default; best rolling MAPE for these tickers |
| CVX | Glia3a (run_glia_3a/best) | Per-ticker selective routing (+0.28pp improvement vs Run4) |
| NVDA, GS, KO | Glia4b (run_glia_4b/best) | Per-ticker selective routing |
| All Set B/C tickers | Run4 (checkpoints/best) | No per-ticker optimization done for Sets B/C |

---

*Report generated: March 17, 2026*
*Forecast model: TimesFM 2.5 (200M) — Run4 + Glia3a + Glia4b selective routing*
*Forecast data: finetune_expts/forecast_us_mar16.json (context: March 13, 2026)*
*Entry prices: actual March 16, 2026 open*
*Live simulation: scripts/sim_2week_live_mar16.py*
*Reference: finetune_expts/report_clean_simulation.md (18-ticker clean universe)*
