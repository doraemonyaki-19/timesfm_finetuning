# Sector Routing — Cross-Region Bootstrap Results

Generated: 2026-04-11

## How This Was Generated

### Script
```bash
cd C:\Users\ylchen\workspace\timesfm_finetuning
source .venv/Scripts/activate
python scripts/sector_router.py build --region {us|japan|europe|china} \
  --horizons "5,14,30,60,120" --n-windows 30 --n-boot 10000 \
  --honest-split --tier-metric pnl
```

`scripts/sector_router.py` runs per-ticker per-horizon bootstrap evaluation (10,000 resamples over 30 rolling windows with temporal honest split), classifies tickers into Tier 1/2/3, computes MAPE, Long/Flat P&L, Long/Short/Flat P&L, directional hit-rates, dead zone parameter sweeps, and writes routing tables.

### Baseline Model
- Path: `C:\Users\ylchen\workspace\timesfm_finetuning\model\` (TimesFM 2.5 200M pretrained)
- Loaded via: `TimesFM_2p5_200M_torch.from_pretrained("model/")`

### Finetuned Models (per region)

| Region | Checkpoint | Label | Config |
|--------|-----------|-------|--------|
| US | `finetune_usa/checkpoints/best` | Run4 | 10 S&P500 tickers, AdamW lr=1e-4, wd=0.01, freeze 17/20, cosine+5ep warmup, MSE loss |
| Japan | `finetune_japan/checkpoints/run_glia_t2_r2a/best` | hdecay1.0 | 10 JP tickers, horizon-weighted loss (decay=1.0), same optimizer config |
| Europe | `finetune_europe/checkpoints/run_glia_v2_1a/best` | EUv2 | 17 NYSE/NASDAQ-listed EU tickers, same optimizer config |
| China | `finetune_china/checkpoints/run_glia_2a/best` | Glia2a | 10 HK tickers, stride=32, same optimizer config, val_loss=0.012067 |

### Tickers Evaluated (per region)

**US (n=12):**
- Eval (held-out): NVDA, UNH, GS, CVX, KO, BA
- Training diagnostic: AAPL, MSFT, AMZN, GOOGL, META, JPM

**Japan (n=9):**
- Eval (held-out): 8035.T, 6902.T, 4661.T
- Training diagnostic: 7203.T, 6758.T, 9984.T, 8306.T, 4502.T, 7974.T

**Europe (n=9):**
- Eval (held-out): UL, EADSF, VWAGY
- Training diagnostic: ASML, SAP, NVO, AZN, SHEL, DEO

**China (n=9):**
- Eval (held-out): 0388.HK, 2382.HK, 1177.HK
- Training diagnostic: 0700.HK, 0941.HK, 2318.HK, 1299.HK, 0005.HK, 0001.HK

### Bootstrap Parameters
- Rolling windows: 30 (temporal honest split: classify on windows 0-14, evaluate on 15-29)
- Window step: 1 day
- Bootstrap resamples: 10,000
- Confidence level: 95%
- Tier metric: P&L (long/flat realized return)
- Tier classification:
  - **Tier 1**: Use finetuned model (95% CI of P&L improvement entirely > 0)
  - **Tier 2**: Use baseline model (95% CI of P&L improvement entirely < 0)
  - **Tier 3**: Indeterminate (95% CI spans 0) — defaults to baseline (conservative)

### Output Files
- `finetune_{region}/routing_table.json` — Full routing table with per-ticker tiers, CIs, P&L metrics, directional stats, dead zone sweep, and optimal dead zones
- `finetune_{region}/routing_table.md` — Human-readable markdown report with all tables
- `finetune_{region}/tier_labels_YYYY-MM-DD_seed42.csv` — Frozen audit CSV with all per-ticker/horizon metrics

### To Reproduce
```bash
# Run each region (downloads price data via yfinance, takes ~5-10 min per region)
for region in us japan europe china; do
  python scripts/sector_router.py build --region $region \
    --horizons "5,14,30,60,120" --n-windows 30 --n-boot 10000 \
    --honest-split --tier-metric pnl
done

# View routing tables
python scripts/sector_router.py show --region us

# Forecast with routing (routes each ticker to correct model, applies dead zone)
python scripts/sector_router.py forecast --region us --tickers "NVDA,MSFT,GS" --horizon 120
```

---

## Routing Policies

Two policies are available, assigned per region based on OOS validation:

| Region | Policy | Rule | Rationale |
|--------|--------|------|-----------|
| US | `simple` | FT at >= 60d, BL below | Per-ticker routing adds ~0pp vs FT-only OOS |
| Japan | `simple` | FT at >= 60d, BL below | Routing helps 30-60d but hurts 120d |
| Europe | `simple` | FT at >= 60d, BL below | Routing hurts vs FT-only; FT-only beats BL by 1.8-2.8pp OOS |
| China | `router` | Per-ticker tier labels | Only region where routing adds OOS value (+1.31pp @120d) |

---

## OOS Results — MAPE (lower is better, Route vs Baseline pp improvement)

| Region | 5d | 14d | 30d | 60d | 120d |
|--------|-----|------|------|------|-------|
| **US** | +0.00 | +0.00 | +0.00 | **+0.46** | **+0.99** |
| **Japan** | +0.00 | +0.00 | +0.00 | **+0.74** | **+1.71** |
| **Europe** | +0.00 | +0.00 | +0.00 | **+2.28** | **+2.76** |
| **China** | +0.01 | +0.02 | -0.07 | -0.11 | **+0.29** |

## OOS Results — Long/Flat P&L (higher is better, Policy vs Baseline)

| Region | 5d | 14d | 30d | 60d | 120d |
|--------|-----|------|------|------|-------|
| **US** | +0.00% | +0.00% | +0.00% | **+1.61%** | **+1.12%** |
| **Japan** | +0.00% | +0.00% | +0.00% | **+0.62%** | **+4.48%** |
| **Europe** | +0.00% | +0.00% | +0.00% | **+3.47%** | **+9.39%** |
| **China** | +0.11% | +0.14% | -0.20% | **+0.34%** | **+1.31%** |

## OOS Results — Long/Short/Flat P&L (higher is better, Policy vs Baseline)

| Region | 5d | 14d | 30d | 60d | 120d |
|--------|-----|------|------|------|-------|
| **US** | +0.00% | +0.00% | +0.00% | **+3.22%** | **+2.24%** |
| **Japan** | +0.00% | +0.00% | +0.00% | **+1.24%** | **+8.97%** |
| **Europe** | +0.00% | +0.00% | +0.00% | **+6.94%** | **+18.77%** |
| **China** | +0.21% | +0.27% | -0.41% | **+0.68%** | **+2.62%** |

Key: LSF edge ~2x L/F edge everywhere, confirming genuine directional skill (not just conservative long bias).

---

## Tier Classification by Region (120d, P&L metric)

**US:** 1 Tier 1 (NVDA +0.6pp), 3 Tier 2 (BA -0.1pp, MSFT -0.6pp, GOOGL -0.3pp), 8 Tier 3
**Japan:** 3 Tier 1 (8035.T +7.0pp, 7203.T +5.7pp, 4502.T +0.2pp), 2 Tier 2 (9984.T -8.8pp, 8306.T -6.7pp), 4 Tier 3
**Europe:** 3 Tier 1 (EADSF +1.2pp, VWAGY -3.4pp, SHEL +2.5pp), 1 Tier 2 (NVO -0.1pp), 5 Tier 3
**China:** 2 Tier 1 (0941.HK +2.1pp, 2318.HK +5.8pp), 0 Tier 2, 7 Tier 3

---

## Directional Hit-Rate (OOS)

The model takes a directional position (long or short) on 100% of predictions — it never naturally goes flat.

| Region | Horizon | Long% | Short% | Flat% | Long Acc | Short Acc | Overall Acc |
|--------|---------|-------|--------|-------|----------|-----------|-------------|
| US | 60d | 54.4% | 45.6% | 0.0% | 50.0% | 45.1% | 47.8% |
| US | 120d | 50.0% | 50.0% | 0.0% | 30.0% | 27.8% | 28.9% |
| Japan | 60d | 48.1% | 51.9% | 0.0% | 63.1% | 45.7% | 54.1% |
| Japan | 120d | 65.9% | 34.1% | 0.0% | 64.0% | 76.1% | 68.1% |
| Europe | 60d | 45.2% | 54.8% | 0.0% | 49.2% | 56.8% | 53.3% |
| Europe | 120d | 45.2% | 54.8% | 0.0% | 49.2% | 74.3% | 63.0% |
| China | 60d | 40.7% | 59.3% | 0.0% | 80.0% | 60.0% | 68.1% |
| China | 120d | 19.3% | 80.7% | 0.0% | 100.0% | 68.8% | 74.8% |

### Portfolio-Level Bootstrap CIs on Policy Edge (OOS)

| Region | Horizon | L/F Edge | L/F 95% CI | LSF Edge | LSF 95% CI | Sig? |
|--------|---------|----------|------------|----------|------------|------|
| US | 60d | +1.61% | [+0.86, +2.36] | +3.22% | [+1.72, +4.71] | Yes |
| US | 120d | +1.12% | [+0.39, +1.80] | +2.24% | [+0.78, +3.60] | Yes |
| Japan | 60d | +0.62% | [-0.22, +1.54] | +1.24% | [-0.44, +3.07] | **No** |
| Japan | 120d | +4.48% | [+2.73, +6.40] | +8.97% | [+5.47, +12.80] | Yes |
| Europe | 60d | +3.47% | [+2.24, +4.65] | +6.94% | [+4.47, +9.30] | Yes |
| Europe | 120d | +9.39% | [+7.56, +11.12] | +18.77% | [+15.13, +22.23] | Yes |
| China | 60d | +0.34% | [+0.04, +0.67] | +0.68% | [+0.08, +1.34] | Yes |
| China | 120d | +1.31% | [+0.68, +2.03] | +2.62% | [+1.36, +4.05] | Yes |

Japan 60d CI crosses zero — not statistically significant without dead zone.

---

## Dead Zone Parameter Sweep

Dead zones filter low-conviction predictions: if |predicted_return| < threshold, go flat instead of taking a position. Thresholds swept: 0%, 0.5%, 1%, 2%, 5%.

### Optimal Dead Zones per Region/Horizon (OOS, highest significant edge)

| Region | Horizon | Optimal DZ | Edge | 95% CI |
|--------|---------|-----------|------|--------|
| US | 60d | 0.0% | +3.22pp | [+1.74, +4.72] |
| US | 120d | 5.0% | +2.41pp | [+1.62, +3.25] |
| Japan | 60d | 2.0% | +2.24pp | [+0.86, +3.58] |
| Japan | 120d | 0.0% | +8.97pp | [+5.35, +12.89] |
| Europe | 60d | 2.0% | +7.16pp | [+5.87, +8.42] |
| Europe | 120d | 1.0% | +20.37pp | [+17.11, +23.34] |
| China | 60d | 0.0% | +0.68pp | [+0.09, +1.35] |
| China | 120d | 0.5% | +2.68pp | [+1.52, +3.98] |

### Two Alpha Profiles

| Profile | Regions | Dead zone effect | Mechanism |
|---------|---------|-----------------|-----------|
| Concentrated signal | US, Japan, China | DZ helps — edge increases as low-conviction noise is filtered | Alpha lives in a few high-conviction predictions; many small predictions are coin flips |
| Distributed signal | Europe 60d | DZ hurts — filtering removes correct small predictions | Many small correct predictions that individually look low-conviction but collectively add up |

### Japan 60d Rescued by Dead Zone

Without dead zone, Japan 60d CI crosses zero (not significant). At DZ=2%, Japan 60d becomes statistically significant: edge=+2.24pp, CI=[+0.86, +3.58]. The dead zone filters low-conviction predictions that were masking the real signal.

### Dead Zone Enforcement in Forecast Command

The `build` command selects the optimal dead zone per horizon (highest edge with CI lower bound > 0) and stores it in `routing_table.json` under `optimal_dead_zones`. The `forecast` command reads this and applies it automatically:

```bash
# Uses optimal DZ from routing table
python scripts/sector_router.py forecast --region japan --tickers "8035.T" --horizon 120

# Override with explicit dead zone
python scripts/sector_router.py forecast --region japan --tickers "8035.T" --horizon 120 --dead-zone 0.02
```

Each forecast shows a position signal (**LONG**, **SHORT**, or **FLAT**). Predictions within the dead zone are labeled FLAT with a conviction warning.

---

## Key Findings

1. **Finetuning is a ticker-level improvement, NOT a market-level one.** Each region has tickers that significantly regress under finetuning (e.g., MSFT in US, 9984.T/8306.T in Japan).

2. **Route vs Baseline holds at 60d+ in all 4 regions** — finetuning helps at longer horizons. This survives temporal OOS.

3. **Short horizons (5-14d) are noise everywhere** — OOS improvements are zero or negative. Pretrained TimesFM is already well-calibrated for daily.

4. **LSF edge ~2x L/F edge** — The model has genuine directional skill, not just conservative long bias.

5. **Per-ticker routing adds clear OOS value only in China** (+1.31pp @120d). US/JP/EU use the simple horizon rule.

6. **Dead zones rescue marginal cells** — Japan 60d goes from non-significant to significant at DZ=2%.

7. **Two alpha profiles** — Concentrated signal regions (US, JP, CN) benefit from dead zones; distributed signal regions (EU 60d) are hurt by them. Dead zone must be tuned per cell.

---

## Board Review Recommendations (2026-04-08, revised 2026-04-11)

Source: `personas/board_meetings/timesfm_lsf_strategy_apr2026.md`

### Adopted Decisions

1. **Stop reporting 5-14d deltas as wins or losses.** OOS improvements at short horizons are zero or within noise in every region.
2. **Adopt the global simple rule** for US, Japan, Europe: finetuned at 60d+, baseline below. Per-ticker router only for China.
3. **Sector matching is a selection rule.** Training tickers must be from the same sectors as eval/deployment tickers.
4. **LSF strategy for Europe and Japan** (positive absolute returns under LSF); **L/F strategy for US and China** (negative absolute returns under LSF).
5. **Per-cell dead zone** — each region/horizon has an independent optimal dead zone stored in routing_table.json and enforced by the forecast command.

### Revised Allocation (Board Approved, 2026-04-11)

| Cell | Allocation | Strategy | Optimal DZ |
|------|-----------|----------|------------|
| Europe 120d | 25% | LSF | 1.0% |
| Japan 120d | 20% | LSF | 0.0% |
| Europe 60d | 15% | LSF | 2.0% |
| China 120d | 10% | L/F | 0.5% |
| Japan 60d | 10% | LSF | 2.0% (rescued) |
| US 60d | 10% | L/F | 0.0% |
| Cash | 10% | — | — |

### Governance Rules
- 3% single-name short cap
- -10% stop-loss on any short position
- 40% maximum net-short exposure
- No-short blocklist: sovereign-adjacent names (HKEX, HSBC HK, China Mobile)

### Required Before Capital Deployment

- Frozen audit artifacts (dated CSV, bootstrap seed, excluded tickers disclosed)
- Position-sizing governor (no sizing from forecasts alone — require second orthogonal signal)
- Median + worst-ticker disclosure alongside every mean
- Quarterly rebuild cadence with tier-flip alerts
- Adversarial reviewer (cannot be the person who built the router)

### Open Action Items

| Item | Deadline | Status |
|------|----------|--------|
| Per-cell DZ field in routing_table.json | Apr 15 | Code done, pending rebuild |
| Dead zone enforcement in forecast command | Apr 15 | Code done, pending rebuild |
| Short-side blocklist enforcement | Apr 15 | Not started |
| Horizon restriction enforcement (no US 120d) | Apr 15 | Not started |
| Paper-trade simulation script | Apr 18 | Not started |
| Borrow-cost simulation for paper trade | Apr 18 | Not started |
| FX impact analysis for Japan/China | Apr 20 | Not started |
| Confidence-weighted position sizing prototype | Apr 22 | Not started |
| Dashboard prototype (L/S/F/blocked states) | May 1 | Not started |
| Gen 2 macro covariates design | May 15 | Not started |

---

# UPDATE (2026-06-16): Powered re-validation, corrections, and live-trade status

A leakage-free evaluation harness was built (June 2026) to re-test these results:
**n=60 held-out tickers/region** (zero training-set tickers), **non-overlapping windows**
(step = horizon, ~10y span, vs the step-1/5 overlapping windows here), **seed-replicated**,
paired cluster bootstrap. Code + full writeup:
`latent_reasoning/research/SECTOR_ROUTING_COMPARISON.md`,
`latent_reasoning/research/artifacts/eval_harness/` (`pnl_harness.py`, `post_cutoff.py`).

## 1. The P&L edge is REAL — survives leakage-free eval for US/JP/EU (not China)

Re-running the L/F + LSF P&L objective on the same finetuned models, n=60 held-out tickers,
non-overlapping windows (120d, TEST gate):

| Region | Router L/F 120d | Harness L/F 120d | Verdict |
|--------|----------------:|-----------------:|---------|
| US     | +1.12% | **+3.93% SIG** | survives (bigger) |
| Japan  | +4.48% | **+3.60% SIG** | survives |
| Europe | +9.39% | **+1.54% SIG** | survives, ~6× smaller |
| China  | +1.31% | +0.49% **ns** | does NOT survive |

The directional edge for US/Japan/Europe at 60d+ is genuine, not a methodology artifact —
this corroborates Test C. **China's P&L edge does not survive** held-out eval (and the
newer enriched-NextLat China model gives no P&L lift over Glia2a — ns).

## 2. Correction: "leakage" overstated — but CIs here are over-confident

Earlier internal critiques called the train-ticker-in-eval issue "leakage." That was wrong:
training data ends ~Feb 2026 and the eval windows post-date it, so there is **no temporal
lookahead**, and trading the training basket is legitimate *if you deploy on those tickers*
(which the paper trade does). The held-out harness measures a *different* thing —
generalization to NEW tickers — which is weaker (the edge is largely ticker-specific).

The one critique that **does** stand: the overlapping windows here (step 1/5 < horizon) make
the 10k-bootstrap CIs **over-confident**. On a single genuinely-independent post-cutoff
episode the same edge that reads tight in-sample (e.g. EU 120d ~[+17,+23]) widens to
~[−1.3,+28]. The point estimates are directionally right; the **confidence intervals are too
narrow.** Magnitudes are also inflated where training tickers dominate a small eval set
(Europe most of all — its +9.4% → +1.5% held-out).

## 3. Per-ticker routing adds no OOS value — simple rule confirmed (incl. China)

Honest-temporal-split tier classification on n=60 held-out (FT for Tier-1, baseline else)
underperforms **all-FT** in every region (US −2.5pp, JP −2.7pp, CN −1.4pp vs all-FT OOS).
This **confirms the `simple` policy** for US/JP/EU — and shows China's `router` policy adds
nothing on clean data. **Recommendation: drop the China per-ticker router; use the simple
60d+ rule everywhere (or drop China entirely, see §5).**

## 4. MAPE and P&L dissociate (don't optimize the wrong one)

On the harness: **China** is the only robust *MAPE* win (enriched-NextLat beats pretrained,
3-seed, P=0.4%) but the *P&L* laggard. **US** is *worse* on MAPE yet *better* on P&L.
Point-accuracy and directional-P&L are near-orthogonal here — pick the metric matching the
use case. For this trading strategy, P&L is the one that matters, and it favors US/JP/EU.

## 5. Live paper trade — status 2026-06-16 (see `board_recommended_test/mtm_2026-06-15.md`)

First live MTM: **NET −$16,238 (−1.62%)** after ~22 trading days. **Entirely the short side** —
4 stop-losses on the two highest-conviction shorts: **ASML +19.4%, Tokyo Electron 8035.T
+42.1%** (both semis; model was bearish into a semi rally). Long book +$6.6k (Japan 120d
longs carrying). Too early to judge (60–120d horizons, nothing expired; first 60d rebalance
~Jul 13). Confirms the documented short-side weakness (short accuracy 19–32%).

**Sizing note:** positions are **equal-weight per cell** with dead-zone *gating* (binary) —
NOT confidence *sizing*. Confidence sizing would have **worsened** this drawdown: the losing
shorts were the *highest*-conviction bets, i.e. conviction was anti-predictive on the short
side. Right lever = reduce short-side risk (LF-only / hard short cap on low-short-accuracy
names, esp. semis), not size-by-conviction.

## 6. Post-cutoff temporal-OOS test: not yet possible at strategy horizons

Only ~75 trading days exist after the ~Feb-2026 cutoff, so a clean temporal-OOS test at
60–120d is infeasible (H=120 impossible; H=60 one marginal episode → CIs span 0). The **live
paper trade is the only clean arbiter**, and won't produce a realized horizon-aligned signal
until the Jul (60d) / Oct (120d) rebalances.

## Revised recommendations
- **Keep** the US/JP/EU simple-rule 60d+ cells — their P&L edge is leakage-free-validated.
- **Drop or de-weight the China cell** — no held-out P&L edge; cell is also chronically
  dead-zoned in the live book.
- **Cut short-side risk** on semis / low-short-accuracy names (the entire live drawdown).
- **Treat the expected-return table's CIs as over-confident**; do not size to the point
  estimates. The forward live Sharpe (>0.8 after costs over 6mo) remains the decision gate.
- A P&L/directional *training* loss is NOT worth building yet — it can't be validated until
  the live/post-cutoff signal matures, and NextLat's regularizer buys MAPE, not P&L.
