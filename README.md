# TimesFM Finetuning & Sector-Routing Trading Strategy

This project finetunes Google Research's **TimesFM 2.5 (200M)** time-series foundation
model on regional equity baskets and tests whether the resulting directional edge is real
enough to trade. It spans four regional finetunes, a bootstrap evaluation/routing harness,
a leakage-free re-validation, a board review process, and a live governed paper trade.

> **Bottom line:** finetuning produces a genuine, leakage-validated directional-P&L edge at
> 60–120 day horizons in **US, Japan, and Europe** (not China). A board-approved
> long/short/flat portfolio is running as a paper trade to decide, on realized forward
> Sharpe, whether it deploys real capital.

---

## Repository layout

| Path | What it is |
|------|------------|
| `finetune_timesfm.py` | Finetuning script (loads `.npy` series, trains with MSE / horizon-weighted loss) |
| `finetune_{usa,japan,europe,china}/` | Per-region training data, checkpoints, and routing tables |
| `model/` | Base TimesFM 2.5 200M pretrained checkpoint |
| `scripts/sector_router.py` | Bootstrap evaluation, tier classification, dead-zone sweep, routing/forecast |
| `sector_routing_config.yaml` | Regions, ticker lists, checkpoints, policies, dead zones, blocklists |
| `sector_routing_results.md` | **Master results doc** — bootstrap results + June 2026 re-validation |
| `board_recommended_test/` | Live governed **paper trade** (state, journal, MTM reports) |
| `evaluate_model.py`, `rolling_eval_results.md` | Rolling-window evaluation utilities and output |
| `CONTINUATION.md` | Checkpoint-restoration notes (models retrained June 2026) |
| `Fu_Hirano_Imajo-ArXiv_2024*.pdf` | Reference paper (financial finetuning of a large TS model) |

Environment note: the Python venv lives in the sibling `timesfm/` repo, **not** here. Use
`C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe` for all commands.

---

## 1. Finetuning

Four independent per-region finetunes of TimesFM 2.5 200M, each starting from the pretrained
base and trained on that region's equity basket. Shared optimizer recipe: **AdamW, lr=1e-4,
weight-decay=0.01, freeze 17/20 layers, cosine schedule + warmup, early stopping**,
max-context 512, horizon 128. (All four were retrained in **June 2026** after a disk cleanup
deleted the original checkpoints — see `CONTINUATION.md`.)

| Region | Checkpoint label | Data | Distinguishing config |
|--------|------------------|------|-----------------------|
| US | `Run4` | 10 S&P 500 tickers | MSE loss, 3-epoch warmup |
| Japan | `hdecay1.0` | 10 JP tickers | horizon-weighted loss (decay=1.0), stride=64 |
| Europe | `EUv2` | 17 NYSE/NASDAQ-listed EU names | stride=64 |
| China | `Glia2a` | 10 HK tickers (long-history) | stride=64, val_loss≈0.0121 |

Checkpoint paths are wired in `sector_routing_config.yaml`. Finetuning is deliberately a
**ticker-level** improvement, not a market-level one — every region contains individual
tickers that *regress* under finetuning, which is what the routing/tiering step exists to
handle.

---

## 2. Testing & evaluation

### 2a. Bootstrap sector router (April 2026)

`scripts/sector_router.py build` runs per-ticker, per-horizon bootstrap evaluation:
**10,000 resamples over 30 rolling windows**, temporal honest split (classify on windows
0–14, evaluate on 15–29), horizons 5/14/30/60/120d. It produces MAPE, Long/Flat and
Long/Short/Flat (LSF) P&L, directional hit-rates, a dead-zone parameter sweep, and per-region
routing tables (`finetune_{region}/routing_table.{json,md}`).

**Key findings:**

1. **Finetuning helps at 60d+ in all four regions; 5–14d is noise everywhere.** Pretrained
   TimesFM is already well-calibrated for short horizons.
2. **The model has genuine directional skill** — the LSF edge is ≈2× the Long/Flat edge in
   every OOS cell, meaning its *bearish* calls are also correct on average (not just a
   conservative long bias).
3. **Per-ticker routing adds clear OOS value only in China** (+1.31pp @120d); US/JP/EU use a
   simple "finetuned at 60d+, baseline below" rule.
4. **Dead zones rescue marginal cells** — filtering low-conviction predictions turns Japan 60d
   from non-significant to significant (edge +2.24pp at DZ=2%). Two alpha profiles exist:
   *concentrated* (US/JP/CN, helped by dead zones) vs *distributed* (EU 60d, hurt by them).
5. **The short side is the liability** — short accuracy is only 19–32% in several cells;
   this later drove the entire live drawdown.

### 2b. Leakage-free re-validation (June 2026)

A stricter harness re-tested everything: **n=60 held-out tickers per region** (zero
training-set tickers), **non-overlapping windows** (step = horizon, ~10y span),
seed-replicated, paired cluster bootstrap.

- **The P&L edge is real for US / Japan / Europe** at 120d and survives held-out eval
  (US +3.93%, Japan +3.60%, Europe +1.54%, all significant). **China does not survive**
  (+0.49%, ns).
- **"Leakage" was overstated** — training data ends ~Feb 2026 and eval windows post-date it,
  so there is no temporal lookahead. But the **in-sample bootstrap CIs are over-confident**
  (overlapping windows); point estimates are directionally right, intervals too narrow.
- **Per-ticker routing adds no OOS value on clean data** — confirms the simple rule
  everywhere, including China.
- **MAPE and P&L dissociate** — China is the only robust MAPE win but the P&L laggard; US is
  worse on MAPE yet better on P&L. For a trading strategy, P&L is the metric that matters.

Full detail: `sector_routing_results.md` (see the "UPDATE (2026-06-16)" section).

---

## 3. Board recommendations

The strategy was reviewed by a standing analyst board (personas: Buffett, Munger, Burry,
Simons, Feynman, Kurzweil, Stephenson, Oppenheimer, Brin, Jobs). Three meeting records live
in `C:\Users\ylchen\workspace\personas\board_meetings\`:

| File | Date | Topic |
|------|------|-------|
| [`timesfm_regional_finetuning_apr2026.md`](../personas/board_meetings/timesfm_regional_finetuning_apr2026.md) | 2026-04-08 | Review of the regional finetunes + bootstrap sector-routing results |
| [`timesfm_investment_plan_apr2026.md`](../personas/board_meetings/timesfm_investment_plan_apr2026.md) | 2026-04-10 | Translating the P&L-tiered evaluation into a concrete investment plan |
| [`timesfm_lsf_strategy_apr2026.md`](../personas/board_meetings/timesfm_lsf_strategy_apr2026.md) | 2026-04-10 → 05-14 | Long/Short/Flat strategy; CI analysis, dead-zone sweep, final allocation (multiple reconvenes) |

### Adopted decisions

1. **Stop treating 5–14d deltas as signal** — zero or within noise in every region.
2. **Adopt the global simple rule** (finetuned at 60d+, baseline below) for US/JP/EU; China
   was the only per-ticker `router` region (later dropped on clean-data evidence).
3. **Sector matching is a selection rule** — training tickers must match the sectors of
   eval/deployment tickers.
4. **Strategy by region: LSF for Europe and Japan** (positive absolute LSF returns);
   **Long/Flat for US and China** (negative absolute LSF returns).
5. **Per-cell dead zones are a mandatory governance parameter**, stored in the routing tables
   and enforced by the forecast command; changing one requires board approval.

### Board-approved allocation (2026-04-11)

$1,000,000 notional, dead zones applied per cell from the routing tables:

| Cell | Weight | Strategy | Model |
|------|--------|----------|-------|
| Europe 120d | 25% | LSF | EUv2 |
| Japan 120d | 20% | LSF | hdecay1.0 |
| Europe 60d | 15% | LSF | EUv2 |
| China 120d | 10% | L/F | Glia2a |
| Japan 60d | 10% | LSF | hdecay1.0 |
| US 60d | 10% | L/F | Run4 |
| Cash | 10% | — | — |

**US 120d is permanently blocked** (28.9% directional accuracy — a statistical artifact).

### Governance rules

- 3% single-name short cap · −10% stop-loss on any short · 40% max net-short exposure
- 1.5% annualized borrow cost on shorts
- No-short blocklist for sovereign-adjacent names (HKEX 0388.HK, HSBC HK 0005.HK,
  China Mobile 0941.HK)
- Required before real capital: frozen audit artifacts, a position-sizing governor (no sizing
  from forecasts alone), median + worst-ticker disclosure, quarterly rebuild with tier-flip
  alerts, and an independent adversarial reviewer.

### Revised recommendations after June 2026 re-validation

- **Keep** the US/JP/EU simple-rule 60d+ cells — their P&L edge is leakage-free-validated.
- **Drop or de-weight China** — no held-out P&L edge, and chronically dead-zoned in the live book.
- **Cut short-side risk** on semis / low-short-accuracy names — the entire live drawdown so far.
- **Treat the CIs as over-confident**; do not size to the point estimates.
- Confidence-weighted sizing is **not** the fix — the losing shorts were the *highest*-conviction
  bets (conviction was anti-predictive on the short side).

### Open items

- **US 60d strategy discrepancy** — the board's final allocation specifies **LSF**, but the
  paper trade deploys **L/F**. Unresolved; needs a board decision at the next rebalance.
- Carried/overdue: confidence-weighted sizing prototype, FX-impact analysis for Japan/China,
  L/S/F/blocked dashboard, Gen-2 macro covariates.

---

## 4. Live paper trade

`board_recommended_test/` runs the board-approved allocation as a governed, mark-to-market
paper trade — the **only clean forward arbiter**, since <75 trading days exist after the
~Feb-2026 training cutoff (too few for a temporal-OOS test at 60–120d horizons).

- **Live basis:** 2026-05-14 · **Notional:** $1,000,000 · **Decision gate:** realized
  6-month Sharpe > 0.8 after costs.
- **Status (2026-07-12):** 39 open positions, **net P&L −$5,844 (−0.58%)** — recovered from
  −1.62% at the first live mark. All realized losses to date are shorts (four semiconductor
  stop-losses, −$22,680); the long book, led by Japan, is carrying.
- See `board_recommended_test/README.md` and the dated `mtm_YYYY-MM-DD.md` reports.

```bash
PY="C:/Users/ylchen/workspace/timesfm/.venv/Scripts/python.exe"

# Daily lifecycle: MTM -> stop-losses -> expiry -> rebalance (+ optional Markdown report)
$PY board_recommended_test/paper_trade.py run --report

# Portfolio status / closed-trade journal
$PY board_recommended_test/paper_trade.py status
$PY board_recommended_test/paper_trade.py history
```

---

## 5. Reproducing the evaluation

```bash
PY="C:/Users/ylchen/workspace/timesfm/.venv/Scripts/python.exe"

# Rebuild routing tables for a region (downloads prices via yfinance, ~5-10 min/region)
$PY scripts/sector_router.py build --region europe \
  --horizons "5,14,30,60,120" --n-windows 30 --n-boot 10000 \
  --honest-split --tier-metric pnl

# Inspect a routing table, or forecast with routing + dead-zone enforcement
$PY scripts/sector_router.py show --region europe
$PY scripts/sector_router.py forecast --region japan --tickers "8035.T" --horizon 120
```

---

## Key references

- `sector_routing_results.md` — master results + June 2026 re-validation
- `board_recommended_test/README.md` — paper-trade design, tickers, governance, MTM reports
- Board meetings — `../personas/board_meetings/timesfm_*.md`
- `CONTINUATION.md` — checkpoint restoration / training commands
- Base model & backends — the sibling `timesfm/` repo (`CLAUDE.md` there for setup)
