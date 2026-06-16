# Board Proposal: Additional Ticker Combination Tests

**Date:** 2026-04-13
**Status:** Test C complete; Tests A/B/D pending board approval
**Context:** The current paper trade runs a single ticker combination per region.
We propose parallel tests with alternative combinations to stress-test whether
the observed edge is robust or dependent on the specific train/eval split.

**Test C result (2026-04-13):** Cross-region evaluation confirms the edge is
region-specific, not a methodological artifact. See `cross_region_test/README.md`.

---

## Why This Matters

The current portfolio's expected return (~20.8% annualized) is dominated by
Europe 120d (10.2pp of 20.8pp). If the edge is an artifact of the specific
3 eval tickers chosen (UL, EADSF, VWAGY), we have a fragile strategy. Testing
alternative splits answers the question: **is the edge in the model or in the
ticker selection?**

---

## Current Configuration (Baseline)

| Region | Train Tickers | Eval Tickers | Model |
|--------|---------------|--------------|-------|
| US | AAPL, MSFT, AMZN, GOOGL, META, JPM | NVDA, UNH, GS, CVX, KO, BA | Run4 |
| Japan | 7203.T, 6758.T, 9984.T, 8306.T, 4502.T, 7974.T | 8035.T, 6902.T, 4661.T | hdecay1.0 |
| Europe | ASML, SAP, NVO, AZN, SHEL, DEO | UL, EADSF, VWAGY | EUv2 |
| China | 0700.HK, 0941.HK, 2318.HK, 1299.HK, 0005.HK, 0001.HK | 0388.HK, 2382.HK, 1177.HK | Glia2a |

---

## Proposed Test Combinations

### Test A: Swap Train/Eval Roles

The simplest robustness check. Retrain each region with current eval tickers
as training data and current train tickers as eval. If the edge persists when
evaluated on a completely different set, the model generalizes.

| Region | New Train | New Eval | Requires |
|--------|-----------|----------|----------|
| US | NVDA, UNH, GS, CVX, KO, BA | AAPL, MSFT, AMZN, GOOGL, META, JPM | New data prep + retrain |
| Japan | 8035.T, 6902.T, 4661.T | 7203.T, 6758.T, 9984.T, 8306.T, 4502.T, 7974.T | New data prep + retrain |
| Europe | UL, EADSF, VWAGY | ASML, SAP, NVO, AZN, SHEL, DEO | New data prep + retrain |
| China | 0388.HK, 2382.HK, 1177.HK | 0700.HK, 0941.HK, 2318.HK, 1299.HK, 0005.HK, 0001.HK | New data prep + retrain |

**Risk:** Smaller train sets (3 tickers) may not produce valid checkpoints
(val_loss < 0.015). Japan and China already struggle with window counts.

### Test B: Expanded Universe (v2 data already prepared)

Training data variants already exist on disk. These were prepared during Glia
optimization but never run through the full sector-router evaluation pipeline.

| Region | Dataset | Tickers | vs Baseline |
|--------|---------|---------|-------------|
| Japan | `train_v2/` | 18 tickers (incl. 4063.T, 4502.T, 6367.T, 6723.T, 7267.T, 8316.T, 8801.T, 9433.T) | +8 tickers, broader sector coverage |
| Japan | `train_semi/` | 10 tickers (semiconductor-focused: 4063.T, 6146.T, 6723.T, 6857.T, 7735.T) | Sector-matched to 8035.T (Tokyo Electron) |
| Japan | `train_sector_matched/` | 6 tickers (auto/electronics: 6723.T, 7267.T) | Focused on eval-sector analogues |
| Europe | `train_v2/` | 17 tickers (incl. DEO, GSK, ING, LIN, PHG, SAN, STLA) | +7 tickers, adds automotive (STLA for VWAGY) |
| China | `train_v2/` | 17 tickers (incl. 0001-0003.HK, 0016.HK, 0027.HK, 0883.HK, 1398.HK) | +7 long-history HK blue chips |
| China | `train_long/` | 6 tickers (15-20yr history only) | Drops short-history noise sources |
| US | `train_ba_post2019/` | 11 tickers (AAPL, AMT, BA, CAT, GSPC, JNJ, JPM, NEE, PG, WMT, XOM) | Different sector mix, post-2019 regime |

**Advantage:** Data already on disk. Only requires retraining + eval, no new
data preparation.

### Test C: Cross-Region Eval (No Retrain)

Use existing checkpoints but evaluate on tickers from other regions' training
sets. Tests whether the model learned generalizable time-series patterns or
region-specific artifacts.

| Checkpoint | Evaluate On | Tests |
|------------|-------------|-------|
| Europe EUv2 | US train tickers (AAPL, MSFT, etc.) | Does EU model work on US equities? |
| Japan hdecay1.0 | Europe eval tickers (UL, EADSF, VWAGY) | Cross-market generalization |
| US Run4 | Japan eval tickers (8035.T, 6902.T, 4661.T) | Baseline vs finetuned cross-region |

**Advantage:** Zero retraining cost. Pure evaluation runs (~5 min each).

### Test D: Leave-One-Out Ticker Rotation

For each region, rotate which single ticker is held out for eval while the rest
train. Produces N models per region (where N = total tickers). This is the gold
standard for small-sample robustness but expensive.

| Region | Total Tickers | Models to Train | Compute Cost |
|--------|---------------|-----------------|--------------|
| US | 12 | 12 | ~12 hours |
| Japan | 9 | 9 | ~9 hours |
| Europe | 9 | 9 | ~9 hours |
| China | 9 | 9 | ~9 hours |

**Advantage:** Eliminates ticker-selection bias entirely.
**Disadvantage:** 39 training runs. ~39 hours of GPU time.

---

## Recommended Priority

| Priority | Test | Effort | Value | Status |
|----------|------|--------|-------|--------|
| 1 | **Test C** (cross-region eval) | Low (eval only) | Quick sanity check on generalization | **DONE** -- edge is region-specific, not artifact |
| 2 | **Test B** (v2 expanded data) | Medium (retrain 4 models) | Tests if more data helps, data already prepared | Pending |
| 3 | **Test A** (swap train/eval) | Medium (retrain 4 models) | Direct robustness check on ticker selection | Pending |
| 4 | **Test D** (leave-one-out) | High (39 models) | Definitive answer but expensive | Pending |

---

## Decision Requested

1. Approve Test C (cross-region eval) as immediate next step -- can run today.
2. Approve Test B (v2 data retrain) as the primary expansion -- data on disk,
   retrain over the week.
3. Decide whether Test D (leave-one-out) justifies the compute budget given
   the paper trade is still in dry-run mode.

---

## Success Criteria

For any new combination to be added to the live paper trade:
- Bootstrap policy edge CI lower bound > 0 on OOS windows
- Directional accuracy > 50% at the traded horizon
- No regression vs pretrained baseline (MAPE improvement > 0)
