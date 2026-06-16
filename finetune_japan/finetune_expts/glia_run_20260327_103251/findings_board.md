# Findings Board

## Pre-run baseline (rolling eval, 15 windows, 2026-03-27)
- Baseline: 18.07% at 120d, 8.77% at 60d, 3.73% at 30d, 4.93% at 14d
- Best checkpoint v2_2phase: 4.52% at 14d (+0.41pp), 3.60% at 30d (+0.13pp)
- 60d: ALL checkpoints regress (-3.1pp to -3.8pp). 8035.T drives regression (18.6%->26.7%)
- 120d: glia_2b 17.28% (+0.79pp) but high variance (std 3.09%)
- Dead checkpoints: glia_1a, glia_2a (NaN outputs)
- Key problem: 8035.T (Tokyo Electron, semiconductor) drives long-horizon regression

## Turn 1
- R1: Semi-heavy 10tk (4 semi equip + wafer + 2 elec + 2 auto + idx). Full training: 50ep, early stop ep32, val_loss=0.020.
  - Rolling 15w: 14d 5.44% (-0.47pp), 60d **8.35% (+0.52pp)**, 120d 18.78% (-0.84pp)
  - FIRST MODEL TO BEAT BASELINE AT 60d. 8035.T: 17.4% vs 18.8% baseline vs 26.0% v2_2phase.
  - v2_2phase wins 14d (+0.53pp), semi_10tk wins 60d (+0.52pp). Horizon-routing gives both.
  - 120d: both models regress. Unsolved.
- R2: Horizon-weighted loss (decay=2.0), original 10tk data, val_loss=0.015 (at threshold). NEW BEST at 14d!
  - Rolling 15w: **14d 4.10% (+0.86pp)**, 60d 10.36% (-1.50pp), **120d 17.67% (+0.26pp)**
  - Beats v2_2phase at 14d (4.10% vs 4.44%) AND first model to improve at 120d.
  - 60d still regresses (8035.T 20.9% vs 18.8% baseline) — sector mismatch still dominates there.
  - Horizon-selective routing: hdecay for 14d/120d, semi_10tk for 60d = improvements everywhere.

## Turn 2
- R1: Combination (semi_10tk data + hdecay_2.0 loss). val_loss=0.021 (ep46, ran all 50).
  - Rolling 15w: 14d 5.83% (-0.86pp WORST), 60d 10.13% (-1.27pp), **120d 17.09% (+0.85pp NEW BEST)**
  - Combination INTERFERES at 14d/60d — destroys both parents' gains at those horizons.
  - BUT additive at 120d: 8035.T 28.1% vs 32.7% baseline (-4.6pp). Best 120d result by far.
  - Methods supervisor confirmed: sector-matching changes which positions benefit, so data × loss interact.
  - Revised horizon routing: hdecay_2.0 for 14d, semi_10tk for 60d, semi+hdecay for 120d.
- R2: Horizon-weighted loss (decay=1.0, gentler), original 10tk data, val_loss=0.013026 (ep47). **[BREAKTHROUGH]**
  - Rolling 15w: **14d 3.98% (+0.99pp NEW BEST)**, 60d 12.12% (-3.26pp worst), **120d 15.74% (+2.20pp NEW BEST)**
  - Largest 120d improvement in any Japan experiment. NOT single-ticker: 8035.T -4.7pp AND 4661.T -3.0pp.
  - Gentler decay (1.0 vs 2.0) gives more weight to long positions → massive 120d gain.
  - But 60d regression is WORST of all models. 8035.T 23.8% vs 18.8% baseline.
  - Updated best per horizon: hdecay_1.0 for 14d/120d, semi_10tk for 60d.

## 2×2 Factorial Summary (original data axis)
| Decay | 14d | 60d | 120d | val_loss |
|-------|-----|-----|------|----------|
| 0.0 (MSE) | 4.52% (v2_2phase) | 11.84% | 19.25% | ~0.013 |
| 2.0 | 4.10% | 10.36% | 17.67% | 0.015 |
| 1.0 | **3.98%** | 12.12% | **15.74%** | 0.013 |
Pattern: lower decay → better 14d AND 120d, but worse 60d.

## Regional eval validation (hdecay_1.0)
- Eval tickers: 14d **-2.0pp** (strong), 120d **-2.6pp** (strong), 60d -0.2pp
- Training tickers: 30d +2.0pp (REGRESSED), 120d +0.5pp (REGRESSED)
- Mixed: eval improves, training regresses → error redistribution, not free lunch

## Evaluator T3 Decisions
- [KILL: semi_10tk_standalone], [KILL: semi+hdecay_2.0_combination]
- 60d dead zone accepted as structural
- semi+decay=1.0 is the FINAL combination test → then approve/terminate
- Killed directions: semi_10tk standalone, semi+hdecay_2.0

## Turn 3
- R1: semi data + decay=1.0. val_loss=0.021874 (ep45). FAILED — regresses at ALL horizons vs baseline.
  - 14d 5.37% (-0.41pp), 60d 9.76% (-0.90pp), 120d 19.36% (-1.42pp)
  - Semi data × horizon decay confirmed UNIVERSALLY destructive (both decay=2.0 and 1.0)
  - Only exception: semi+hdecay_2.0 at 120d (17.09%) — but that was single-ticker driven and killed.

## Complete 2×2+1 Factorial
| Config | 14d | 60d | 120d | val_loss | Status |
|--------|-----|-----|------|----------|--------|
| MSE, orig (v2_2phase) | 4.52% | 11.84% | 19.25% | ~0.013 | Prior best |
| decay=2.0, orig | 4.10% | 10.36% | 17.67% | 0.015 | Good |
| **decay=1.0, orig** | **3.98%** | 12.12% | **15.74%** | **0.013** | **BEST** |
| MSE, semi | 5.44% | **8.35%** | 18.78% | 0.020 | 60d only |
| decay=2.0, semi | 5.83% | 10.13% | 17.09% | 0.021 | Killed |
| decay=1.0, semi | 5.37% | 9.76% | 19.36% | 0.022 | Failed |

**FINAL ANSWER: hdecay_1.0 (decay=1.0, original 10tk data) is the best Japan checkpoint.**
- Best at 14d (+0.99pp) and 120d (+2.20pp)
- 60d regresses (-3.26pp) — structural dead zone, unfixable in single checkpoint

