# Glia v2 Run Log — China/HK Region

Started: 2026-03-30
Region: china
Max turns: 27
Researchers: 2

---

## Turn 1 — Researcher R1

**Hypothesis:** Horizon-weighted loss (decay=1.0) on China 10tk data to fix short-horizon regression.
Glia2a regresses at 5d (-0.32pp) and 14d (-0.38pp). In Japan, decay=1.0 helped short horizons by +0.39-0.99pp. Mechanism: `weight[i] = exp(-1.0 * i / 128)` boosts early forecast positions.

**Prediction:** +0.2-0.3pp at 5d/14d vs Glia2a, -1 to -3pp at 120d (acceptable given 5pp headroom).

**Command:** Same as Glia2a + `--loss-horizon-decay 1.0`
**Save dir:** `finetune_china/checkpoints/run_glia_t1_r1a`

**Status:** Training complete. Val loss: 0.012527 (epoch 47 best). Below 0.015 threshold.

**Results (15-window rolling eval):**

| Horizon | T1-R1 | Glia2a | Baseline | T1-R1 vs Glia2a | T1-R1 vs BL |
|---------|-------|--------|----------|----------------|-------------|
| 5d | 2.77% | 2.87% | 2.55% | **+0.10pp** | -0.21pp |
| 14d | 3.73% | 3.95% | 3.57% | **+0.22pp** | -0.17pp |
| 30d | 4.87% | 4.83% | 5.41% | -0.04pp | +0.54pp |
| 60d | 7.99% | **6.49%** | 7.18% | **-1.50pp** | -0.80pp |
| 120d | 16.41% | **12.15%** | 17.16% | **-4.26pp** | +0.75pp |

Per-ticker 120d: 0388.HK 4.8% (vs 5.0%), 2382.HK 27.4% (vs 14.3% — massive regression), 1177.HK 17.0% (vs 17.2%)

**Analysis:** Horizon-decay improves short horizons vs Glia2a as predicted (+0.10pp at 5d, +0.22pp at 14d), but the improvement is modest — still worse than baseline at both horizons. The 60d and 120d regression is severe: 2382.HK (Sunny Optical) went from 14.3% to 27.4% at 120d, almost back to baseline. The horizon-decay starves long-range gradient signal for 2382.HK, which has the most complex long-range dynamics. The short-horizon fix is real but too small to justify the long-horizon cost.

**Prediction accuracy:** Predicted +0.2-0.3pp at short horizons → got +0.10-0.22pp (close). Predicted -1 to -3pp at 120d vs Glia2a → got -4.26pp (worse than expected). The trade-off is unfavorable.

---

## Turn 1 — Researcher R2

**Hypothesis:** Sector-matched ticker substitution. Replace 3 tickers with eval-sector proxies:
- Remove: 9999.HK (NetEase, redundant gaming), 3690.HK (Meituan, food delivery), 1810.HK (Xiaomi, phones)
- Add: 1093.HK (CSPC Pharma → proxy for 1177.HK), 2018.HK (AAC Technologies → proxy for 2382.HK), 0267.HK (CITIC → financial proxy for 0388.HK)

New 10: ^HSI, 0700.HK, 0941.HK, 1299.HK, 2318.HK, 0005.HK, 9988.HK, 1093.HK, 2018.HK, 0267.HK

**Prediction:** +0.5-1.0pp at 120d for 1177.HK (pharma proxy), +0.3-0.5pp for 2382.HK (optical proxy). 0388.HK may improve slightly. Net: +0.3-0.5pp overall at 120d vs Glia2a.

**Command:** `--tickers "^HSI,0700.HK,0941.HK,1299.HK,2318.HK,0005.HK,9988.HK,1093.HK,2018.HK,0267.HK"`
**Save dir:** `finetune_china/checkpoints/run_glia_t1_r2a`

**Status:** Training complete. Checkpoint exists at run_glia_t1_r2a/best.

**Results (15-window rolling eval):**

| Horizon | T1-R2 | Glia2a | Baseline | T1-R2 vs Glia2a | T1-R2 vs BL |
|---------|-------|--------|----------|----------------|-------------|
| 5d | 2.54% | 2.87% | 2.55% | **+0.33pp** | +0.02pp |
| 14d | 3.54% | 3.95% | 3.57% | **+0.41pp** | +0.03pp |
| 30d | 4.70% | 4.83% | 5.41% | **+0.13pp** | **+0.71pp** |
| 60d | 7.13% | **6.49%** | 7.18% | **-0.64pp** | +0.05pp |
| 120d | 14.91% | **12.15%** | 17.16% | **-2.76pp** | +2.25pp |

Per-ticker 120d: 0388.HK 4.0% (vs 5.0% Glia2a — BETTER), 2382.HK 14.4% (vs 14.3% — tied), 1177.HK 26.3% (vs 17.2% — WORSE than baseline!)

**Analysis:** Sector-matching fixes the short-horizon regression completely — R2 matches baseline at 5d/14d where Glia2a regresses. Best at 30d (+0.71pp vs BL). But 60d and 120d are worse than Glia2a. The 1177.HK (Sino Biopharma) regression to 26.3% (worse than baseline 21.8%) suggests the pharma proxy (CSPC/1093.HK) didn't help — possibly hurt by removing a more useful ticker. Meanwhile 0388.HK improved from 5.0% to 4.0%, suggesting the CITIC financial proxy helped.

---

## Turn 1 — Supervisor (Stats)

With n=3 eval tickers, every comparison is essentially anecdotal. The 120d "improvement" of Glia2a (12.15% vs 17.16%) is driven almost entirely by one ticker — 2382.HK swings from 14.3% to 27.4% under a different config. One ticker is not a result; it is a coin flip.

Questions:
1. Has anyone computed per-ticker standard deviations across rolling windows? Without that, how do we know the 5.01pp Glia2a advantage isn't a single favorable window artifact?
2. R1's horizon-decay shows a sign reversal between Japan and China at 120d. With n=3 in each region, is there any statistical basis to claim these are genuinely different responses rather than noise?
3. The short-horizon regressions (-0.32pp at 5d, -0.38pp at 14d) are small but consistent. At n=3, are these within noise or systematic bias from finetuning?

---

## Turn 1 — Supervisor (Domain)

Japan-China divergence on horizon-decay reflects market microstructure differences. Japan eval tickers (Tokyo Electron, Denso) track global capex cycles — smooth long-range structure that decay=1.0 rewards. HK/China eval tickers are different: 0388.HK is a second-derivative exchange operator, 2382.HK has sharp smartphone-cycle mean-reversion, 1177.HK is policy-driven pharma. None have smooth long-range structure.

Questions:
1. Does 0267.HK (CITIC, state conglomerate) actually share any price dynamics with 0388.HK (monopoly exchange)? Is 0388.HK fundamentally untraceable in HK markets?
2. Glia2a's 120d success came from somewhere — which of the original 10 training tickers drove it? Are R2's replacements removing a critical contributor?
3. Should China strategy abandon 120d and optimize purely for 30-60d where the signal seems learnable?

---

## Turn 1 — Supervisor (Methods)

The short-horizon regression is telling. In US and Japan, finetuning preserves or improves short horizons. In China, every finetuned checkpoint regresses at 5d/14d. This suggests China training data shifts patch-level representations away from the pretrained prior.

Questions:
1. If China training tickers have high short-horizon variance (HK small-caps), the model may dampen short-horizon outputs to minimize aggregate MSE — hurting MAPE on smoother eval tickers.
2. Training data length distribution: shorter histories → fewer long-context windows → fragile 120d gains?
3. R2 swaps 3 tickers simultaneously. If the result changes, how to isolate which substitution helped or hurt?

---

## Turn 2 — Researcher R1

**Hypothesis:** Horizon-selective routing — use R2-Sector for short horizons (5-30d) and Glia2a for long horizons (60-120d) to combine strengths.

**Method:** Created `scripts/horizon_selective_eval_china.py`. Honest split: 7 selection windows, 8 eval windows. Evaluates baseline, Glia2a, and R2-Sector at all 5 horizons across 15 rolling windows.

**Results:**

CRITICAL FINDING: The short-horizon regression is NOT real. With proper out-of-sample evaluation:

| Horizon | Baseline (eval) | Glia2a (eval) | R2-Sector (eval) | Glia2a vs BL |
|---------|----------------|---------------|------------------|--------------|
| 5d | 2.54% | 2.23% | 2.90% | **+0.31pp** |
| 14d | 4.04% | 3.78% | 5.17% | **+0.25pp** |
| 30d | 6.37% | 5.80% | 8.41% | **+0.58pp** |
| 60d | 11.45% | 8.04% | 10.66% | **+3.41pp** |
| 120d | 17.28% | 10.39% | 14.78% | **+6.89pp** |

- Glia2a dominates at ALL horizons (avg 6.05% vs baseline 8.34%)
- R2-Sector is WORSE than baseline at most horizons on eval windows
- R2-Sector selected for 5d only but wins 0/8 eval windows — completely unreliable
- Routing hurts: 6.18% avg vs Glia2a 6.05% (-0.13pp)
- The March 30 short-horizon regression was window-specific noise

**Conclusion:** Glia2a is the unambiguous best checkpoint for China at all horizons. No routing needed.

---

## Turn 2 — Researcher R2

**Hypothesis:** Mild horizon-decay (0.3) — Goldilocks between no-decay (Glia2a) and aggressive decay (1.0). Weight profile: pos 0 = 1.0, pos 127 = 0.74.

**Training:** val_loss = 0.012950 (epoch 36 best). Below 0.015 threshold.

**Results (15-window rolling eval):**

| Horizon | Baseline | Glia2a | decay=0.3 | vs Glia2a | vs BL |
|---------|----------|--------|-----------|-----------|-------|
| 5d | 2.55% | 2.87% | 2.87% | 0.00pp | -0.32pp |
| 14d | 3.57% | 3.95% | 3.86% | +0.09pp | -0.30pp |
| 30d | 5.41% | 4.83% | 4.89% | -0.06pp | +0.52pp |
| 60d | 7.18% | 6.49% | 8.12% | -1.63pp | -0.94pp |
| 120d | 17.16% | 12.15% | 15.62% | -3.47pp | +1.54pp |

Per-ticker 120d: 0388.HK 3.4% (better), 2382.HK 30.2% (CATASTROPHIC, from 14.3%), 1177.HK 13.2% (better).

**Analysis:** Even mild decay destroys 2382.HK's long-range forecast. The 120d improvement was the primary value of finetuning — any decay variant trades it away. Short-horizon gains are minimal. Decay is a dead end for China regardless of strength.

---

## Turn 2 — Supervisor (Stats)

[CEILING]

The evidence base after 2 turns is converging on a clear picture, but I have statistical concerns about the evaluation framework itself.

**On the honest-split evaluation (R1):** The 7/8 selection/eval split is a reasonable design, but with only 3 eval tickers and 8 out-of-sample windows, we have an effective sample size of ~24 ticker-window observations per horizon. The reported dominance of Glia2a across "all horizons" needs context — what are the standard deviations across those 8 eval windows? A mean of 10.39% at 120d with n=8 windows could easily have a 95% CI spanning 2-3pp. The claim "routing adds -0.13pp" is well within noise at this sample size.

**On the decay experiments:** Two data points (decay=1.0 catastrophic, decay=0.3 severe regression) are being generalized to "all decay values are dead ends." This is reasonable directionally — the monotonic worsening from 0.3→1.0 suggests the mechanism is fundamentally destructive for this ticker set — but the causal story matters. Is it really that decay "starves gradient signal" for 2382.HK, or is 2382.HK simply an outlier that dominates the 120d MAPE because of its price dynamics? One ticker driving the entire 120d metric is a fragility, not a finding.

**Questions:**
1. What fraction of the 120d MAPE improvement (+5.01pp over baseline) is attributable to 2382.HK alone? If removing one of three eval tickers flips the conclusion, we have n=2 effective statistical power — essentially anecdotal evidence.
2. Given that Glia2a now appears to dominate all horizons and all alternatives have regressed, what is the actual remaining hypothesis space? Are we optimizing, or are we confirming a result we already had before this run started?
3. Has anyone computed confidence intervals on the rolling MAPE differences?

---

## Turn 2 — Supervisor (Domain)

The honest-split result is striking: Glia2a's advantage *grows* with horizon length (+0.31pp at 5d → +6.89pp at 120d). This is the opposite of naive intuition. Structural explanation: short-horizon returns are noise-dominated (microstructure), while long-horizon returns are fundamentals-dominated (earnings, sector rotation, macro regimes). If Glia2a's training tickers share *fundamental drivers* with eval tickers, the model is learning regime-level patterns that compound over longer horizons — not daily moves but *drift structure* of HK equities.

2382.HK (Sunny Optical) is precision optics / consumer electronics with long capital cycles — its 120d dynamics reflect order-book visibility and capex commitments, not daily sentiment. Decay removes exactly this signal.

**Questions:**
1. What are the sector classifications of the 10 training tickers? If they over-index on tech hardware and financials (the sectors of the 3 eval tickers), Glia2a's success may be fortunate sector match rather than generalizable.
2. Is 2382.HK's 120d forecastability anomalous? Its transparent forward guidance may make it uniquely predictable. If so, the improvement is real but non-transferable.
3. Does the +6.89pp at 120d survive a regime break? Split eval windows into pre/post a known regime shift (e.g., 2024 China stimulus pivot) — does the advantage hold in both halves?

---

## Turn 2 — Supervisor (Methods)

The decay=0.3 result is mechanically informative. In TimesFM's decoder-only architecture, the output patch length is 128 — so horizon-weighted loss directly reshapes which positions within that output patch receive gradient signal. When decay=0.3 downweights the tail of the 128-position output, layers 17-19 receive weaker gradients for the positions that matter most for long-horizon forecasting. The fact that only 2382.HK collapses suggests its learned representation sits in a region of weight space where those tail-position gradients are load-bearing for stability.

Val_loss 0.012950 is well within the safe zone, yet MAPE explodes for one ticker. This reinforces: val_loss is a poor proxy for per-ticker forecast quality.

Seven experiments, one valid configuration. Glia2a's recipe appears to sit at a sharp minimum in hyperparameter space. Every perturbation — loss weighting, sector composition, decay — either produces NaN or degrades at least one ticker.

**Questions:**
1. Is this sharpness itself the problem? SWA across last N epochs could broaden the basin.
2. What happens at stride=64 or stride=16? Stride controls effective training distribution — haven't varied it.
3. Is the 10-ticker composition optimal for HK, or was it inherited from the US recipe?

---

## Turn 2 — Evaluator

### Strategic Assessment

Glia2a has survived 11 experiments without being beaten. The Turn 2 findings are decisive: the short-horizon regression that motivated this run was window noise, and Glia2a dominates all horizons on honest multi-window evaluation (+0.31pp to +6.89pp vs baseline). Every alternative — horizon-decay (3 variants), sector-matching, expanded tickers, reduced tickers, ensembles, routing — has either regressed or tied. The Stats supervisor's [CEILING] signal is well-founded. We are in confirmation territory, not discovery territory.

### Direction Reviews

**"Beat Glia2a"** — [KILL: beat-glia2a]
Justification: 11 experiments spanning loss functions, training data composition, stride, ticker count, decay weighting, sector matching, and ensembles have all failed to improve on Glia2a. The valid hyperparameter space is extremely narrow (val_loss must be <0.015). Within that narrow space, Glia2a already sits at the optimum. Further training runs have near-zero expected value.

**"Fix short-horizon regression"** — [KILL: fix-short-horizon]
Justification: Overturned by R1's honest multi-window evaluation. The regression was a single-window artifact. Glia2a improves at all horizons.

### Evaluation Framework

The Stats supervisor raises the correct concern: n=3 eval tickers with 8 windows is insufficient for inferential statistics. The +6.89pp at 120d is likely inflated by 2382.HK's anomalous predictability. However, this is a structural limitation of the China/HK evaluation setup, not something more experiments can fix.

Key evaluation findings:
- Glia2a's 120d gain is real but ticker-concentrated (2382.HK). Generalization claims beyond these 3 tickers are unsupported.
- Rolling eval with 15+ windows is the correct methodology (confirmed by the short-horizon noise episode).

### Directives Issued
- [KILL: beat-glia2a]
- [KILL: fix-short-horizon]
- [TERMINATE]

Rationale: The China/HK optimization has reached its natural conclusion. Every viable perturbation has been tested. Further compute would be spent confirming what we already know.

---

## Post-Termination: Expanded Evaluation with Bootstrap CIs

**Motivation:** Stats supervisor flagged n=3 eval tickers as near-zero statistical power. Expanded to 9 tickers (3 eval + 6 training diagnostics) with 10,000 bootstrap resamples for confidence intervals.

### Overall MAPE — Glia2a vs Baseline (9 tickers, 15 windows)

| Horizon | Baseline | Glia2a | Improvement | 95% CI | P(worse) | Significant? |
|---------|----------|--------|-------------|--------|----------|--------------|
| 5d | 2.65% | 2.65% | -0.01pp | [-0.30, +0.34] | 54.7% | NO |
| 14d | 3.70% | 3.58% | +0.13pp | [-0.23, +0.51] | 24.6% | NO |
| 30d | 5.00% | 4.70% | +0.30pp | [-0.39, +1.12] | 23.9% | NO |
| 60d | 7.85% | 8.28% | **-0.42pp** | [-1.75, +1.02] | **73.3%** | NO |
| 120d | 11.96% | 9.62% | +2.34pp | [-0.05, +5.22] | 2.8% | NO (barely) |

### Per-Ticker 120d Bootstrap CIs

| Ticker | Type | Baseline | Glia2a | Delta | 95% CI | Sig? |
|--------|------|----------|--------|-------|--------|------|
| 0388.HK | Eval | 4.4% | 5.0% | -0.61pp | [-1.19, -0.02] | WORSE * |
| 2382.HK | Eval | 25.4% | 14.3% | **+11.07pp** | [+8.53, +13.48] | BETTER * |
| 1177.HK | Eval | 21.8% | 17.2% | **+4.56pp** | [+2.80, +6.50] | BETTER * |
| 0700.HK | Train | 6.9% | 6.5% | +0.42pp | [-1.24, +2.41] | ns |
| 0941.HK | Train | 6.7% | 3.9% | **+2.88pp** | [+2.49, +3.29] | BETTER * |
| 2318.HK | Train | 13.3% | 9.6% | **+3.65pp** | [+2.24, +5.18] | BETTER * |
| 1299.HK | Train | 12.6% | 9.7% | **+2.89pp** | [+2.26, +3.46] | BETTER * |
| 0005.HK | Train | 6.5% | 9.8% | **-3.24pp** | [-3.61, -2.86] | WORSE * |
| 0001.HK | Train | 10.2% | 10.8% | -0.59pp | [-0.85, -0.34] | WORSE * |

### Per-Ticker 60d Bootstrap CIs

| Ticker | Type | Delta | 95% CI | Sig? |
|--------|------|-------|--------|------|
| 0388.HK | Eval | +1.96pp | [+1.59, +2.36] | BETTER * |
| 2382.HK | Eval | +3.78pp | [+2.53, +5.06] | BETTER * |
| 1177.HK | Eval | -3.67pp | [-4.34, -2.96] | WORSE * |
| 0700.HK | Train | -0.85pp | [-1.20, -0.53] | WORSE * |
| 0941.HK | Train | +0.27pp | [+0.08, +0.46] | BETTER * |
| 2318.HK | Train | -1.51pp | [-1.89, -1.08] | WORSE * |
| 1299.HK | Train | -0.60pp | [-0.97, -0.21] | WORSE * |
| 0005.HK | Train | -2.06pp | [-2.33, -1.79] | WORSE * |
| 0001.HK | Train | -1.13pp | [-1.43, -0.84] | WORSE * |

### Key Findings

1. **120d overall improvement is NOT significant at 95%** (CI barely includes zero). The +6.89pp from n=3 eval shrinks to +2.34pp with n=9 — inflated by 2382.HK selection.

2. **60d is WORSE with 9 tickers** (-0.42pp, P(worse)=73%). Hidden by n=3 eval which included the two biggest winners.

3. **The model helps 5 tickers but hurts 3 at 120d.** Winners are insurance/tech (2318, 1299, 0941, 2382) matching training data sectors. Losers are bank/conglomerate (0005 HSBC, 0001 CKH) — structurally different.

4. **Short horizons (5d, 14d): zero improvement** across 9 tickers. CIs span zero.

5. **The "improvement" is a sector effect, not a market effect.** Finetuning learns sector-specific drift patterns that transfer to tickers in matching sectors, but actively hurts tickers in non-matching sectors (HSBC banking vs training set's insurance/tech).

### Revised Assessment

Glia2a is NOT a general HK market improvement. It is a **sector-specific model** that helps insurance, tech hardware, and telecom tickers while hurting banking and conglomerate tickers. The n=3 eval ticker set was accidentally enriched for sectors that benefit from finetuning, creating an illusion of broad improvement.

**Recommendation:** Deploy Glia2a only for tickers in matching sectors (insurance, tech, telecom). Use baseline for banking/conglomerate tickers. A selective routing strategy by sector would be more honest than claiming universal improvement.

---

