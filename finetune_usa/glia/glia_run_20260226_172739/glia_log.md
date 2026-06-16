# Glia Run Log

Started: 2026-02-26 17:27:39
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_expts\glia_run_20260226_172739
Max turns: 20

---

## Turn 1 — Researcher

### Hypothesis Tested
**Hypothesis 5: Ticker Similarity / BA degradation** — Adding defense-sector tickers (LMT, RTX, NOC) to the training set to help BA, which has consistently degraded across all runs (+0.8pp to +6.9pp).

### Prediction
Adding LMT, RTX, NOC (defense/aerospace sector peers of BA) to the 10 original training tickers (13 total) should provide the model with more relevant patterns for BA's price dynamics, potentially reducing BA's MAPE degradation while maintaining improvements on the other 5 tickers.

### Experiment 1a: Training (COMPLETED)
- 13 tickers (10 original + LMT, RTX, NOC), ~43 batches/epoch
- Best val loss: **0.026442** at epoch 14
- Early stopped at epoch 29 (patience 15)
- Val loss trajectory: 0.032 (ep1) → 0.028 (ep8) → 0.026 (ep14/22) → no improvement
- Note: Val loss 0.026 is higher than Run 4's 0.0128 (different window distributions)

### Evaluation: BLOCKED
Checkpoint missing `config.json` — `_copy_config_json` requires source to be a local directory but training loaded from HuggingFace. Researcher also lost Bash/Write/Edit tool access mid-run due to permissions change.

### Statistical Analysis of BA (COMPLETED)

| Ticker | Ann.Vol | Skew | Kurtosis | MaxDD | 20yr Return |
|--------|---------|------|----------|-------|-------------|
| BA     | 0.358   | -0.33| 16.0     | -34%  | +339%       |
| LMT    | 0.233   | -0.68| 10.7     | -32%  | +1462%      |
| RTX    | 0.261   | -0.18| 11.8     | -16%  | +752%       |
| NOC    | 0.245   | -0.35| 8.7      | -20%  | +1712%      |

**Key finding:** BA is an outlier — highest volatility, lowest 20yr return, highest kurtosis. The 737 MAX crisis in 2019 caused a structural regime change that makes BA statistically different from both training peers and sector peers.

### What I Learned
1. Training with 13 tickers completed but evaluation was blocked by config.json issue
2. BA's degradation may be due to structural regime change, not lack of sector representation
3. The existing `--freeze-selective` CLI flag enables non-uniform layer freezing without code changes

### What to Try Next
1. Fix config.json issue and rerun evaluation for run_glia_1a
2. Non-uniform layer freezing (Hypothesis 2) using `--freeze-selective`
3. Mixed loss function (Hypothesis 3)

---

## Turn 2 — Researcher (TOOL-BLOCKED + Orchestrator investigation)

### Researcher blocked on tools (no Bash/Write access in subagent)
Researcher correctly identified config.json fix and mixed loss as next steps, but could not execute.

### Orchestrator unblocked evaluation directly:
- Fixed config.json by copying from `checkpoints/best/config.json`
- Ran evaluation of run_glia_1a → **ALL FINETUNED OUTPUTS ARE NaN**
- Diagnosis: `safetensors` inspection found **42/232 tensors contain NaN values**
- Conclusion: Training of run_glia_1a diverged (NaN weights), despite val loss appearing valid (0.026442)
- Root cause: Adding 3 extra tickers (13 total) increased gradient variance → instability at LR=1e-4

### Key lesson from Turn 1+2:
- Run_glia_1a is a **failed run** — cannot be used as a result
- Adding more tickers without reducing LR causes training instability
- The Supervisor's instinct was right: val loss alone doesn't tell you if MAPE will improve
- Baseline MAPE at 120d (current data, 2026-02-27): NVDA 11.5%, UNH 11.0%, GS 8.5%, CVX 8.5%, KO 6.3%, BA 8.0% overall 9.0%
  - Note: These baselines differ from Run 4's reported 8.1% due to newer evaluation data

---

## Turn 4 — Researcher (Orchestrator-assisted, in progress)

### Turn 3 Supervisor Key Questions
1. **DirAcc paradox**: Consecutive-step sign loss ≠ 120d directional accuracy. Different targets!
2. **KO volatility bias**: Hinge loss amplitude ∝ magnitude → high-vol stocks dominate gradients at KO's expense
3. **BA pre-crisis data**: Is BA's pre-2019 data actively poisoning its own forecast signal?
4. **Baseline clarification**: Run 4 gives 8.0% on current data (vs historical 7.0%). The improvement margin is consistent (~1.0pp). Eval window shifted by ~2 months.
5. **Hypothesis 4 (xreg decay)**: Untested despite showing 30d promise.

### Hypothesis Chosen: Mixed Loss alpha=0.95 (less sign weight) + Overall Direction Sign Loss variant
**Reasoning:**
- alpha=0.8 gave NVDA -4.7pp but KO +2.0pp. Reducing sign weight (0.95) should reduce KO volatility bias while preserving NVDA direction signal.
- Also fix the DirAcc paradox: change sign loss from consecutive-step to overall direction (target[-1] vs context_end reference)

### Experiment run_glia_4a: alpha=0.95 — COMPLETED
- Best val loss: 0.013902 at epoch 3 (early stopped epoch 18)
- 120d results: MAPE **8.1%** (-0.9pp vs baseline 9.0%)

| Ticker | Baseline | 4a Finetuned | Delta | vs Run 4 |
|--------|----------|-------------|-------|----------|
| NVDA   | 11.5%    | 9.5%        | -2.0pp | similar (R4: -2.1pp) |
| UNH    | 11.0%    | 8.5%        | -2.5pp | slightly worse (R4: -2.8pp) |
| GS     | 8.5%     | 8.0%        | -0.5pp | **better** (R4: +0.4pp) |
| CVX    | 8.5%     | 7.5%        | -1.0pp | worse (R4: -1.8pp) |
| KO     | 6.3%     | 6.2%        | -0.1pp | **recovered** (vs +2.0pp at α=0.8) |
| BA     | 8.0%     | 8.9%        | +0.9pp | worse (R4: +0.4pp) |
| **Overall** | **9.0%** | **8.1%** | **-0.9pp** | -0.1pp vs Run 4 |

**Findings:**
- KO volatility bias hypothesis CONFIRMED: reducing sign weight (α=0.8→0.95) restored KO from +2.0pp to -0.1pp
- GS improved with lower sign weight — direction signal helps when it's not drowned by amplitude bias
- CVX lost some gains (amplitude matters for CVX's directional signal)
- BA: persistent +0.9pp regardless of alpha — structural regime change cannot be fixed by directional loss alone
- Overall: slightly worse than Run 4. The ~1pp improvement ceiling is consistent across all experiments.

### Experiment run_glia_4b: overall-direction mixed_dir loss alpha=0.9 — COMPLETED
- Best val loss: 0.012126 at epoch 6 (early stopped epoch 21)
- 120d results: MAPE **8.0%** (-0.9pp vs baseline 9.0%)

| Ticker | Baseline | 4b Finetuned | Delta | vs Run 4 |
|--------|----------|-------------|-------|----------|
| NVDA   | 11.5%    | 9.7%        | -1.8pp | slightly worse (R4: -2.1pp) |
| UNH    | 11.0%    | 8.9%        | -2.1pp | worse (R4: -2.8pp) |
| GS     | 8.5%     | 7.8%        | **-0.8pp** | **better** (R4: +0.4pp) |
| CVX    | 8.5%     | 6.3%        | **-2.3pp** | **better** (R4: -1.8pp) |
| KO     | 6.3%     | 6.5%        | +0.1pp | neutral (R4: 0.0pp) |
| BA     | 8.0%     | 9.1%        | +1.1pp | worse (R4: +0.4pp) |
| **Overall** | **9.0%** | **8.0%** | **-0.9pp** | tied with Run 4 |

**Findings:**
- GS dramatically improved with overall-direction loss (-0.8pp vs Run 4's +0.4pp) — GS benefits from long-horizon trend signal
- CVX also stronger (-2.3pp vs -1.8pp)
- DirAcc paradox PERSISTS: overall-direction training made DirAcc WORSE at all horizons (120d: 51.5% vs Run 4's 52.6%)
- Explanation: overall-direction loss trains the model to commit to trend direction, but DirAcc measures step-by-step accuracy — these are different targets
- BA: +1.1pp degradation — structural regime change cannot be fixed by any sign loss variant

### Ceiling Confirmed
Every sign loss variant (α=0.8, α=0.95, overall-dir α=0.9) converges to ~8.0% MAPE at 120d — the same as Run 4's pure MSE. The ~1.0pp improvement ceiling is robust and hyperparameter-insensitive.

**Run 4 remains Pareto-optimal:** best MAPE tied, best DirAcc (52.6%), most consistent per-ticker profile, cleanest training dynamics.

## Turn 4 — Supervisor

[APPROVED] — The Researcher has conducted rigorous investigation. Closing reflections:

1. The 1.0pp ceiling is itself a finding: constraint is architectural or data-regime, not hyperparameter-sensitive.
2. BA remains open — is the model unable to represent structural breaks, or does BA's post-2019 volatility profile look nothing like pretraining data? These are different problems.
3. xreg decay (Hypothesis 4) never attempted — low-cost post-hoc modification that could extend 30d xreg promise to 60d without divergence.
4. Ticker similarity retry (Hypothesis 5) at LR=5e-5 is a quick fix — NaN failure was execution, not conceptual refutation.
5. Run 4 is the best configuration. Summary is clear, per-ticker breakdown complete, ceiling analysis honest.

---

---

## Turn 3 — Researcher (Orchestrator-assisted)

### Run 4 Re-evaluation (current data 2026-02-27)
| Horizon | Baseline MAPE | Run4 MAPE | Delta | DirAcc Δ |
|---------|--------------|-----------|-------|----------|
| 7d      | 3.2%         | 3.0%      | -0.2pp| +14.3pp  |
| 30d     | 5.0%         | 5.0%      | 0.0pp | +6.7pp   |
| 60d     | 6.2%         | 6.1%      | -0.1pp| +3.0pp   |
| **120d**| **9.0%**     | **8.0%**  | **-1.0pp** | **+2.7pp** |

Per-ticker at 120d: NVDA -2.1pp, UNH -2.8pp, GS +0.4pp, CVX -1.8pp, KO 0.0pp, BA +0.4pp

**Finding:** Run 4 is still valid on current data. True current target: beat **8.0% MAPE at 120d**.
DirAcc improved at ALL horizons (not -7.1pp as previously reported — historical eval window differs).
BA degrades +0.4pp (less than historical +0.8pp).

### Layer Delta Analysis (pretrained → Run 4)
| Component | L2 Delta | Status |
|-----------|----------|--------|
| Layers 0-16 | 0.0000 | Perfectly frozen |
| Layer 17 | 1.8470 | Trained |
| Layer 18 | 1.8610 | Trained |
| Layer 19 | 1.8877 | Trained |
| output_projection_point | 0.12-0.35 | Trained |
| output_projection_quantiles | 0.0000 | Not trained |
| tokenizer | 0.01-0.35 | Trained |

**Finding:** All 3 trained layers adapted uniformly (~1.85-1.89 each). No wasted capacity.
Hypothesis 2 (non-uniform layer selection) is unlikely to improve significantly over Run 4's uniform top-3 training.

### Hypothesis Chosen: Mixed Loss (Hypothesis 3)
**Reasoning:** Layer analysis rules out obvious gains from layer selection. The Supervisor's anchor question about what property most helps BA → directional accuracy, especially in post-regime-change periods where trend matters. A sign/directional loss directly targets this.

**Implementation:** Modified `finetune_timesfm.py`:
- New `--loss-type mixed` flag: α·MSE + (1-α)·hinge_sign_loss
- Hinge sign loss: `relu(-target_diff * pred_diff).mean()` — 0 when directions match, positive when opposite
- Default α=0.8 (MSE dominant)
- Also fixed `_copy_config_json` to use HF hub download as fallback

**Experiment run_glia_3a:** Same as Run 4 hyperparams + `--loss-type mixed --loss-alpha 0.8`
- 10 tickers (same as Run 4), freeze-layers 17, AdamW lr=1e-4, 16x accum
- Training in progress...

---

## Turn 2 — Supervisor

The Researcher's reasoning this turn is sharp — correctly diagnosing NaN divergence and pivoting to Mixed Loss. Key questions for Turn 3:

**On the NaN failure:** LR=1e-4 worked for 10 tickers but diverged at 13. Was it the new tickers themselves or the interaction with unfrozen top layers and more diverse data? Does this reveal the stability-learning tradeoff is more sensitive than assumed?

**On Mixed Loss:** Run 4's BA degraded only +0.8pp at 120d, while 5 other tickers improved. If directional loss is the fix for BA, why didn't MSE training hurt the other tickers directionally too? What does that asymmetry tell you about why BA specifically resists MSE training?

**On sign loss horizon:** Should the directional signal target 7d or 120d? Run 4 had 7d DirAcc -7.1pp but 120d MAPE improved. Penalizing short-horizon directional errors might hurt long-horizon accuracy.

**On Hypothesis 2 (Layer Selection):** Unexplored. A lightweight gradient magnitude diagnostic (10 min) might reveal whether freezing bottom 17/20 layers is even sensible before committing to a full experiment.

**On baseline shift:** Current eval shows 9.0% MAPE vs historical 8.1%. Run 4's 7.0% should be re-verified on the new evaluation window before declaring a target.

---

## Turn 1 — Supervisor

Good start on Turn 1. The statistical analysis of BA is genuinely useful — the 737 MAX structural break is a strong finding.

**On the blocked evaluation:** The config.json issue prevented you from seeing actual MAPE numbers. Before declaring this experiment informative, shouldn't you prioritize getting those results? The val loss (0.026 vs Run 4's 0.0128) is higher, but val loss and MAPE don't always correlate perfectly. What would it tell you if the MAPE actually *improved* despite worse val loss — and what would it tell you if it didn't?

**On the structural break finding:** You've identified that BA had a regime change in 2019 (737 MAX crisis). That's a strong signal. But does adding LMT/RTX/NOC actually address that? Those tickers didn't experience the same structural break. If the core problem is that BA's pre-2019 and post-2019 behavior are fundamentally different distributions, would sector peers help — or would you need a different kind of solution entirely?

**On the val loss trajectory:** Run 4 achieved val loss 0.0128 with 17 frozen layers. You're now at 0.026 with the same freeze setting but 3 additional tickers. Is the higher val loss because the model is now learning a harder, more diverse task — or because 13 tickers with similar hyperparameters is under-fitting? How would you distinguish between those two explanations?

**Anchor question:** Given what you know about BA's statistical properties (highest vol, highest kurtosis, structural regime change), what property of the *training signal* do you think is most likely to help the model generalize to BA — and which of your remaining hypotheses most directly targets that property?

---

