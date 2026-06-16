# Glia Run Log

Started: 2026-02-23 20:59:03
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_expts\glia_run_20260223_205903
Max turns: 2

---

## Turn 1 — Researcher

### 1. What I Read and Explored

**Source code reviewed:**
- `finetune_timesfm.py` (614 lines) — training pipeline with log-transform + random masking, training_step (patching, ReVIN, MSE loss in log-space), AdamW/SGD, early stopping.
- `evaluate_forecast.py` (509 lines) — MAPE, DirAcc, MSE, MAE, RMSE metrics, optional xreg, multi-horizon.
- `experiment_log.md` and `README.md` — full run history with per-epoch logs.
- Verified Run 4 best checkpoint exists (model.safetensors, 925MB).

### 2. Lightweight Analysis: BA Degradation Root Cause

**Volatility analysis:** BA (z=+1.66) has very similar volatility to GS (z=+1.58). GS improved -1.3pp in Run 4. NVDA (z=+3.82) is far more volatile yet improved most. **Volatility does NOT explain BA degradation.**

**Correlation with training tickers:**

| Eval Ticker | Best Training Match | Match Corr | Run 4 MAPE delta |
|-------------|---------------------|------------|------------------|
| GS          | JPM                 | 0.819      | -1.3pp ✓         |
| CVX         | XOM                 | 0.831      | -1.3pp ✓         |
| BA          | ^GSPC               | 0.602      | +0.8pp ✗         |
| NVDA        | ^GSPC               | 0.658      | -1.6pp ✓         |

Lack of a strong training analogue is partial but insufficient — NVDA also lacks one yet improves most.

**CRITICAL finding — Trend dynamics:**

| Ticker | Context Trend | Target Trend | Pattern        |
|--------|--------------|-------------|----------------|
| NVDA   | +300.8%      | +12.2%      | Continues      |
| GS     | +139.4%      | +22.7%      | Continues      |
| CVX    | +11.5%       | +16.7%      | Continues      |
| KO     | +21.2%       | +18.4%      | Continues      |
| UNH    | -36.3%       | -7.4%       | Continues      |
| **BA** | **+2.1%**    | **-2.9%**   | **REVERSAL** ← |

BA is the only ticker with a **trend reversal** in its eval window. Finetuning reinforces the model's "trends persist" prior — helpful for 5/6 tickers, harmful for BA.

### 3. Selected Hypothesis: H3 — Mixed Loss (MSE + Sign Loss)

Root cause of BA degradation is over-learning trend continuation from MSE-only training. Fix: `loss = 0.8 * MSE + 0.2 * sign_BCE_loss`. This directly penalizes wrong-direction predictions without sacrificing magnitude accuracy.

Modification site: `training_step()` at line ~260 in `finetune_timesfm.py`.

### 4. Recommended Next Experiment

Run Glia-1: identical to Run 4 hyperparameters (AdamW lr=1e-4, freeze 17, 16x accum, 10 tickers, from pretrained) but with mixed loss alpha=0.8. Sweep alpha={0.7, 0.8, 0.9} as runs glia_1a/b/c.

Expected: BA MAPE improves or stops degrading; DirAcc improves from 51.4% toward 53-55%; overall MAPE < 7.0%.

---

## Turn 1 — Supervisor

The trend-reversal finding for BA is sharp and well-reasoned. Before you run the alpha sweep, a few questions worth sitting with:

**On the diagnosis itself:** You've shown BA has a context-to-target reversal, but how confident are you this is a *training* artifact versus a *pretrained model* artifact? In other words, did BA already degrade before finetuning, or did finetuning specifically push it in the wrong direction? Do you have Run 4's zero-shot (unfinetuned) BA numbers to compare against?

**On the proposed fix:** Sign BCE loss will penalize wrong-direction predictions, but BA's issue is a trend *reversal* — the model confidently continues upward when it should turn down. Is sign correctness the right lever, or could the model be miscalibrated on *magnitude of deviation from trend* rather than direction per se? What would the loss surface look like for a reversal case under your mixed formulation?

**On evaluation scope:** Your table shows 6 tickers. If BA improves under sign loss but another ticker — say one with a strong continuing trend — degrades because the sign penalty dampens its trend-following behavior, how will you detect that tradeoff? Is your eval set large enough to see it?

**On the alpha sweep design:** You're proposing alpha = {0.7, 0.8, 0.9} — three full training runs. That's expensive. Is there a cheaper proxy experiment you could run first to validate whether the sign loss gradient is even meaningful at your training data scale? For instance, what does the sign BCE loss value *look like on the training set* during a Run 4 replay, before you change anything?

**On what you're not yet considering:** You've diagnosed one ticker's degradation. But DirAcc across all tickers is 51.4% — barely above random. Is BA the signal, or is it a symptom of a broader directional weakness in the model? Have you looked at whether the pretrained model's directional accuracy was already low before any finetuning?

One concrete question to answer before running anything: can you compute what the sign BCE loss value *would have been* during Run 4 training, using the saved checkpoints and training data, without retraining? That one diagnostic could tell you whether the gradient signal you're betting on is real.

---

## Turn 2 — Researcher

### Q1: Finetuning vs Pretrained Artifact
**BA degradation is largely a pretrained artifact, amplified modestly by finetuning.**

| Source | BA MAPE | BA DirAcc |
|--------|---------|-----------|
| Pretrained (Run 3/4 window) | 8.2% | 49.2% |
| Run 4 finetuned | 9.0% | 50.8% |

BA was already the worst ticker at baseline. The +0.8pp MAPE increase is within natural eval-window variability (BA's own MAPE varies 1.1pp across windows). Critically: BA DirAcc *improved* from 49.2% → 50.8% under finetuning. The problem is magnitude miscalibration, not directional error.

### Q2: Sign Correctness vs Magnitude — Revised Diagnosis
The model gets direction slightly better after finetuning but overshoots on magnitude. Sign BCE is the wrong lever. A magnitude-dampening loss (penalizing large deviations from last observed value) or directional MSE weighting would be more appropriate.

### Q3: Tradeoff Risk on Strong-Trend Tickers
Low risk. Sign BCE gradient is self-regulating: ~0.2 when model is correct (NVDA), ~0.6 when wrong (BA). At alpha=0.01-0.05 it would be a weak regularizer, not a dominant objective.

### Q4: Scale Mismatch — Critical Correction
Sign BCE ≈ 0.55 in log-space, vs MSE ≈ 0.003. The original alpha=0.2 proposal would make sign loss 45× larger than MSE — catastrophically wrong. **Revised alpha range: {0.01, 0.02, 0.05}.** Recommend running a 3-minute forward-pass diagnostic on Run 4 checkpoint first to measure actual sign BCE and go/no-go on the approach.

### Q5: Broader Directional Weakness
Pretrained DirAcc was 50.1-50.4% (essentially random). Finetuning improved it to 51.4%. BA is a symptom of the foundation model's general inability to predict daily direction on financial data, not a finetuning-specific failure.

### Revised Experiment Plan
1. **Step 0 (3 min, no training):** Forward-pass diagnostic on Run 4 checkpoint — compute actual sign BCE and sign agreement rate. Go/no-go threshold: sign agreement > 60% (structured) vs < 55% (noise).
2. **Step 1:** If go, alpha sweep {0.01, 0.02, 0.05} with directional-MSE weighting variant as safer alternative.
3. **Step 2:** Evaluate on both 6-ticker and 50-ticker sets for tradeoff visibility.

**Alternative loss:** `loss = MSE * (1 + beta * direction_disagreement_mask)` — preserves MSE gradient landscape, no scale mismatch, doubling penalty on wrong-direction steps.

---

## Turn 2 — Supervisor

[APPROVED]

The self-correction on alpha scaling was the key gate, and you caught it yourself before running anything. That is the behavior that prevents wasted compute.

Three short questions to sharpen execution before you run Step 0:

**1. What is your go/no-go criterion's theoretical basis?**
You set sign agreement > 60% as the "structured" threshold. Why 60% specifically? If the pretrained model sits at ~50% and finetuning moves it to ~51.4%, is 60% achievable at all — and if not, does that mean the entire directional-loss approach should be abandoned regardless of the diagnostic result?

**2. The alternative loss `MSE * (1 + beta * direction_disagreement_mask)` — what is beta's scale?**
You solved the alpha scale mismatch for sign BCE. Have you checked whether the multiplicative mask creates a different scale problem? When direction is wrong, MSE is multiplied by `(1 + beta)`. What range of beta keeps the gradient perturbation small enough to not destabilize the MSE baseline you've already achieved?

**3. Evaluation scope sequencing.**
You plan to evaluate on both 6-ticker and 50-ticker sets. The 6-ticker set is your signal; the 50-ticker set is your generalization check. But if the directional loss improves BA while degrading NVDA, at what aggregate metric level would you call that a net loss and stop? Defining that stopping criterion now prevents post-hoc rationalization after seeing results.

The plan is ready to execute. Run the diagnostic.

---

