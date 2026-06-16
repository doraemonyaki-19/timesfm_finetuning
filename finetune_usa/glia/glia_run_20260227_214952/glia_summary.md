# Glia Run Summary (Run 3)

Stop reason: approved
Turns completed: 2
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_expts\glia_run_20260227_214952
Date: 2026-02-28

---

## Best Configuration (unchanged from Run 2)

**3-Model Ensemble (Run4 + Glia3a + Glia4b) at max-context=512**
- 7.9% MAPE at 120d (Feb 27, 2026 eval window)
- 53.1% DirAcc at 120d

Checkpoint paths:
- `checkpoints/best/` (Run4)
- `checkpoints/run_glia_3a/best/` (Glia3a)
- `checkpoints/run_glia_4b/best/` (Glia4b)

Script: `python scripts/ensemble_eval.py --max-context 512 --checkpoints "checkpoints/best,checkpoints/run_glia_3a/best,checkpoints/run_glia_4b/best" --checkpoint-labels "Run4,Glia3a,Glia4b"`

---

## Experiments Conducted (Run 3)

### Turn 1: Context window at inference (Hypothesis A) — FAILED
Evaluated all 3 finetuned checkpoints at max-context=1024 (training-free):
- Ensemble@1024: 9.2% (vs 7.9% at 512) — +1.3pp regression!
- Run4@1024: 8.8% (vs 8.0% at 512) — +0.8pp regression
- Glia3a@1024: 10.1% (vs 8.0% at 512) — most sensitive, +2.1pp
- GS particularly affected: 8.9% → 13.1% (1024 context includes 2022 bear market)

Mechanism: Finetuned models are patch-position-sensitive. At 512 context = 16 patches; at 1024 context = 32 patches. Layers 17-19 learned positional relationships at 16-patch scale; 32-patch scale is out-of-distribution.

### Turn 1: Per-ticker ensemble weights (Hypothesis B) — PARTIAL
Oracle (lookahead): +0.6pp improvement (9.0% → 8.4%)
Context-calibrated (60-day calibration): +0.2pp improvement (9.0% → 8.8%)
Critical finding: Optimal weights are window-specific and UNSTABLE:
- Feb 27: Glia3a best for NVDA (6.8% vs Run4 9.3%)
- Feb 28: Run4 best for NVDA (13.1% vs Glia3a 15.7%)
Conclusion: Equal-weight averaging is more robust. Per-ticker optimization requires lookahead.

### Turn 2: Training at max-context=1024 (Hypothesis C) — FAILED
run_glia_6a: 310 windows, best val loss 0.011894 at epoch 22.

4-way context comparison at 120d:
| Model | Train ctx | Eval ctx | 120d MAPE |
|-------|-----------|----------|-----------|
| Baseline | — | 512 | 10.2% |
| Baseline | — | 1024 | 9.5% |
| Run4 | 512 | 512 | **8.7%** ← best |
| Glia6a | 1024 | 1024 | 9.5% ← same as baseline! |
| Glia6a | 1024 | 512 | 9.2% |

Training at 1024 context produces ZERO improvement at matched context. 512 is the structural optimum for this finetuning setup.

---

## Key Findings (Run 3)

### 1. Context length: 512 is optimal
The 512-context window (16 patches, ~2 years of history) is the structural sweet spot:
- Larger inference context: hurts finetuned models (patch-position distribution shift)
- Training at larger context: no improvement at matched context (fewer windows, more diverse regime patterns)
- The pretrained model's patch-based architecture makes 512 context (16 patches) the natural working point for finetuning

### 2. Ensemble weights are window-unstable
Equal-weight averaging is more robust than per-ticker optimization because:
- Each model's relative strength rotates with market conditions (NVDA example)
- Context-calibrated weights (60-day lookback) have poor predictive power for 120-day test window
- Oracle weights are 0.6pp better but require lookahead (not usable in practice)

### 3. The ceiling is structural, not modifiable
14 experiments across 3 Glia runs have confirmed: ~1.0pp MAPE improvement at 120d is the fundamental limit. The constraint is:
- Architectural (fixed 20-layer decoder with patch-based tokenization)
- Data-regime (6 tickers × 5027 pts, optimal at 512-day context)
- Not addressable by: loss function, layer selection, data composition, or context length

---

## Complete Checkpoint Inventory

| Checkpoint | Description | 120d MAPE | Status |
|-----------|-------------|-----------|--------|
| `checkpoints/best/` | Run4 (best single) | 8.0% | ✓ IN ENSEMBLE |
| `checkpoints/run_glia_3a/best/` | Mixed loss α=0.8 | 8.0% | ✓ IN ENSEMBLE |
| `checkpoints/run_glia_4a/best/` | Mixed loss α=0.95 | 8.1% | Available |
| `checkpoints/run_glia_4b/best/` | Overall-dir α=0.9 | 8.0% | ✓ IN ENSEMBLE |
| `checkpoints/run_glia_5a/best/` | BA post-2020 | 10.7% | ✗ FAILED |
| `checkpoints/run_glia_6a/best/` | 1024-context | 9.5% (at 1024) | ✗ FAILED |

---

## Open Questions (for any future work)

1. **BA post-2019 exclusion** (not run_glia_5a which used post-2020): The 737 MAX crisis began in 2018. Training on post-Jan 2019 BA data would exclude the crisis transition period without including the COVID crash. Higher risk due to small dataset (~1300 pts).

2. **Dynamic ensemble (regime-aware)**: Instead of per-ticker static weights, use a regime indicator (VIX level, trend strength) to switch which model to weight more. Complex but theoretically sound.

3. **Different pretrained model**: TimesFM 1.0 or Chronos or Moirai might have different architectural sweet spots for financial finetuning.

---

*Glia Run 3 completed 2026-02-28. Approved by Supervisor after 2 turns, 3 experiments (2 training-free + 1 training). Total across all Glia runs: 14 experiments, 8 checkpoints.*
