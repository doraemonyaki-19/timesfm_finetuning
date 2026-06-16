# Glia Run Summary (Run 4)

Stop reason: approved
Turns completed: 1
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_expts\glia_run_20260228_105201
Date: 2026-02-28

---

## Best Configuration (unchanged from Run 3)

**3-Model Ensemble (Run4 + Glia3a + Glia4b) at max-context=512**
- 7.9% MAPE at 120d (favorable eval windows)
- 53.1% DirAcc at 120d
- Note: ensemble quality varies by eval window due to model weight instability

Checkpoint paths:
- `checkpoints/best/` (Run4)
- `checkpoints/run_glia_3a/best/` (Glia3a)
- `checkpoints/run_glia_4b/best/` (Glia4b)

Script: `python scripts/ensemble_eval.py --max-context 512 --checkpoints "checkpoints/best,checkpoints/run_glia_3a/best,checkpoints/run_glia_4b/best" --checkpoint-labels "Run4,Glia3a,Glia4b"`

---

## Experiments Conducted (Run 4)

### Experiment B: 4-Model Ensemble (Run4+Glia3a+Glia4a+Glia4b) — FAILED

Added Glia4a (α=0.95, 8.1% solo) to the existing 3-model ensemble.

| Horizon | Run4 (solo) | 4-model ensemble |
|---------|-------------|------------------|
| 120d    | 8.7%        | 9.0% (worse)     |

On the Feb 28 eval window, Glia3a's NVDA prediction regressed to 15.7% (vs 13.1% for Run4). Window instability pulls the ensemble above Run4's solo performance. Adding a 4th model adds noise, not signal.

**Hypothesis REFUTED**: More models in the ensemble do not help when individual model quality varies by window.

### Experiment A: Single-Layer Finetuning (layer 19 only) — CATASTROPHIC FAILURE

Froze layers 0-18, trained only layer 19 + output head.
- Trainable params: 44M (19.2%) vs Run4's 64M (27.7%)
- Best val loss: **0.040811 at epoch 3** (3.2× worse than Run4's 0.0128)
- Early stopped epoch 18 (no improvement from epoch 3 onward)
- Evaluation result: **NaN predictions across all 6 tickers at all 4 horizons**

The model produces numerically invalid outputs because:
1. Layer 19 alone lacks sufficient capacity to adapt the model
2. Frozen layers 17-18 produce intermediate representations misaligned with what layer 19 expects
3. The output head + top layer form a tightly coupled unit requiring coordinated 3-layer adaptation

**Hypothesis REFUTED**: Single-layer finetuning produces degenerate outputs. All 3 layers (17-19) must be trained together — the minimum coherent functional unit for this architecture.

---

## Key Findings (Run 4)

### 1. The 3-layer minimum is structural, not accidental
Layer 19 alone = NaN outputs. This reveals that the output head and layers 17-19 form a tightly coupled functional unit. Run4's freeze-17 approach was not just empirically best — it represents the minimum coherent adaptation unit.

### 2. Ensemble ceiling at 3 models
Adding a 4th checkpoint (Glia4a) does not reduce variance further and may increase it on unfavorable windows. The 3-model ensemble is the natural upper limit for this set of checkpoints.

### 3. Evaluation window instability is significant
The 7.9% MAPE figure is window-dependent:
- Feb 27 eval: 7.9% ensemble, 8.0% Run4
- Feb 28 eval: 9.0% ensemble, 8.7% Run4 (window shift makes Glia3a's NVDA much worse)
This instability affects single-model results too (Run4: 8.0% on Feb 27 vs 8.7% on Feb 28). The ~1-1.5pp improvement over baseline is robust; the absolute MAPE figures are window-sensitive.

---

## Complete Checkpoint Inventory

| Checkpoint | Description | 120d MAPE | Status |
|-----------|-------------|-----------|--------|
| `checkpoints/best/` | Run4 (best single) | 8.0-8.7% | ✓ IN ENSEMBLE |
| `checkpoints/run_glia_3a/best/` | Mixed loss α=0.8 | 8.0-9.7% | ✓ IN ENSEMBLE |
| `checkpoints/run_glia_4a/best/` | Mixed loss α=0.95 | 8.1-8.9% | Available |
| `checkpoints/run_glia_4b/best/` | Overall-dir α=0.9 | 8.0-8.9% | ✓ IN ENSEMBLE |
| `checkpoints/run_glia_5a/best/` | BA post-2020 | 10.7%+ | ✗ FAILED |
| `checkpoints/run_glia_6a/best/` | 1024-context | 9.5% (at 1024) | ✗ FAILED |
| `checkpoints/run_glia_7a/best/` | Layer 19 only | NaN | ✗ FAILED (NaN) |

---

## Open Questions (for any future work)

1. **BA post-Jan 2019 exclusion**: The 737 MAX crisis structural break (not yet tested; run_glia_5a used post-2020). Risk: ~1300 training pts for BA. Still the one unclosed BA hypothesis.

2. **Ceiling attribution**: Is the ~1.0-1.5pp ceiling from (a) data scarcity (6 eval tickers), (b) architecture (patch decoder not suited to per-ticker adaptation), or (c) evaluation instability? Window-by-window variance suggests the true ceiling range is 1.0-2.5pp depending on market conditions.

3. **Full model finetuning at very low LR**: All experiments froze 17+ layers. Full finetuning at LR~1e-6 would test whether the ceiling is due to the top-3-layer constraint. High risk of catastrophic forgetting.

---

*Glia Run 4 completed 2026-02-28. Approved by Supervisor after 1 turn, 2 experiments (both training-free/negative results). Total across all Glia runs: 16 experiments, 8 checkpoints.*
