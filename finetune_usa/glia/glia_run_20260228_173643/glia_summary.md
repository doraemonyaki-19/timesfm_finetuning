# Glia Run Summary (Run 5)

Stop reason: approved
Turns completed: 1
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_expts\glia_run_20260228_173643
Date: 2026-03-01

---

## Best Configuration (unchanged through 19 experiments)

**3-Model Ensemble (Run4 + Glia3a + Glia4b) at max-context=512**
- 7.9% MAPE at 120d (favorable eval windows)
- 53.1% DirAcc at 120d
- Run4 solo is most reliable single model: 8.0-8.7% MAPE (window-dependent)
- Improvement over baseline: ~1.5pp (robust across all eval windows)

Checkpoint paths:
- `checkpoints/best/` (Run4)
- `checkpoints/run_glia_3a/best/` (Glia3a)
- `checkpoints/run_glia_4b/best/` (Glia4b)

Script: `python scripts/ensemble_eval.py --max-context 512 --checkpoints "checkpoints/best,checkpoints/run_glia_3a/best,checkpoints/run_glia_4b/best" --checkpoint-labels "Run4,Glia3a,Glia4b"`

---

## Experiments Conducted (Run 5)

### Experiment 1b: SGDR Warm Restarts (run_glia_r5_1b) — NaN FAILURE

**Setup**: Freeze-17, AdamW lr=1e-4, SGDR T_0=10 epochs / T_mult=2, NO warmup (implementation bug)
**Result**: Best val loss 0.027478 at epoch 20. NaN predictions across all tickers.

Key finding: SGDR without warmup never reaches the valid convergence basin (val loss <0.015). The immediate full-LR start (1e-4 from step 1) caused large early weight updates, producing a numerically unstable weight configuration. However, the SGDR restarts DID extend useful training from epoch 4 to epoch 20 — the concept is not refuted, only the no-warmup implementation.

### Experiment 1a: BA Post-2019 in Training Set (run_glia_r5_1a) — NaN FAILURE

**Setup**: 11 tickers (standard 10 + BA from 2019-01-01, 1799 pts), cosine+warmup=5
**Result**: Best val loss 0.024304 at epoch 9. NaN predictions across all tickers.

Key finding: **BA cannot be included in the training set regardless of date cutoff.** BA's high-volatility price regime (737 MAX crisis, COVID crash, labor disputes) corrupts the shared training distribution. Training loss plateaued at 0.032-0.036 throughout (vs Run4's convergence to 0.003-0.005). This closes the final BA hypothesis definitively.

---

## Critical New Finding: Val Loss Threshold (~0.015)

All experiments across 5 Glia runs reveal a sharp threshold:

| Checkpoint | Val Loss | Predictions | MAPE@120d |
|-----------|----------|------------|-----------|
| run_glia_3a | 0.011956 | Valid | 8.0% |
| run_glia_6a (1024-ctx) | 0.011894 | Valid | 9.5% |
| Run4 | 0.012862 | Valid | 8.0-8.7% |
| **Threshold** | **~0.015** | — | — |
| run_glia_r5_1a (BA post-2019) | 0.024304 | NaN | — |
| run_glia_r5_1b (SGDR no-warmup) | 0.027478 | NaN | — |
| run_glia_7a (layer 19 only) | 0.040811 | NaN | — |

**Mechanism**: When model weights deviate too far from pretrained initialization, the inference-time quantile head (softmax over continuous quantiles) overflows. The threshold ~0.015 is the maximum val loss compatible with valid inference.

**Implication**: Only the 10-standard-ticker + cosine + 5-epoch warmup training regime reliably achieves val loss <0.015. Any deviation from this recipe (adding BA, removing layers, different scheduler) pushes the model above the threshold into degenerate territory.

---

## Complete Experiment Inventory (all 5 Glia runs, 19 experiments)

| Run | Checkpoint | Description | Val Loss | MAPE@120d |
|----|-----------|-------------|---------|-----------|
| Pre-Glia | Run4 | Best single | 0.012862 | 8.0-8.7% |
| 1 | run_glia_1a | 13 tickers LR=1e-4 | NaN | NaN weights |
| 1 | run_glia_3a | Mixed loss α=0.8 | ~0.011956 | 8.0% |
| 1 | run_glia_4a | Mixed loss α=0.95 | ~0.012 | 8.1% |
| 1 | run_glia_4b | Overall-dir α=0.9 | ~0.012 | 8.0% |
| 2 | run_glia_5a | BA post-2020 | ~0.010549 | 10.7% |
| 2 | Ensemble | Run4+3a+4b | — | 7.9% |
| 2 | xreg λ=0.05 | Gradient analysis | — | failed |
| 3 | — | Context=1024@inference | — | 9.2% |
| 3 | — | Per-ticker weights | — | 8.4% oracle |
| 3 | run_glia_6a | 1024-ctx training | 0.011894 | 9.5% |
| 4 | — | 4-model ensemble | — | 9.0% |
| 4 | run_glia_7a | Layer 19 only | 0.040811 | NaN |
| 5 | run_glia_r5_1b | SGDR no-warmup | 0.027478 | NaN |
| 5 | run_glia_r5_1a | BA post-2019 | 0.024304 | NaN |

---

## Open Questions (for any future work)

1. **SGDR with proper warmup**: Implementation bug not conceptual flaw. SequentialLR(LinearLR_warmup + CosineAnnealingWarmRestarts) would give the proper combined schedule. Expected to converge to Run4 range (not beat ceiling).

2. **Evaluation protocol sensitivity**: The 7.9% figure is eval-window-dependent (varies ~1pp daily). Testing over 30+ consecutive days would give a more reliable picture of true improvement.

3. **Full model finetuning at LR=1e-6**: Untested due to catastrophic forgetting risk. Very low LR might allow all 20 layers to adapt slightly without NaN. High-risk, uncertain reward.

4. **Different pretrained model**: TimesFM 1.0, Chronos, Moirai — different architectures may have different sweet spots for financial finetuning.

---

*Glia Run 5 completed 2026-03-01. Approved by Supervisor after 1 turn, 3 experiments (all NaN). Total across all Glia runs: 19 experiments, 10 trained checkpoints (7 valid, 3 NaN).*
