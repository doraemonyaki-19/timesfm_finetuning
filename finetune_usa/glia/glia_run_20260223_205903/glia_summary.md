# Glia Run Summary

**Stop reason:** approved (Supervisor signalled [APPROVED] at end of Turn 2)
**Turns completed:** 2 (smoke test)
**Output dir:** C:\Users\ylchen\workspace\timesfm\finetune_expts\glia_run_20260223_205903

---

## Key Findings

### Root Cause of BA Degradation (Revised)
BA degradation (+0.8pp MAPE) is primarily a **pretrained model characteristic**, not a
finetuning artifact. BA was already the weakest ticker at baseline. Critically, BA's DirAcc
*improved* from 49.2% → 50.8% under Run 4 — the degradation is a **magnitude
miscalibration**, not a directional error.

The original hypothesis (trend-reversal → sign loss) was partially correct but misdiagnosed
the mechanism. The model overshoots on magnitude when trend context is flat, not because it
gets direction wrong, but because it amplifies a weak upward trend into a larger predicted
move than the target supports.

### Critical Self-Correction: Alpha Scale Mismatch
The Turn 1 proposal of `loss = 0.8*MSE + 0.2*sign_BCE` was **critically flawed**:
- Sign BCE ≈ 0.55 in log-space
- MSE ≈ 0.003 in log-space
- At alpha=0.2, sign loss would be **45× larger** than MSE — the MSE objective would be
  effectively discarded

Corrected alpha range: **{0.01, 0.02, 0.05}** for sign weight.

### DirAcc Baseline
Pretrained model DirAcc was 50.1-50.4% (essentially random). Finetuning improved to 51.4%.
This is a property of the foundation model on financial data — daily price direction is
near-random walk. The 60% go/no-go threshold set in the plan may be unachievable;
the diagnostic will clarify.

---

## Recommended Next Steps (for a real run)

### Immediate: Step 0 Diagnostic (~3 min, no training)
```
python scripts/sign_loss_diagnostic.py \
  --checkpoint checkpoints/best \
  --tickers "<Run 4 training tickers>" \
  --max-context 512 \
  --horizon 128
```
Compute: sign BCE value, sign agreement rate, gradient magnitude distribution.
**Go threshold:** sign agreement > 60% structured. If < 55%, abandon sign loss.

### If Go: Experiment Glia-1 (directional MSE weighting)
Preferred formulation (avoids scale mismatch):
```python
direction_wrong = ((pred_diff > 0) != (target_diff > 0)).float()
loss = (mse_loss_per_step * (1 + beta * direction_wrong)).mean()
# beta range: {0.5, 1.0, 2.0}
```

Hyperparameters: identical to Run 4 (AdamW lr=1e-4, freeze 17, 16x accum, 10 tickers,
from pretrained, not from Run 4 checkpoint).

### Stopping Criterion (Supervisor's Q3)
Net-loss threshold: if any ticker degrades by more than the average improvement
(i.e., max_degradation > mean_improvement), the run is a regression — stop and
revert to pure MSE. Define before seeing results to avoid post-hoc rationalization.

### Hypotheses Still Untested
- H2: Layer selection (gradient magnitude analysis, non-uniform freezing)
- H4: xreg horizon decay (`correction[t] *= exp(-λt)`)
- H5: Ticker similarity (LMT, RTX, NOC for BA)

---

## Supervisor Final Comments

> "The self-correction on alpha scaling was the key gate, and you caught it yourself
> before running anything. That is the behavior that prevents wasted compute."

Open questions flagged for the next real run:
1. Is sign agreement > 60% achievable given pretrained baseline ~50%? If not, should
   the go/no-go threshold be recalibrated to > baseline + 5pp?
2. What beta range prevents gradient destabilization in the multiplicative mask formulation?
3. Define aggregate stopping criterion before running: at what MAPE tradeoff is
   BA improvement not worth it?
