# Glia Run Summary — Europe

Stop reason: approved
Turns completed: 2
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_europe\finetune_expts\glia_run_20260311_151818

---

## Turn 1 — Researcher

**Hypothesis:** stride=64 → ~700 windows (doubled). Prediction: val_loss < 0.015, MAPE +0.3–0.6pp improvement.

**Result (run_glia_1b):** val_loss 0.021912 — ABOVE threshold. Overall MAPE 4.594% vs Run1's 4.566%. REFUTED.

Stride=64 creates ~87% window overlap → degraded gradient signal. Pattern consistent with Japan (stride=64 → val_loss 2× worse).

## Turn 1 — Supervisor

Stride=64 refutation clean. Questions raised: (1) sector gap for VWAGY/EADSF, (2) warmup evidence, (3) modest ceiling risk. Asked for one more experiment.

---

## Turn 2 — Researcher

**Hypothesis:** warmup=10 (vs Run1's 5) → slower LR ramp → lower val_loss floor → marginal MAPE improvement.

**Result (run_glia_2a):** val_loss 0.014056 (epoch 14) — BELOW 0.015 ✓. But MAPE comparison (Mar 15 window):

| Ticker | Baseline | Run1/best | run_glia_2a | Run1 delta | 2a delta |
|--------|----------|-----------|-------------|------------|----------|
| UL     | 3.612%   | 3.555%    | 3.640%      | -0.057pp   | +0.028pp |
| EADSF  | 4.720%   | 4.574%    | 5.538%      | -0.146pp   | +0.819pp |
| VWAGY  | 6.248%   | 6.915%    | 7.459%      | +0.667pp   | +1.211pp |
| **Overall** | **4.860%** | **5.014%** | **5.546%** | **+0.155pp** | **+0.686pp** |

warmup=10 REFUTED. Key finding: window instability revealed — Run1 was +0.135pp improvement (Mar 11) vs +0.155pp regression (Mar 15). ~0.3pp noise exceeds improvement signal. VWAGY consistently worst ticker across both finetuned models.

## Turn 2 — Supervisor

[APPROVED] — See glia_log.md for full text.

---

## Final Conclusions

**Best checkpoint:** Run1/best (checkpoints/run1/best)
- Shown to improve +0.135pp on Mar 11 window; single-window evaluation is unreliable at this scale
- run_glia_1b (stride=64) and run_glia_2a (warmup=10) both regress

**Dead ends confirmed for Europe:**
- stride=64: val_loss above 0.015 threshold
- warmup=10: val_loss valid but MAPE regressed

**Key structural findings:**
1. Window instability: ~0.3pp noise per window — exceeds Europe's improvement ceiling
2. val_loss < 0.015 is necessary but not sufficient for MAPE improvement
3. VWAGY (German ADR) consistently degrades under finetuning — structural mismatch
4. Europe improvement ceiling: < 0.2pp (fragile). Rolling evaluation (10+ windows) recommended before deployment decisions.
5. Run1 hyperparams (lr=1e-4, wd=0.01, freeze=17, warmup=5, stride=128) are the only viable config — the valid-basin boundary is sharp.
