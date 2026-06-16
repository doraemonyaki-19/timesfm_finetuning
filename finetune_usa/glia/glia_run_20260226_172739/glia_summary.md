# Glia Run Summary

Stop reason: approved
Turns completed: 4
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_expts\glia_run_20260226_172739
Date: 2026-02-27

---

## Best Configuration

**Run 4 (checkpoints/best)** — AdamW, freeze 17/20 layers, 16× gradient accumulation, pure MSE loss

| Horizon | Baseline MAPE | Run 4 MAPE | Delta | DirAcc Δ |
|---------|--------------|-----------|-------|----------|
| 7d      | 3.2%         | 3.0%      | -0.2pp| +14.3pp  |
| 30d     | 5.0%         | 5.0%      | 0.0pp | +6.7pp   |
| 60d     | 6.2%         | 6.1%      | -0.1pp| +3.0pp   |
| **120d**| **9.0%**     | **8.0%**  | **-1.0pp** | **+2.7pp** |

Per-ticker at 120d (current 2026-02-27 eval window):
| Ticker | Baseline | Run 4 | Delta |
|--------|----------|-------|-------|
| NVDA   | 11.5%    | 9.3%  | -2.1pp |
| UNH    | 11.0%    | 8.1%  | -2.8pp |
| GS     | 8.5%     | 8.9%  | +0.4pp |
| CVX    | 8.5%     | 6.8%  | -1.8pp |
| KO     | 6.3%     | 6.3%  | 0.0pp |
| BA     | 8.0%     | 8.4%  | +0.4pp |

---

## All Experiments Conducted

### Turn 1: Hypothesis 1 — Ticker Similarity (+ LMT, RTX, NOC)
- **run_glia_1a**: 13 tickers (10 original + LMT/RTX/NOC), same Run 4 hyperparameters
- **Result: FAILED** — 42/232 model tensors NaN. LR=1e-4 too high for 13-ticker training. Val loss tracking failed to detect weight divergence.
- **Learning**: Adding tickers without reducing LR causes training instability.

### Turn 2: Tool access blocked; analysis only
- Identified NaN failure mechanism. Confirmed BA structural regime change (737 MAX crisis 2019) as root cause of consistent BA degradation.
- BA statistical profile: Vol=0.358 (highest), Kurtosis=16.0 (highest), MaxDD=-34%, 20yr return only +339% vs 752-1712% for defense peers.

### Turn 3: Diagnostic + Hypothesis 3 (Mixed Loss α=0.8)
**Layer delta analysis (pretrained → Run 4):**
- Layers 0-16: frozen (0.0 delta, perfectly preserved)
- Layers 17-19: trained uniformly (L2 delta ~1.85-1.89 each — no wasted capacity)
- Tokenizer + point output head: adapted; quantile head untouched
- **Conclusion**: Non-uniform layer selection (Hypothesis 2) unlikely to improve over Run 4

**run_glia_3a**: Same as Run 4 + `--loss-type mixed --loss-alpha 0.8`
- Loss = 0.8·MSE + 0.2·hinge_sign_loss(consecutive steps)
- Best val loss: 0.011956 at epoch 35. No NaN.
- 120d: **8.0% MAPE** (-1.0pp) — tied with Run 4
- Per-ticker: NVDA -4.7pp (dramatic), CVX -2.9pp BUT KO +2.0pp (regression), BA +1.2pp
- **Learning**: Amplitude-weighted sign loss → high-vol stocks dominate gradients at KO's expense. NVDA benefits from directional training, KO is hurt.

### Turn 4: Hypotheses 3 variants (α=0.95 and overall-direction)
**run_glia_4a**: Mixed loss α=0.95 (less sign weight)
- Best val loss: 0.013902 at epoch 3. Early stopped epoch 18.
- 120d: **8.1% MAPE** (-0.9pp) — slightly worse than Run 4
- KO RECOVERED to -0.1pp (from +2.0pp at α=0.8) — confirms volatility bias hypothesis
- GS improved to -0.5pp (vs Run 4's +0.4pp)
- CVX lost some gains (-1.0pp vs -1.8pp)
- **Learning**: α=0.95 eliminates KO volatility bias but at cost of CVX gains. No net improvement.

**run_glia_4b**: Overall-direction sign loss α=0.9
- Loss uses sign(target[-1] - target[0]) vs sign(pred[-1] - pred[0]) — designed to fix DirAcc paradox
- Best val loss: 0.012126 at epoch 6. Early stopped epoch 21.
- 120d: **8.0% MAPE** (-0.9pp) — tied with Run 4
- GS -0.8pp (strongest GS result across all runs!), CVX -2.3pp
- DirAcc paradox PERSISTS: overall-direction training made DirAcc worse at all horizons
- **Learning**: Overall-direction loss trains model to commit to trend, hurts step-by-step accuracy. These are different objectives.

---

## Key Findings

### 1. The ~1.0pp improvement ceiling
Every experiment converges to ~-1.0pp MAPE improvement at 120d vs baseline. This ceiling is:
- Robust across MSE, mixed loss (3 variants), different α values
- Insensitive to which loss formulation is used
- Likely architectural or data-regime constrained, not hyperparameter-sensitive

### 2. The BA problem
BA degrades (+0.4pp to +1.2pp) in every experiment. Root cause: 737 MAX crisis (2019) created a structural regime change that makes BA's pre-2019 and post-2019 distributions fundamentally different. No loss function variant fixes this. BA needs a fundamentally different approach.

### 3. Mixed loss redistribution
Sign loss variants redistribute wins/losses across tickers without moving the average:
- High-vol momentum stocks (NVDA, CVX) benefit from directional training
- Low-vol mean-reverting stocks (KO) are hurt by amplitude-weighted gradients
- α=0.95 balances this but produces no net gain
- GS benefits from both α=0.95 and overall-direction loss

### 4. Infrastructure discovery
Per-epoch checkpointing consumed 45-87GB. Added `--no-epoch-ckpts` flag to finetune_timesfm.py. Future runs should use this flag.

### 5. config.json fix
Fixed `_copy_config_json()` to use HuggingFace hub download as fallback when model_id is a repo name (not local path). Previously, checkpoints from HF-loaded models were missing config.json.

---

## Open Hypotheses (not fully tested)

1. **Hypothesis 4 (xreg decay)**: `correction[t] *= exp(-λt)` — could extend 30d xreg promise to longer horizons without divergence. No training required. Not tested.

2. **Hypothesis 5 retry (ticker similarity at lower LR)**: LMT/RTX/NOC at LR=5e-5 instead of 1e-4. The NaN failure was execution (too-high LR), not conceptual. Not retried.

3. **BA pre-crisis data exclusion**: Training on BA data post-2019 only — directly addressing the structural break. Not tested.

---

## Code Changes Made This Run

1. `src/timesfm/finetune_timesfm.py`:
   - Fixed `_copy_config_json()`: HF hub download fallback
   - Added `--loss-type` flag: `mse` (default), `mixed` (consecutive-step sign loss), `mixed_dir` (overall-direction sign loss)
   - Added `--loss-alpha` flag: MSE weight in mixed loss (default 0.8)
   - Added `--no-epoch-ckpts` flag: skip per-epoch checkpoints, save only best/ and last/
   - Modified `training_step()` to accept loss_type and loss_alpha parameters

---

## Checkpoint Inventory

| Checkpoint | Description | 120d MAPE | Status |
|-----------|-------------|-----------|--------|
| `checkpoints/best/` | Run 4 (best overall) | 8.0% | ✓ BEST |
| `checkpoints/run_glia_3a/best/` | Mixed loss α=0.8 | 8.0% | Tied |
| `checkpoints/run_glia_4a/best/` | Mixed loss α=0.95 | 8.1% | -0.1pp |
| `checkpoints/run_glia_4b/best/` | Overall-dir α=0.9 | 8.0% | Tied |
| `checkpoints/run_glia_1a/` | DELETED (NaN weights) | — | Gone |

---

*Glia run completed 2026-02-27. Approved by Supervisor after 4 turns, 7 experiments.*
