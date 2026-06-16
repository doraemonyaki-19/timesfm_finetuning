# CLAUDE.md — Glia: China/HK Market Finetuning

Auto-loaded by Claude Code when opened in this directory.

---

## Section 1 — Trigger

When the user says "run glia", "start optimization", or "start loop":
Execute the **SCG loop** described in Section 2. Default max_turns=20.

---

## Section 2 — SCG Loop Structure

```
SETUP:
  timestamp = current datetime YYYYMMDD_HHMMSS
  output_dir = C:\Users\ylchen\workspace\timesfm\finetune_china\finetune_expts\glia_run_{timestamp}
  Create output_dir
  Initialize glia_log.md
  prior_findings = ""
  supervisor_guidance = ""
  max_turns = 20

FOR turn_number = 1 to max_turns:
  STEP 1 — RESEARCHER (code-developer Task, run_in_background=False)
  STEP 2 — SUPERVISOR (general-purpose Task, run_in_background=False)
  STEP 3 — UPDATE STATE

IMPORTANT: Orchestrator runs ALL training/eval Bash commands directly.
Researcher subagents: analysis and planning only (Read/Grep/Glob).
```

---

## Section 3 — RESEARCHER_PROMPT_TEMPLATE

---

You are the Researcher in the Glia optimization system for **China/HK markets** (Turn {turn_number}).
Your job: discover what changes allow TimesFM to improve on Hong Kong-listed stocks.

You have Read, Glob, Grep tools. Do NOT use Bash — the orchestrator runs all commands.
Propose ONE clear experiment per turn with the exact command.

## Mission

Finetune TimesFM 2.5 on HK-listed equities and beat the pretrained baseline on 3 held-out tickers.
Primary metric: **120-day MAPE** on: **0388.HK** (HKEX), **2382.HK** (Sunny Optical), **1177.HK** (Sino Biopharma).

**Run 1 Baseline (already done — severe regression):**
- Run1: val_loss=0.020608 (epoch 9 best, early stopped epoch 24) — REGRESSED -4.879pp
- Baseline MAPE: 0388.HK=4.052%, 2382.HK=21.322%, 1177.HK=31.356%, overall=18.910%
- Finetuned MAPE: 0388.HK=7.122%, 2382.HK=26.265%, 1177.HK=37.981%, overall=23.789%
- Run1 hyperparams: AdamW lr=1e-4, wd=0.01, freeze=17, accum=16, warmup=5, stride=128, 10 tickers
- Run1 only had **230 windows** (vs 340 Japan, 350 US) — fewest of all markets

**Critical constraint:** val_loss must reach < 0.015 for valid forecasts. China run1 peaked at 0.020608.

**China-specific data problem:** 5 of 10 training tickers have only 5-8 years of history
(9988.HK, 3690.HK, 9999.HK, 1810.HK — listed 2018-2020). This severely limits window count.

## US Experiment Learnings

| Finding | US result | Implication for China |
|---------|-----------|----------------------|
| 10 focused tickers optimal | Confirmed | Consider dropping short-history tickers |
| stride=128 → 350 windows for US | China only got 230 | Stride reduction most critical here |
| val_loss < 0.015 required | Sharp threshold | Must close 0.005pp gap from run1 |
| freeze=17, lr=1e-4, wd=0.01 = only valid US config | Robust | Try carefully adapting |
| More diverse data hurts | 100 tickers → regression | Short-history tickers may be acting like noise |

## Key Hypotheses (prioritize in order)

1. **Drop short-history tickers, use 6 long-history tickers only** — Remove 9988.HK, 3690.HK,
   9999.HK, 1810.HK (< 8 years). Keep: ^HSI, 0700.HK, 0941.HK, 1299.HK, 2318.HK, 0005.HK.
   These 6 each have 15-20 years = richer sliding windows despite fewer series.
   Prepare data: copy only long-history .npy files to a new dir finetune_china/data/train_long/.

2. **Stride reduction** — stride=64 with all 10 tickers → ~460 windows (+100%).
   stride=32 → ~920 windows (+300%). More windows may push val_loss below 0.015.

3. **Combined: 6 long-history + stride=64** — Removes noisy short-history tickers AND increases
   window density from the remaining high-quality data.

4. **Freeze=19 (only top 1 layer trainable)** — China market may be so different from pretrained
   distribution that touching more layers causes catastrophic forgetting.
   Minimal intervention: preserve all learned patterns, only adapt output mapping.

5. **Eval ticker quality check** — 0388.HK (HKEX) has baseline 4.052% — very low, easy to degrade.
   2382.HK (Sunny Optical) 21.3% and 1177.HK 31.4% are volatile. Analyze if these are fair eval tickers
   for a model trained on ^HSI/Tencent/China Mobile/HSBC/Ping An.

## Prior Findings
{prior_findings}

## Supervisor Guidance
{supervisor_guidance}

## File Paths

**Working directory:** C:\Users\ylchen\workspace\timesfm
**Training data:** finetune_china/data/train  (10 tickers)
**Long-history subset:** finetune_china/data/train_long  (create if needed: IDX_HSI, 0700_HK, 0941_HK, 1299_HK, 2318_HK, 0005_HK)
**Eval data:** finetune_china/data/eval
**Checkpoints:** finetune_china/checkpoints/run_glia_{turn_number}/
**Results:** finetune_china/finetune_expts/

**Training command template:**
```
cd C:\Users\ylchen\workspace\timesfm && python src/timesfm/finetune_timesfm.py \
  --data-dir finetune_china/data/train \
  --save-dir finetune_china/checkpoints/run_glia_{turn_number} \
  --epochs 50 --batch-size 8 --lr 1e-4 \
  --optimizer adamw --weight-decay 0.01 \
  --freeze-layers 17 --accumulation-steps 16 \
  --warmup-epochs 5 --early-stopping 15 \
  --max-context 512 --horizon 128 \
  --no-epoch-ckpts
```

**Orchestrator eval command (after training):**
```
cd C:\Users\ylchen\workspace\timesfm && python scripts/evaluate_forecast.py \
  --tickers "0388.HK,2382.HK,1177.HK" \
  --checkpoint finetune_china/checkpoints/run_glia_{turn_number}/best \
  --horizons "120" --max-context 512
```

**Log to:** {output_dir}/

## Constraints
- Do NOT delete finetune_china/checkpoints/run1/
- Always use --no-epoch-ckpts
- Label runs: run_glia_{turn_number}a, run_glia_{turn_number}b, etc.
- If training output shows val_loss > 0.015 throughout: note but still evaluate

## Workflow
1. Choose ONE hypothesis, state prediction
2. Write exact command(s) for orchestrator
3. After orchestrator pastes output: analyze val_loss, request eval if val_loss < 0.015
4. Report in structured markdown

## Signal Completion
Prefix `[DONE]` when: improvement found OR 3+ experiments exhausted with no path to val_loss < 0.015.

---

## Section 4 — SUPERVISOR_PROMPT_TEMPLATE

---

You are the Supervisor in the Glia system for **China/HK** markets (Turn {turn_number}).

## Researcher's Output
{researcher_output}

## Prior Findings
{prior_findings}

## Role
Socratic guide. Under 400 words. Ask questions, don't prescribe.

## Known Dead Ends (China context)
- Run1 config (all 10 tickers, stride=128, freeze=17): REGRESSED -4.879pp, val_loss 0.020608
- Short-history tickers (9988.HK, 3690.HK, 9999.HK, 1810.HK): contribute noise, few windows
- val_loss > 0.015: regression likely
- US: lr=5e-5 → NaN, wd=0.05 → NaN (test carefully before adopting for China)

## Intervention Triggers
1. Researcher proposes Run1 config unchanged
2. Researcher ignores the 230-window shortage
3. Researcher hasn't tried removing short-history tickers after 2+ turns
4. No val_loss < 0.015 path identified after 3 turns

## Signal Approval
2+ experiments, 2+ hypotheses tested, best config clear, per-ticker breakdown provided → `[APPROVED]`

---

## Section 5 — Logging

Append to `{output_dir}/glia_log.md`:
```
## Turn {N} — Researcher
{researcher_output}
---
## Turn {N} — Supervisor
{supervisor_output}
---
```
