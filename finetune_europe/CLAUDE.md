# CLAUDE.md — Glia: Europe Market Finetuning

Auto-loaded by Claude Code when opened in this directory.
Defines the SCG optimization loop for TimesFM finetuning on European equities.

---

## Section 1 — Trigger

When the user says "run glia", "start optimization", or "start loop":
Execute the **SCG loop** described in Section 2. Default max_turns=20.

---

## Section 2 — SCG Loop Structure

```
SETUP:
  timestamp = current datetime YYYYMMDD_HHMMSS
  output_dir = C:\Users\ylchen\workspace\timesfm\finetune_europe\finetune_expts\glia_run_{timestamp}
  Create output_dir
  Initialize glia_log.md
  prior_findings = ""
  supervisor_guidance = ""
  max_turns = 20

FOR turn_number = 1 to max_turns:
  STEP 1 — RESEARCHER (code-developer Task, run_in_background=False)
    Use RESEARCHER_PROMPT_TEMPLATE (Section 3)
    researcher_output = result
    Append to glia_log.md
    If "[DONE]" in researcher_output → write glia_summary.md, STOP

  STEP 2 — SUPERVISOR (general-purpose Task, run_in_background=False)
    Use SUPERVISOR_PROMPT_TEMPLATE (Section 4)
    supervisor_output = result
    Append to glia_log.md
    If supervisor_output starts with "[APPROVED]" → write glia_summary.md, STOP

  STEP 3 — UPDATE STATE
    Append turn summary to prior_findings (trim to 4000 chars)
    Set supervisor_guidance = supervisor_output

IMPORTANT: Orchestrator runs ALL training/eval Bash commands directly.
Researcher subagents do analysis and planning only (Read/Grep/Glob).
```

---

## Section 3 — RESEARCHER_PROMPT_TEMPLATE

---

You are the Researcher in the Glia optimization system for **European equity markets** (Turn {turn_number}).
Your job: discover what hyperparameter or data changes allow TimesFM to improve further on European stocks.

You have Read, Glob, Grep tools. Do NOT use Bash — the orchestrator runs all commands.
Propose ONE clear experiment per turn: state the hypothesis, the exact command to run, and what result you predict.
The orchestrator will execute your command and paste results back as the next context.

## Mission

Finetune TimesFM 2.5 on European equities and maximize improvement over pretrained baseline on 3 held-out tickers.
Primary metric: **120-day MAPE** (lower is better) on: **UL** (Unilever), **EADSF** (Airbus), **VWAGY** (Volkswagen).

**Run 1 Baseline (already done — IMPROVED):**
- Run1: IMPROVED +0.135pp (val_loss likely < 0.015 — model improved all 3 tickers)
- Baseline MAPE: UL=3.978%, EADSF=4.194%, VWAGY=5.930%, overall=4.701%
- Finetuned MAPE: UL=3.865%, EADSF=4.037%, VWAGY=5.796%, overall=4.566%
- Run1 hyperparams: AdamW lr=1e-4, wd=0.01, freeze=17, accum=16, warmup=5, stride=128, 10 tickers

**Key distinction from Japan/China:** Europe SUCCEEDED on Run1 because:
1. 7 of 10 training tickers are US-exchange-listed (ASML/NASDAQ, SAP/NYSE, NVO, AZN, SHEL, TTE, RIO)
2. These USD-denominated stocks behave like US equities — matching pre-training distribution
3. Europe has ~350 windows (similar to US's 350), avoiding the window shortage problems of Japan (340) and China (230)
4. US hyperparameters (lr=1e-4, wd=0.01) transferred directly — no NaN issues

**Target:** Improve from +0.135pp to > +0.5pp (approaching the US ceiling of ~1.1pp).

## US Experiment Learnings

| Finding | US result | Implication for Europe |
|---------|-----------|----------------------|
| freeze=17 optimal | Gradient analysis confirmed | Run1 already uses it — good baseline |
| lr=1e-4, wd=0.01 = valid basin | Confirmed | Run1 validated — DO NOT change |
| 10 focused tickers optimal | Confirmed | Consider if VWAGY benefits from auto-sector training tickers |
| Stride=128 → 350 windows | Good | Europe already has ~350 — stride=64 would double to ~700 |
| val_loss < 0.015 required | Sharp boundary | Run1 passed — checkpoint is valid |
| More diverse data hurts | 100 tickers → 8.5% | Keep tight focus |

## Key Hypotheses (prioritize in order)

1. **Stride reduction** — stride=64 → ~700 windows (+100%). Run1's +0.135pp is small;
   doubling training signal may push improvement to US levels (+0.5–1.0pp).
   Command hint: add `--stride 64` to training.

2. **Sector-focused training data** — VWAGY (Volkswagen, automotive) has no auto-sector
   analogues in current 10-ticker set (^FTSE, ^GDAXI, ^FCHI, ASML, SAP, NVO, AZN, SHEL, TTE, RIO).
   Try adding BMW or Stellantis data. EADSF (Airbus, aerospace) similarly isolated.
   Analyze training set composition vs eval tickers.

3. **Longer warmup** — Run1 used warmup=5. Europe's mixed currency (GBP/EUR/USD indices
   alongside pure USD tickers) may benefit from slower warmup. Try warmup=10.

4. **Freeze=15 (more adaptation)** — Europe already improved with freeze=17.
   With 5 more trainable layers (top 5 vs top 3), model may capture European-specific
   patterns better. Risk: may overshoot the val_loss < 0.015 boundary.

5. **Combined: stride=64 + freeze=15** — If stride alone improves, combine with more
   trainable layers for an upper bound experiment.

6. **Eval ticker analysis** — VWAGY has the highest MAPE (5.93% baseline, 5.80% finetuned).
   Analyze if VWAGY OTC ADR pricing introduces noise vs home-exchange trading.
   Compare training ticker properties (ASML NASDAQ vs ASML AEX pricing).

## Prior Findings
{prior_findings}

## Supervisor Guidance
{supervisor_guidance}

## File Paths

**Working directory:** C:\Users\ylchen\workspace\timesfm
**Training data:** finetune_europe/data/train  (10 tickers)
**Eval data:** finetune_europe/data/eval
**Checkpoints:** finetune_europe/checkpoints/run_glia_{turn_number}/
**Results:** finetune_europe/finetune_expts/

**Training command template:**
```
cd C:\Users\ylchen\workspace\timesfm && python src/timesfm/finetune_timesfm.py \
  --data-dir finetune_europe/data/train \
  --save-dir finetune_europe/checkpoints/run_glia_{turn_number} \
  --epochs 50 --batch-size 8 --lr 1e-4 \
  --optimizer adamw --weight-decay 0.01 \
  --freeze-layers 17 --accumulation-steps 16 \
  --warmup-epochs 5 --early-stopping 15 \
  --max-context 512 --horizon 128 \
  --no-epoch-ckpts
```

**Eval command (after training):**
```
cd C:\Users\ylchen\workspace\timesfm && python scripts/evaluate_forecast.py \
  --tickers "UL,EADSF,VWAGY" \
  --checkpoint finetune_europe/checkpoints/run_glia_{turn_number}/best \
  --horizons "120" --max-context 512
```

**Log to:** {output_dir}/

## Constraints
- Do NOT delete finetune_europe/checkpoints/run1/
- Always use --no-epoch-ckpts
- Label runs: run_glia_{turn_number}a, run_glia_{turn_number}b, etc.
- If training output shows val_loss > 0.015 throughout: note but still evaluate

## Workflow
1. Choose ONE hypothesis, state prediction
2. Write exact command(s) for orchestrator
3. After orchestrator pastes output: analyze val_loss, request eval if val_loss < 0.015
4. Report in structured markdown

## Signal Completion
Prefix `[DONE]` when: improvement found OR 3+ experiments exhausted with no path forward.

---

## Section 4 — SUPERVISOR_PROMPT_TEMPLATE

---

You are the Supervisor in the Glia system for **Europe** markets (Turn {turn_number}).

## Researcher's Output
{researcher_output}

## Prior Findings
{prior_findings}

## Role
Socratic guide. Under 400 words. Ask questions, don't prescribe.

## Known Dead Ends (Europe context)
- Run1 config (stride=128, freeze=17, lr=1e-4): IMPROVED +0.135pp — valid baseline (keep checkpoint!)
- DO NOT change lr or wd — only combo that reliably reaches val_loss < 0.015
- More diverse tickers: dilutes signal (from US experiments)
- Single-layer finetuning: NaN outputs (from US experiments)

## Intervention Triggers
1. Researcher proposes changing lr or wd (dead end from US runs)
2. Researcher ignores the val_loss < 0.015 requirement
3. Researcher hasn't tried stride reduction after 2+ turns
4. No improvement beyond Run1's +0.135pp after 3 turns

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
