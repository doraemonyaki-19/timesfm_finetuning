# CONTINUATION — Sector Router Checkpoint Restoration

**Status as of 2026-05-29**

---

## Context

The disk cleanup deleted all four regional sector router checkpoints referenced in
`sector_routing_config.yaml`. The US was partially mitigated by moving exp_11, but
exp_11 has different training params from Run 4 (the original US model), so it also
needs to be re-trained.

All four regional training data directories are intact.

---

## Blocker Found

**The venv is in `timesfm/`, not `timesfm_finetuning/`.**

```
C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe   ← correct
C:\Users\ylchen\workspace\timesfm_finetuning\.venv\           ← does not exist
```

The background job launched earlier used the wrong python path and failed silently.
The `all_regions_done.log` sentinel file was never created.

---

## What Was Completed This Session

- [x] exp_11 moved from `research_group/artifacts/finetune_checkpoints/exp_11/` to
      `timesfm_finetuning/finetune_usa/checkpoints/` (model.safetensors + metadata)
- [x] `sector_routing_config.yaml` US entry updated:
      `checkpoint: C:\Users\ylchen\workspace\timesfm_finetuning\finetune_usa\checkpoints\best`
      `label: run4`
- [x] `research_group_timesfm_expt.md` path reference updated to match
- [x] Checkpoint dirs created:
      - `finetune_japan/checkpoints/run_glia_t2_r2a/`
      - `finetune_europe/checkpoints/run_glia_v2_1a/`
      - `finetune_china/checkpoints/run_glia_2a/`
      - `finetune_usa/checkpoints/` (holds exp_11 model as placeholder)

---

## What Needs To Be Done

### 1. Re-train all four regions

Run sequentially from `C:\Users\ylchen\workspace\timesfm_finetuning\` using the
correct python path. Log to `finetune_logs\`.

**Python to use:** `C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe`

#### US — Run 4 (save-dir overwrites the exp_11 placeholder)

```powershell
& "C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe" finetune_timesfm.py `
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,NEE,AMT,WMT" `
  --cache-dir "C:\Users\ylchen\workspace\research_group\data\us" `
  --save-dir finetune_usa/checkpoints `
  --model-id "C:\Users\ylchen\workspace\timesfm_finetuning\model" `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 3 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --no-epoch-ckpts
```

#### Japan — run_glia_t2_r2a (stride=64, horizon-loss-decay=1.0)

```powershell
& "C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe" finetune_timesfm.py `
  --data-dir finetune_japan/data/train `
  --save-dir finetune_japan/checkpoints/run_glia_t2_r2a `
  --model-id "C:\Users\ylchen\workspace\timesfm_finetuning\model" `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 5 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --stride 64 --horizon-loss-decay 1.0 `
  --no-epoch-ckpts
```

#### Europe — run_glia_v2_1a (stride=64)

```powershell
& "C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe" finetune_timesfm.py `
  --data-dir finetune_europe/data/train `
  --save-dir finetune_europe/checkpoints/run_glia_v2_1a `
  --model-id "C:\Users\ylchen\workspace\timesfm_finetuning\model" `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 5 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --stride 64 `
  --no-epoch-ckpts
```

#### China — run_glia_2a (stride=64, long-history tickers only)

```powershell
& "C:\Users\ylchen\workspace\timesfm\.venv\Scripts\python.exe" finetune_timesfm.py `
  --data-dir finetune_china/data/train_long `
  --save-dir finetune_china/checkpoints/run_glia_2a `
  --model-id "C:\Users\ylchen\workspace\timesfm_finetuning\model" `
  --epochs 50 --batch-size 8 --lr 1e-4 `
  --optimizer adamw --weight-decay 0.01 `
  --freeze-layers 17 --accumulation-steps 16 `
  --warmup-epochs 5 --early-stopping 15 `
  --max-context 512 --horizon 128 `
  --stride 64 `
  --no-epoch-ckpts
```

### 2. After all training completes

Verify `sector_routing_config.yaml` checkpoint paths are reachable:

```powershell
Test-Path "C:\Users\ylchen\workspace\timesfm_finetuning\finetune_usa\checkpoints\best\model.safetensors"
Test-Path "C:\Users\ylchen\workspace\timesfm_finetuning\finetune_japan\checkpoints\run_glia_t2_r2a\best\model.safetensors"
Test-Path "C:\Users\ylchen\workspace\timesfm_finetuning\finetune_europe\checkpoints\run_glia_v2_1a\best\model.safetensors"
Test-Path "C:\Users\ylchen\workspace\timesfm_finetuning\finetune_china\checkpoints\run_glia_2a\best\model.safetensors"
```

### 3. Other pending tasks (from board directives, May 29 2026)

- [ ] Draft upstream collaboration email to Alizadeh & Balakrishnan (MIT CSAIL)
      — due ~June 3, 2026. Share train-test skew finding, propose joint fix.
      Do NOT cold-pitch co-authorship.
- [ ] Log Kill Gate decision in `EVALUATION_PROTOCOL.md`
- [ ] Clean up failed background job artifacts in `finetune_logs/`

---

## sector_routing_config.yaml — Current State

| Region | Checkpoint path | Status |
|--------|----------------|--------|
| US     | `finetune_usa/checkpoints/best` | exp_11 placeholder — **needs Run 4 re-train** |
| Japan  | `finetune_japan/checkpoints/run_glia_t2_r2a/best` | **needs training** |
| Europe | `finetune_europe/checkpoints/run_glia_v2_1a/best` | **needs training** |
| China  | `finetune_china/checkpoints/run_glia_2a/best` | **needs training** |
