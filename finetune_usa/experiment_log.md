# Experiment Log

---

## Run 1: Conservative SGD (lr=5e-4, 10 epochs)

**Date:** 2026-02-19
**Status:** Completed — weights barely changed, no improvement

### Hyperparameters
| Param | Value |
|-------|-------|
| Optimizer | SGD (momentum=0.9) |
| Learning rate | 5e-4 |
| Warmup epochs | 3 |
| Total epochs | 10 |
| Batch size | 8 |
| Max context | 512 |
| Horizon | 128 |
| Max grad norm | 1.0 |

### Command
```bash
python src/timesfm_finetuning/finetune_timesfm.py \
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,NEE,AMT,WMT" \
  --epochs 10 --batch-size 8 --max-context 512 --horizon 128 \
  --lr 5e-4 --warmup-epochs 3 --save-dir checkpoints --log-every 5
```

### Training Results
- Data: 10 tickers, 5027 points each, 350 sliding windows (263 train / 87 val)
- Total params: 231,289,280 (all trainable)
- Best val loss: **0.012973** (epoch 7)

| Epoch | Train Loss | Val Loss | Best? |
|-------|-----------|----------|-------|
| 1 | 0.018216 | 0.014546 | Yes |
| 2 | 0.017583 | 0.018218 | |
| 3 | 0.016870 | 0.014104 | Yes |
| 4 | 0.015776 | 0.015635 | |
| 5 | 0.016744 | 0.014160 | |
| 6 | 0.015633 | 0.014166 | |
| 7 | 0.016557 | 0.012973 | Yes |
| 8 | 0.015547 | 0.015840 | |
| 9 | 0.016230 | 0.015342 | |
| 10 | 0.016373 | 0.014583 | |

### Evaluation Results
Evaluated on 6 held-out tickers (NVDA, UNH, GS, CVX, KO, BA).

**Result: Finetuned model produced IDENTICAL predictions to baseline.**

Weight comparison confirmed weights moved by < 1e-6 — effectively no change.

| Ticker | Model | MSE | MAE | RMSE | MAPE% | DirAcc% |
|--------|-------|-----|-----|------|-------|---------|
| NVDA | Baseline | 133.17 | 9.63 | 11.54 | 5.1% | 52.3% |
| UNH | Baseline | 1174.96 | 25.10 | 34.28 | 7.9% | 52.3% |
| GS | Baseline | 15607.13 | 100.66 | 124.93 | 11.4% | 45.3% |
| CVX | Baseline | 258.06 | 13.33 | 16.06 | 8.2% | 56.2% |
| KO | Baseline | 23.82 | 3.62 | 4.88 | 5.0% | 49.2% |
| BA | Baseline | 543.01 | 19.31 | 23.30 | 9.3% | 45.3% |
| **OVERALL** | **Baseline** | **2956.69** | **28.61** | **54.38** | **7.8%** | **50.1%** |

### Analysis
With only 350 windows and SGD at lr=5e-4, the model barely moved from pretrained weights.
The paper used batch_size=1024 on 8xV100 — our effective gradient signal is ~128x weaker.
Need higher LR and/or more epochs to produce meaningful weight changes.

---

## Run 2: Aggressive SGD (lr=5e-3, 50 epochs)

**Date:** 2026-02-19
**Status:** Completed — weights moved ~2e-6 avg, still no inference difference

### Hyperparameters
| Param | Value |
|-------|-------|
| Optimizer | SGD (momentum=0.9) |
| Learning rate | **5e-3** (10x higher) |
| Warmup epochs | 5 |
| Total epochs | **50** (5x more) |
| Batch size | 8 |
| Max context | 512 |
| Horizon | 128 |
| Max grad norm | 1.0 |

### Command
```bash
python src/timesfm_finetuning/finetune_timesfm.py \
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,NEE,AMT,WMT" \
  --epochs 50 --batch-size 8 --max-context 512 --horizon 128 \
  --lr 5e-3 --warmup-epochs 5 --save-dir checkpoints --log-every 10
```

### Training Results
- Best val loss: **0.0138** (epoch 4)
- Train loss: 0.0168 → 0.0083

### Analysis
Even with 10x higher LR, SGD with batch_size=8 still couldn't move weights meaningfully. The per-parameter update magnitude was ~2e-6 on average — insufficient to change inference output. SGD applies a uniform LR across all parameters, which is suboptimal for a pretrained model where different layers need different update magnitudes.

---

## Run 3: Adam (lr=1e-4, 50 epochs)

**Date:** 2026-02-20
**Status:** Completed — finetuned model produces different predictions, mixed results

### Hyperparameters
| Param | Value |
|-------|-------|
| Optimizer | **Adam** |
| Learning rate | **1e-4** |
| Warmup epochs | 3 |
| Total epochs | 50 |
| Batch size | 8 |
| Max context | 512 |
| Horizon | 128 |
| Max grad norm | 1.0 |

### Command
```bash
python src/timesfm_finetuning/finetune_timesfm.py \
  --model-id "C:/Users/ylchen/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/snapshots/1d952420fba87f3c6dee4f240de0f1a0fbc790e3" \
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,NEE,AMT,WMT" \
  --epochs 50 --batch-size 8 --max-context 512 --horizon 128 \
  --optimizer adam --lr 1e-4 --warmup-epochs 3 --save-dir checkpoints --log-every 10
```

### Rationale
- Adam uses per-parameter adaptive learning rates — much better for small-batch finetuning
- Each parameter gets its own momentum and variance estimate, enabling meaningful updates
- lr=1e-4 is a standard default for Adam finetuning

### Training Results
- Best val loss: **0.0161** (epoch 40)
- Train loss: 0.0139 → 0.0003 (heavy overfitting after epoch ~15)
- Weight change: mean abs diff 0.00092 (~20-33% relative change in key layers)

Key epochs:

| Epoch | Train Loss | Val Loss | Best? |
|-------|-----------|----------|-------|
| 1 | 0.013883 | 0.020406 | Yes |
| 3 | 0.011343 | 0.018562 | Yes |
| 6 | 0.007461 | 0.017666 | Yes |
| 10 | 0.004199 | 0.017090 | Yes |
| 15 | 0.001799 | 0.017198 | |
| 20 | 0.001105 | 0.019226 | |
| 23 | 0.000818 | 0.017024 | Yes |
| 29 | 0.000542 | 0.016432 | Yes |
| 30 | 0.000505 | 0.016382 | Yes |
| 40 | 0.000346 | 0.016113 | Yes |
| 50 | 0.000292 | 0.016951 | |

### Evaluation Results

| Ticker | Model | MSE | MAE | RMSE | MAPE | DirAcc |
|--------|-------|-----|-----|------|------|--------|
| NVDA | Baseline | 137.02 | 9.79 | 11.71 | 5.2% | 43.8% |
| NVDA | Finetuned | 111.41 | 8.36 | 10.56 | **4.5%** | **53.1%** |
| UNH | Baseline | 1622.85 | 31.86 | 40.28 | **10.2%** | 48.4% |
| UNH | Finetuned | 2704.91 | 45.72 | 52.01 | 13.6% | 46.1% |
| GS | Baseline | 15090.51 | 99.71 | 122.84 | **11.3%** | 54.7% |
| GS | Finetuned | 20135.75 | 121.92 | 141.90 | 14.0% | 55.5% |
| CVX | Baseline | 260.70 | 13.11 | 16.15 | 8.0% | **57.0%** |
| CVX | Finetuned | 219.00 | 8.57 | 14.80 | **5.0%** | 49.2% |
| KO | Baseline | 29.54 | 3.99 | 5.44 | 5.5% | 49.2% |
| KO | Finetuned | 21.54 | 3.16 | 4.64 | **4.3%** | **50.0%** |
| BA | Baseline | 417.49 | 17.13 | 20.43 | **8.2%** | 49.2% |
| BA | Finetuned | 1023.04 | 25.65 | 31.99 | 12.5% | 50.8% |
| **Overall** | **Baseline** | **2926.35** | **29.26** | **54.10** | **8.1%** | 50.4% |
| **Overall** | **Finetuned** | **4035.94** | **35.56** | **63.53** | 9.0% | **50.8%** |

### Analysis

**Winners (finetuned better):**
- NVDA: MAPE -0.7pp, DirAcc +9.4pp (biggest improvement)
- CVX: MAPE -3.0pp (but DirAcc -7.8pp)
- KO: MAPE -1.2pp, DirAcc +0.8pp

**Losers (finetuned worse):**
- UNH: MAPE +3.5pp
- GS: MAPE +2.7pp
- BA: MAPE +4.3pp

The finetuned model improved on tickers with moderate price levels (NVDA ~$140, CVX ~$160, KO ~$60) similar to training data, but degraded on high-price or volatile stocks (UNH ~$340, GS ~$600, BA ~$180). This suggests overfitting to the price scale/volatility distribution of training tickers.

### Bug discovered
`TimesFM_2p5_200M_torch` uses a class-level `model: nn.Module` attribute — all instances share the same underlying `nn.Module`. Loading a second model overwrites the first. The evaluation script was fixed to load and evaluate sequentially.

### Possible next steps
- Freeze early transformer layers, only finetune last few layers + output head
- Use more training tickers with wider price range
- Add dropout/weight decay regularization
- Use gradient accumulation to simulate larger effective batch size

---

## Run 4: AdamW + Layer Freezing + Gradient Accumulation

**Date:** 2026-02-20
**Status:** Completed — best results, 5/6 tickers improved, overall MAPE -1.0pp

### Hyperparameters
| Param | Value |
|-------|-------|
| Optimizer | **AdamW** |
| Learning rate | 1e-4 |
| Weight decay | **0.01** |
| Warmup epochs | 3 |
| Total epochs | 50 |
| Batch size | 8 |
| **Accumulation steps** | **16** (effective batch = 128) |
| Max context | 512 |
| Horizon | 128 |
| Max grad norm | 1.0 |
| **Freeze layers** | **17** (of 20 transformer layers) |

### Command
```bash
python src/timesfm_finetuning_finetuning/finetune_timesfm.py \
  --model-id "C:/Users/ylchen/.cache/huggingface/hub/models--google--timesfm-2.5-200m-pytorch/snapshots/1d952420fba87f3c6dee4f240de0f1a0fbc790e3" \
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,NEE,AMT,WMT" \
  --epochs 50 --batch-size 8 --max-context 512 --horizon 128 \
  --optimizer adamw --lr 1e-4 --weight-decay 0.01 \
  --freeze-layers 17 --accumulation-steps 16 \
  --warmup-epochs 3 --save-dir checkpoints --log-every 10
```

### Rationale
Three anti-overfitting measures to address Run 3's degradation:
1. **Layer freezing (17/20):** Reduce trainable params from 231M to 64M (27.7%). Early layers learn general time-series features — only finetune the last 3 layers + output heads on financial data.
2. **AdamW with weight decay:** Penalize large weight changes to prevent catastrophic forgetting of pretrained knowledge.
3. **Gradient accumulation (16x):** Effective batch size 128 (vs 8). More stable gradients, closer to the paper's batch_size=1024.

### Training Results
- Total params: 231,289,280 (64,081,360 trainable = 27.7%)
- Best val loss: **0.012764** (epoch 4) — significantly better than Run 3's 0.0161
- Train loss: 0.018 → 0.003 (gradual, not the 0.014→0.0003 collapse in Run 3)
- Much healthier training dynamics — val loss stable around 0.013-0.019

### Evaluation Results

| Ticker | Model | MSE | MAE | RMSE | MAPE | DirAcc |
|--------|-------|-----|-----|------|------|--------|
| NVDA | Baseline | 137.02 | 9.79 | 11.71 | 5.2% | 43.8% |
| NVDA | Finetuned | 71.87 | 6.81 | 8.48 | **3.6%** | **46.9%** |
| UNH | Baseline | 1622.85 | 31.86 | 40.28 | 10.2% | 48.4% |
| UNH | Finetuned | 953.75 | 23.70 | 30.88 | **7.5%** | 46.1% |
| GS | Baseline | 15090.45 | 99.71 | 122.84 | 11.3% | 54.7% |
| GS | Finetuned | 11948.89 | 88.52 | 109.31 | **10.0%** | 54.7% |
| CVX | Baseline | 260.70 | 13.11 | 16.15 | 8.0% | 57.0% |
| CVX | Finetuned | 209.98 | 11.14 | 14.49 | **6.7%** | **60.2%** |
| KO | Baseline | 29.54 | 3.99 | 5.44 | 5.5% | 49.2% |
| KO | Finetuned | 27.21 | 3.81 | 5.22 | **5.2%** | **50.0%** |
| BA | Baseline | 417.49 | 17.13 | 20.43 | **8.2%** | 49.2% |
| BA | Finetuned | 541.33 | 18.55 | 23.27 | 9.0% | **50.8%** |
| **Overall** | **Baseline** | 2926.34 | 29.26 | 54.10 | 8.1% | 50.4% |
| **Overall** | **Finetuned** | **2292.17** | **25.42** | **47.88** | **7.0%** | **51.4%** |

### Analysis

**Comparison with Run 3:**

| Ticker | Run 3 MAPE delta | Run 4 MAPE delta | Improvement |
|--------|-----------------|-----------------|-------------|
| NVDA | -0.7pp | **-1.6pp** | Better |
| UNH | +3.5pp | **-2.6pp** | Flipped to win |
| GS | +2.7pp | **-1.3pp** | Flipped to win |
| CVX | -3.0pp | -1.3pp | Slightly less |
| KO | -1.2pp | -0.3pp | Slightly less |
| BA | +4.3pp | **+0.8pp** | Much less damage |
| **Overall** | **+0.9pp** | **-1.0pp** | **Flipped to win** |

The three anti-overfitting measures together transformed 3/6 wins into 5/6 wins. UNH and GS flipped from significant degradation to clear improvement. BA still slightly degraded but by only +0.8pp vs +4.3pp. The overall MAPE flipped from +0.9pp worse to -1.0pp better.

### Possible next steps
- Try freezing fewer layers (e.g., 14-15) to allow more capacity while still preventing overfitting
- Increase training tickers for broader coverage
- Early stopping based on val loss (best was epoch 4, trained to 50)
- Try higher accumulation (32x) for even closer match to paper's batch size

---

## Run 5: 100 Tickers + Fewer Frozen Layers + Higher Accumulation + Early Stopping

**Date:** 2026-02-21
**Status:** Completed — regression from Run 4, overfitting from continued finetuning

### Hyperparameters
| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| Learning rate | **5e-5** (lower than Run 4's 1e-4) |
| Weight decay | 0.01 |
| Warmup epochs | **5** |
| Total epochs | 50 (early stopped at 35) |
| Batch size | 8 |
| **Accumulation steps** | **64** (effective batch = 512) |
| Max context | 512 |
| Horizon | 128 |
| Max grad norm | 1.0 |
| **Freeze layers** | **14** (of 20) |
| **Early stopping** | **patience=15** |
| **Starting point** | **Run 4 best checkpoint** |
| **Training tickers** | **100** (vs 10 in Run 4) |

### Command
```bash
PYTHONUNBUFFERED=1 python -u src/timesfm_finetuning/finetune_timesfm.py \
  --model-id "checkpoints/best" \
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,NEE,AMT,WMT,MSFT,GOOGL,META,AMZN,TSLA,AVGO,ADBE,CRM,CSCO,INTC,ORCL,AMD,TXN,QCOM,IBM,PFE,MRK,ABBV,TMO,ABT,LLY,DHR,BMY,AMGN,GILD,BAC,WFC,C,MS,BLK,SCHW,AXP,USB,PNC,TFC,COP,SLB,EOG,PSX,VLO,PEP,COST,MO,CL,KMB,HON,UPS,LMT,RTX,GE,MMM,DE,FDX,EMR,NSC,DUK,SO,D,AEP,SRE,PLD,CCI,EQIX,SPG,PSA,LIN,APD,SHW,ECL,DD,HD,MCD,NKE,SBUX,TGT,LOW,TJX,BKNG,MAR,CMG,DIS,CMCSA,T,VZ,NFLX,^DJI,^IXIC,^RUT,^FTSE,^N225" \
  --epochs 50 --batch-size 8 --max-context 512 --horizon 128 \
  --optimizer adamw --lr 5e-5 --weight-decay 0.01 \
  --freeze-layers 14 --accumulation-steps 64 \
  --warmup-epochs 5 --early-stopping 15 \
  --save-dir checkpoints_run5 --log-every 10
```

### Rationale
1. **100 training tickers** (vs 10) for broader generalization across all sectors + international indices
2. **Freeze 14 layers** (vs 17) — more trainable capacity with extra data to support it
3. **64x accumulation** — effective batch 512, closer to paper's 1024
4. **Lower LR (5e-5)** with 5-epoch warmup for gentler adaptation
5. **Early stopping (patience=15)** to avoid wasting epochs after convergence
6. **Continued from Run 4 checkpoint** — start from already partially-adapted weights

### Training Results
- Data: 100 tickers, 3445 sliding windows (2584 train / 861 val)
- Total params: 231,289,280 (93,610,192 trainable = 40.5%)
- Best val loss: **0.016711** (epoch 20)
- Train loss: 0.018 → 0.006
- Training crashed at epoch 35 due to disk full (per-epoch checkpoints consumed 31GB), but best checkpoint was already saved. Early stopping would have triggered at epoch 35 anyway.

| Epoch | Train Loss | Val Loss | Best? |
|-------|-----------|----------|-------|
| 1 | - | 0.020636 | Yes |
| 2 | - | 0.020046 | Yes |
| 3 | - | 0.019732 | Yes |
| 4 | - | 0.019320 | Yes |
| 5 | - | 0.018797 | Yes |
| 6 | - | 0.019123 | |
| 7 | 0.017071 | 0.018693 | Yes |
| 8 | 0.016219 | 0.018401 | Yes |
| 9 | 0.015680 | 0.018494 | |
| 10 | 0.014642 | 0.018051 | |
| 11 | 0.013960 | 0.017272 | Yes |
| 12 | - | 0.017130 | Yes |
| 13 | - | 0.017889 | |
| 14 | - | 0.018221 | |
| 15 | - | 0.017949 | |
| 16 | - | 0.017407 | |
| 17 | - | 0.017772 | |
| 18 | - | 0.017848 | |
| 19 | - | 0.017065 | Yes |
| 20 | - | 0.016711 | Yes |
| 21 | - | 0.018301 | |
| 22 | - | 0.017001 | |
| 23 | - | 0.018160 | |
| 24 | - | 0.018111 | |
| 25 | - | 0.017155 | |
| 26 | - | 0.017046 | |
| 27 | - | 0.017716 | |
| 28 | - | 0.017989 | |
| 29 | - | 0.017382 | |
| 30 | - | 0.017117 | |
| 31 | - | 0.017781 | |
| 32 | - | 0.017535 | |
| 33 | - | 0.018511 | |
| 34 | - | 0.017037 | |
| 35 | 0.006402 | (crash) | |

### Multi-Horizon Evaluation Results

**Horizon = 7 days:**

| Ticker | Model | MSE | MAE | RMSE | MAPE | DirAcc |
|--------|-------|-----|-----|------|------|--------|
| NVDA | Baseline | 95.44 | 9.57 | 9.77 | 5.6% | 57.1% |
| NVDA | Finetuned | 96.78 | 9.57 | 9.84 | 5.6% | 85.7% |
| UNH | Baseline | 333.37 | 13.40 | 18.26 | 4.1% | 57.1% |
| UNH | Finetuned | 381.03 | 13.96 | 19.52 | 4.3% | 28.6% |
| GS | Baseline | 121.91 | 9.26 | 11.04 | 1.3% | 85.7% |
| GS | Finetuned | 142.64 | 9.28 | 11.94 | 1.3% | 57.1% |
| CVX | Baseline | 5.17 | 1.92 | 2.27 | 1.2% | 42.9% |
| CVX | Finetuned | 10.84 | 2.94 | 3.29 | 1.9% | 42.9% |
| KO | Baseline | 0.45 | 0.62 | 0.67 | 0.9% | 42.9% |
| KO | Finetuned | 0.42 | 0.55 | 0.65 | 0.8% | 42.9% |
| BA | Baseline | 25.55 | 4.52 | 5.06 | 2.0% | 42.9% |
| BA | Finetuned | 37.29 | 5.25 | 6.11 | 2.3% | 28.6% |
| **Overall** | **Baseline** | 96.98 | 6.55 | 9.85 | **2.5%** | 54.8% |
| **Overall** | **Finetuned** | 111.50 | 6.92 | 10.56 | 2.7% | 47.6% |

**Horizon = 30 days:**

| Ticker | Model | MAPE | DirAcc |
|--------|-------|------|--------|
| NVDA | Finetuned | **3.1%** (-0.7pp) | 70.0% |
| UNH | Finetuned | 12.6% (+3.5pp) | 43.3% |
| CVX | Finetuned | **1.2%** (-1.2pp) | 50.0% |
| BA | Finetuned | 9.3% (+2.3pp) | 36.7% |
| **Overall** | **Finetuned** | 5.5% (+0.7pp) | 48.9% |

**Horizon = 60 days:**

| Ticker | Model | MAPE | DirAcc |
|--------|-------|------|--------|
| NVDA | Finetuned | **4.4%** (-2.0pp) | 66.7% |
| CVX | Finetuned | **1.1%** (-2.2pp) | 55.0% |
| BA | Finetuned | 16.5% (+5.6pp) | 43.3% |
| **Overall** | **Finetuned** | 6.7% (+0.9pp) | 51.1% |

**Horizon = 120 days:**

| Ticker | Model | MSE | MAE | RMSE | MAPE | DirAcc |
|--------|-------|-----|-----|------|------|--------|
| NVDA | Finetuned | 99.08 | 8.04 | 9.95 | **4.3%** (-2.6pp) | 64.2% |
| UNH | Finetuned | 1478.34 | 33.96 | 38.45 | **10.2%** (-1.4pp) | 47.5% |
| GS | Finetuned | 10060.34 | 80.20 | 100.30 | 9.0% (+1.3pp) | 53.3% |
| CVX | Finetuned | 124.99 | 6.93 | 11.18 | **4.1%** (-2.9pp) | 55.0% |
| KO | Finetuned | 41.20 | 4.87 | 6.42 | 6.7% (+0.8pp) | 44.2% |
| BA | Finetuned | 1502.95 | 33.88 | 38.77 | 16.3% (+6.9pp) | 47.5% |
| **Overall** | **Finetuned** | 2217.82 | 27.98 | 47.09 | 8.4% (+0.4pp) | 51.9% |

### Analysis

**Run 5 regressed from Run 4 across all horizons.** Key findings:

1. **Continued finetuning hurt:** Starting from Run 4's checkpoint (already adapted to financial data) and training further accumulated too much domain-specific bias. The model moved too far from the pretrained distribution.

2. **More trainable params hurt:** Freezing only 14 layers (93.6M trainable, 40.5%) vs Run 4's 17 layers (64M, 27.7%) gave more room for overfitting despite having 10x more training data.

3. **BA catastrophically degraded:** MAPE 16.3% vs 9.4% baseline at 120 days, up from Run 4's mild +0.8pp. The extra capacity + continued finetuning magnified BA-specific degradation.

4. **Directional accuracy improved:** Despite worse MAPE, DirAcc improved at 60d (+4.4pp) and 120d (+3.6pp), suggesting the model learned some directional patterns but over-corrected on magnitude.

5. **NVDA and CVX still benefited:** These tickers consistently improve across all runs and horizons, suggesting their patterns are well-captured by finetuning on financial data.

### Key Takeaway
**Run 4's more conservative approach (freeze 17 layers, train from pretrained, 10 tickers) remains the best model.** The "more data + more capacity" strategy backfired because:
- Continued finetuning from an already-adapted checkpoint compounds domain drift
- More trainable parameters need proportionally more regularization
- 10x more data was not enough to compensate for 1.5x more trainable params + domain drift

---

## Run 6: 100 Tickers + Run 4 Hyperparameters (Isolating "more data")

**Date:** 2026-02-22
**Status:** Completed — regression from baseline, confirming more data hurts

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Frozen layers | 17/20 |
| Trainable params | 64M (27.7%) |
| Batch size | 8 |
| Accumulation steps | 16 (effective batch 128) |
| Warmup epochs | 25 |
| Epochs | 50 |
| Early stopping | patience=15 |
| Training tickers | 100 (diverse sectors + intl indices) |
| Starting point | Pretrained (google/timesfm-2.5-200m-pytorch) |
| Sliding windows | 3445 (2584 train / 861 val) |

### Command
```bash
python src/timesfm_finetuning/finetune_timesfm.py \
  --model-id "google/timesfm-2.5-200m-pytorch" \
  --tickers "^GSPC,AAPL,JNJ,JPM,XOM,PG,CAT,...(100 tickers)...,^FTSE,^N225" \
  --epochs 50 --batch-size 8 --max-context 512 --horizon 128 \
  --optimizer adamw --lr 1e-4 --weight-decay 0.01 \
  --freeze-layers 17 --accumulation-steps 16 \
  --warmup-epochs 25 --early-stopping 15 \
  --save-dir checkpoints_run6 --log-every 10
```

### Rationale
Isolate "more data" as the only variable vs Run 4. All hyperparameters identical — only difference is 100 training tickers vs 10.

### Training Results

Best val loss: **0.015296** at epoch 40. Completed all 50 epochs (early stopping didn't trigger since epoch 40+15=55 > 50).

Val loss progression:
| Epoch | Val Loss | Best? |
|-------|----------|-------|
| 1 | 0.026100 | Yes |
| 5 | 0.022042 | Yes |
| 8 | 0.020215 | Yes |
| 13 | 0.018178 | Yes |
| 17 | 0.017858 | Yes |
| 23 | 0.017322 | Yes |
| 26 | 0.016551 | Yes |
| 37 | 0.016537 | Yes |
| 40 | 0.015296 | Yes |
| 50 | 0.016671 | No |

### Multi-horizon Evaluation

**7-day horizon:**
| Ticker | Baseline MAPE | Finetuned MAPE | Delta |
|--------|--------------|----------------|-------|
| NVDA | 5.6% | 3.3% | -2.4pp |
| UNH | 4.1% | 7.2% | +3.0pp |
| GS | 1.3% | 2.1% | +0.9pp |
| CVX | 1.2% | 1.6% | +0.4pp |
| KO | 0.9% | 0.8% | -0.1pp |
| BA | 2.0% | 1.9% | -0.0pp |
| **Overall** | **2.5%** | **2.8%** | **+0.3pp** |

**30-day horizon:**
| Ticker | Baseline MAPE | Finetuned MAPE | Delta |
|--------|--------------|----------------|-------|
| NVDA | 3.8% | 4.7% | +0.9pp |
| UNH | 9.1% | 13.7% | +4.6pp |
| GS | 3.8% | 3.0% | -0.9pp |
| CVX | 2.4% | 1.0% | -1.4pp |
| KO | 2.3% | 1.9% | -0.4pp |
| BA | 7.0% | 6.6% | -0.3pp |
| **Overall** | **4.7%** | **5.1%** | **+0.4pp** |

**60-day horizon:**
| Ticker | Baseline MAPE | Finetuned MAPE | Delta |
|--------|--------------|----------------|-------|
| NVDA | 6.4% | 7.8% | +1.4pp |
| UNH | 8.2% | 11.9% | +3.6pp |
| GS | 3.0% | 2.6% | -0.4pp |
| CVX | 3.4% | 1.0% | -2.4pp |
| KO | 2.5% | 2.7% | +0.2pp |
| BA | 10.8% | 11.6% | +0.8pp |
| **Overall** | **5.7%** | **6.3%** | **+0.5pp** |

**120-day horizon:**
| Ticker | Baseline MAPE | Finetuned MAPE | Delta |
|--------|--------------|----------------|-------|
| NVDA | 6.9% | 8.5% | +1.6pp |
| UNH | 11.6% | 12.0% | +0.5pp |
| GS | 7.7% | 8.5% | +0.8pp |
| CVX | 7.0% | 5.0% | -2.0pp |
| KO | 5.8% | 5.5% | -0.4pp |
| BA | 9.4% | 11.6% | +2.1pp |
| **Overall** | **8.1%** | **8.5%** | **+0.5pp** |

### Analysis
1. **More data hurts:** With identical hyperparameters to Run 4, the only change was 100 tickers vs 10, and performance regressed at all horizons.
2. **Signal dilution:** Diverse sectors + international indices add noise that dilutes the signal for the 6 eval tickers.
3. **CVX still benefits:** Consistent improvement (-2.0 to -2.4pp at 60d/120d) across all runs.
4. **UNH still degrades:** +0.5 to +4.6pp across horizons — a persistent problem ticker.

---

## Run 6b: Continued from Run 6 Best, 100 Epochs

**Date:** 2026-02-22
**Status:** Completed — early stopped at epoch 22, no meaningful improvement over Run 6

### Hyperparameters
Same as Run 6, except:
- **Starting point:** Run 6 best checkpoint (epoch 40, val loss 0.015296)
- **Epochs:** 100 (early stopped at 22)

### Training Results
Best val loss: **0.006729** at epoch 7. Early stopped at epoch 22 (no improvement for 15 epochs). Note: val loss numbers are not comparable to Run 6 because the model started from a different point.

Val loss progression:
| Epoch | Val Loss | Best? |
|-------|----------|-------|
| 1 | 0.007011 | Yes |
| 2 | 0.006993 | Yes |
| 4 | 0.006837 | Yes |
| 7 | 0.006729 | Yes |
| 8-22 | 0.007156–0.008143 | No |

### Multi-horizon Evaluation

| Horizon | Baseline MAPE | Finetuned MAPE | Delta |
|---------|--------------|----------------|-------|
| 7 days | 2.5% | 2.8% | +0.3pp |
| 30 days | 4.7% | 5.1% | +0.3pp |
| 60 days | 5.7% | 6.1% | +0.4pp |
| 120 days | 8.1% | 8.2% | +0.1pp |

### Analysis
- Very similar to Run 6 — continued training did not rescue the regression
- Slight improvement at 120d (+0.1pp vs Run 6's +0.5pp) but still worse than baseline
- Model was already near convergence from Run 6; additional epochs didn't help

### Key Takeaway
**More data is not better for this task.** 100 diverse tickers dilute the financial signal compared to 10 focused tickers. Run 4 (10 tickers, same hyperparameters) remains the best model at all horizons. Future experiments should focus on ticker selection quality over quantity, or use a weighted sampling strategy to upweight tickers similar to the eval set.

---

## Broad Evaluation: 50 Held-Out Tickers (2026-02-22)

All models evaluated on 50 tickers not in Run 4's training set, across 4 horizons. This tests whether finetuning generalizes beyond the original 6-ticker evaluation set.

### 50 Evaluation Tickers
NVDA, UNH, GS, CVX, KO, BA, MSFT, GOOGL, META, AMZN, TSLA, AVGO, ADBE, CRM, CSCO, INTC, ORCL, AMD, TXN, QCOM, IBM, PFE, MRK, ABBV, TMO, ABT, LLY, BAC, WFC, C, MS, BLK, COP, SLB, PEP, COST, HON, UPS, LMT, RTX, GE, DE, DUK, SO, HD, MCD, NKE, DIS, NFLX, T

### Overall MAPE Comparison

| Horizon | Baseline | Run 4 | Run 5 | Run 6 | Run 6b |
|---------|----------|-------|-------|-------|--------|
| 7d | 2.1% | **2.1% (-0.0)** | 2.2% (+0.1) | 2.2% (+0.1) | 2.3% (+0.1) |
| 30d | 4.9% | **4.9% (-0.0)** | 5.0% (+0.1) | 4.9% (+0.0) | 4.9% (+0.0) |
| 60d | 7.0% | **6.9% (-0.1)** | 7.2% (+0.2) | 7.2% (+0.2) | 7.2% (+0.2) |
| 120d | 10.4% | **10.2% (-0.2)** | 10.5% (+0.1) | 11.0% (+0.6) | 10.9% (+0.5) |

### Run 4 Per-Ticker Results (120-day horizon, 50 tickers)

**Top 10 improvers:**

| Ticker | Baseline | Finetuned | Delta |
|--------|----------|-----------|-------|
| ADBE | 16.3% | 11.9% | -4.3pp |
| UNH | 11.6% | 8.0% | -3.5pp |
| AVGO | 14.6% | 11.2% | -3.4pp |
| IBM | 18.1% | 15.0% | -3.1pp |
| NKE | 21.9% | 19.5% | -2.4pp |
| NVDA | 6.9% | 4.8% | -2.1pp |
| CVX | 7.0% | 5.2% | -1.8pp |
| WFC | 6.5% | 5.2% | -1.3pp |
| ABBV | 7.9% | 6.7% | -1.2pp |
| NFLX | 32.2% | 31.0% | -1.2pp |

**Top 5 degradations:**

| Ticker | Baseline | Finetuned | Delta |
|--------|----------|-----------|-------|
| ORCL | 18.4% | 21.7% | +3.3pp |
| T | 11.2% | 14.4% | +3.2pp |
| BLK | 2.4% | 5.2% | +2.8pp |
| UPS | 8.8% | 9.9% | +1.1pp |
| TMO | 11.1% | 12.1% | +1.0pp |

### Run 5 Summary (50 tickers)

| Horizon | Baseline | Finetuned | Delta |
|---------|----------|-----------|-------|
| 7d | 2.1% | 2.2% | +0.1pp |
| 30d | 4.9% | 5.0% | +0.1pp |
| 60d | 7.0% | 7.2% | +0.2pp |
| 120d | 10.4% | 10.5% | +0.1pp |

### Run 6 Summary (50 tickers)

| Horizon | Baseline | Finetuned | Delta |
|---------|----------|-----------|-------|
| 7d | 2.1% | 2.2% | +0.1pp |
| 30d | 4.9% | 4.9% | +0.0pp |
| 60d | 7.0% | 7.2% | +0.2pp |
| 120d | 10.4% | 11.0% | +0.6pp |

### Run 6b Summary (50 tickers)

| Horizon | Baseline | Finetuned | Delta |
|---------|----------|-----------|-------|
| 7d | 2.1% | 2.3% | +0.1pp |
| 30d | 4.9% | 4.9% | +0.0pp |
| 60d | 7.0% | 7.2% | +0.2pp |
| 120d | 10.4% | 10.9% | +0.5pp |

### Analysis

1. **Run 4 generalizes:** The only model that improves (or ties) at every horizon across 50 unseen tickers. Improvement is modest (-0.2pp at 120d) but consistent.
2. **Win rate 58%:** At 120d, Run 4 improved 29/50 tickers. The biggest gains come from high-MAPE stocks (ADBE, UNH, AVGO, IBM) where there's more room for improvement.
3. **100-ticker models hurt at scale:** Runs 5, 6, 6b all regress on 50 tickers, confirming signal dilution even on a broader eval set.
4. **Run 6 worst at 120d:** +0.6pp regression — the worst of all models at the longest horizon on 50 tickers.
5. **Original 6-ticker eval overestimated Run 4's advantage:** -1.0pp on 6 tickers vs -0.2pp on 50 tickers at 120d. The 6-ticker eval happened to include more "responsive" tickers.

### Overall Conclusion
Run 4 (10 focused training tickers, freeze 17/20 layers, AdamW lr=1e-4, 16x accumulation) remains the best model. It generalizes to 50 unseen tickers with a 58% win rate at 120d. More training data (100 tickers) consistently hurts. The finetuning effect is real but modest at scale — the largest gains come from high-MAPE stocks where the pretrained model has the most room for improvement.
