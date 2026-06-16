# Glia Run Summary — China/HK

Stop reason: approved
Turns completed: 2
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_china\finetune_expts\glia_run_20260311_151818

---

## Final Result: -6.698pp improvement vs baseline

**Deploy: `finetune_china/checkpoints/run_glia_2a/best`**

## Experiment Results

| Run | Config | Best val_loss | Overall MAPE | vs baseline |
|-----|--------|--------------|-------------|------------|
| Run1 (pre-glia) | freeze=17, stride=128, 10 tickers, 230 windows | 0.020608 | 23.789% | -4.879pp WORSE |
| run_glia_1a | 6 long-history tickers, stride=64, 385 windows | 0.035757 | (not eval'd) | — |
| **run_glia_2a** | **10 tickers, stride=32, 901 windows** | **0.012067** | **12.212%** | **-6.698pp** |

## Per-Ticker MAPE at 120d

| Ticker | Baseline | run_glia_2a | Delta |
|--------|---------|-------------|-------|
| 0388.HK (HKEX) | 4.052% | 5.794% | +1.742pp |
| 2382.HK (Sunny Optical) | 21.322% | 6.256% | **-15.066pp** |
| 1177.HK (Sino Biopharma) | 31.356% | 24.587% | **-6.769pp** |
| **Overall** | **18.910%** | **12.212%** | **-6.698pp** |

## Key Findings

1. **Regime alignment is critical:** The 4 short-history post-2018 tech tickers (9988.HK Alibaba, 3690.HK Meituan, 9999.HK NetEase, 1810.HK Xiaomi) are regime-aligned with eval tickers. Removing them caused val_loss to jump 0.020→0.036.

2. **Stride=32 was the unlock:** With only 5-8 years of data for the 4 recent tickers, stride=128 gave too few windows (~230 total). stride=32 → 901 windows → val_loss 0.012 → valid checkpoint.

3. **val_loss < 0.015 threshold holds for China:** Same threshold as US experiments. 0.012067 → genuine improvement.

4. **0388.HK structural mismatch:** HKEX (exchange operator) has macro/regulatory dynamics vs tech-sector momentum. Tech-biased training data slightly misaligns for this name (+1.742pp regression). All other tickers improved substantially.

## Open Questions
- Can 0388.HK regression be addressed by adding an exchange-operator analogue (e.g., CBOE, ICE) to training?
- Rolling evaluation (20 windows) recommended to validate robustness of the -6.698pp result.
