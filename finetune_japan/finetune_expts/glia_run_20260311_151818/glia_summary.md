# Glia Run Summary — Japan

Stop reason: approved
Turns completed: 2
Output dir: C:\Users\ylchen\workspace\timesfm\finetune_japan\finetune_expts\glia_run_20260311_151818

---

## Final Conclusion: Japan Cannot Be Meaningfully Finetuned

**Use pretrained baseline for Japan. Do not deploy any finetuned checkpoint.**

## Experiment Results

| Run | Config | Best val_loss | vs Run1 |
|-----|--------|--------------|---------|
| Run1 (pre-glia) | freeze=17, stride=128 | 0.027763 | baseline |
| run_glia_1a | stride=64 (670 windows) | 0.057212 | +106% WORSE |
| run_glia_2b | freeze=15 (83.8M trainable) | 0.035277 | +27% WORSE |

All well above the critical val_loss < 0.015 threshold. None are deployable.

## Structural Finding

The val_loss floor for Japan is ~0.027 (freeze=17, stride=128). This is structural:
- More trainable layers → worse (freeze=15: 0.035)
- More windows → far worse (stride=64: 0.057)
- The pretrained TimesFM has strong pretraining priors that actively conflict with Japan market dynamics

## Eval Tickers (pretrained baseline — best configuration)

- 8035.T (Tokyo Electron): **34.943% MAPE** — extreme outlier, likely high-beta semiconductor
- 6902.T (Denso): **2.883% MAPE** — stable auto supplier, easy to predict
- 4661.T (Oriental Land): **19.627% MAPE**
- Overall: **19.151% MAPE** at 120d

## Open Questions

1. **8035.T outlier**: Without it, Japan overall MAPE drops to ~11% — comparable to other markets. Investigate if 8035.T is genuinely unpredictable or a poor eval choice.
2. **val_loss threshold**: The 0.015 threshold is US-derived. Japan may have a different valid convergence criterion.
3. **Pretraining distribution**: TimesFM likely was pretrained on US/global data. Japanese equities may be systematically OOD.
