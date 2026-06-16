# TimesFM MAPE Optimization Strategy

## Current Performance Analysis
From the tier labels data, I observe:
- US NVDA (120d): 4.207% → 3.627% MAPE (-0.58pp improvement)  
- US UNH (120d): 24.007% → 18.98% MAPE (-5.03pp improvement)
- US GS (120d): 6.199% → 6.2% MAPE (no improvement)

The current approach achieves good results on some tickers but fails on others. Key issues:
1. High variance in per-ticker performance 
2. Some tickers (like GS at 120d) show no improvement
3. Hyperparameters may not be optimally tuned for MAPE minimization

## Optimization Strategies

### 1. Advanced Loss Functions
- **MAPE-aware loss**: Direct MAPE loss instead of MSE in log space
- **Asymmetric loss**: Penalize over-prediction vs under-prediction differently
- **Quantile loss**: Train with pinball loss focusing on median prediction
- **Huber loss**: Robust to outliers which can distort MAPE

### 2. Hyperparameter Optimization
- **Learning rate scheduling**: Optimize warmup, decay patterns
- **Layer-wise learning rates**: Different rates for different transformer layers
- **Batch size and accumulation**: Find optimal effective batch size
- **Context length**: Optimize max_context vs min_context ratios

### 3. Data Augmentation & Regularization
- **Stronger regularization**: Increase dropout, weight decay
- **Data augmentation**: Price scaling, temporal jittering
- **Early stopping**: Optimize patience and validation strategy
- **Cross-validation**: K-fold validation for better model selection

### 4. Architecture Modifications
- **Selective layer unfreezing**: Find optimal layers to train
- **Attention masking**: Improve context utilization
- **Multi-horizon training**: Joint training on multiple horizons
- **Ensemble methods**: Combine multiple model variants

### 5. Advanced Training Techniques
- **Curriculum learning**: Start with shorter horizons, progress to longer
- **Knowledge distillation**: Use stronger models as teachers
- **Meta-learning**: Learn to adapt quickly to new tickers
- **Adversarial training**: Improve robustness

## Implementation Plan

1. **Baseline measurement**: Establish current MAPE on all eval tickers
2. **Loss function experiments**: Test MAPE-direct training
3. **Hyperparameter sweep**: Systematic optimization
4. **Advanced techniques**: Implement most promising methods
5. **Ensemble combination**: Combine best individual models
6. **Final validation**: Cross-region validation of improvements