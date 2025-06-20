# SOTA Comparison Study

## Academic Integrity Statement

This comparison study follows strict academic integrity guidelines:

1. **Original Implementation**: All methods are implemented exactly as described in their original papers
2. **No Optimization**: No additional optimizations or modifications beyond the original work
3. **Fair Comparison**: Same datasets, same evaluation metrics, same experimental setup
4. **Reproducible**: All code includes proper citations and follows original methodologies

## Methods Compared

### 1. Brain-Diffuser
- **Paper**: Ozcelik & VanRullen (2023) - Scientific Reports
- **Implementation**: Exact replication of original methodology
- **Location**: `brain_diffuser/`

### 2. Mind-Vis
- **Paper**: Chen et al. (2023) - CVPR 2023
- **Implementation**: Exact replication of original methodology
- **Location**: `mind_vis/`

### 3. CCCV1 (Our Method) - DUAL APPROACH
- **Paper**: [Our CCCV1 Paper Reference]
- **Primary**: Optimized version with all innovations (`../models/` - current trained models)
- **Ablation**: Baseline version for ablation study (`cccv1_baseline/`)

## Folder Structure

```
sota_comparison/
├── brain_diffuser/
│   ├── src/           # Original implementation code
│   ├── configs/       # Original configuration files
│   ├── models/        # Trained models
│   └── results/       # Evaluation results
├── mind_vis/
│   ├── src/           # Original implementation code
│   ├── configs/       # Original configuration files
│   ├── models/        # Trained models
│   └── results/       # Evaluation results
├── cccv1_baseline/
│   ├── src/           # Baseline implementation
│   ├── configs/       # Baseline configurations
│   ├── models/        # Trained models
│   └── results/       # Evaluation results
├── comparison_results/
│   ├── metrics/       # Comparative metrics
│   ├── visualizations/ # Comparison plots
│   └── reports/       # Analysis reports
└── README.md          # This file
```

## Evaluation Protocol

### Primary Comparison (Main Results)
**CCCV1 Optimized vs Brain-Diffuser vs Mind-Vis**
- Each method with their best configurations
- Fair comparison of optimized approaches
- Primary results for paper

### Ablation Study (Supplementary)
**CCCV1 Baseline vs CCCV1 Optimized**
- Shows value of CCCV1 innovations
- Demonstrates contribution of optimizations
- Supplementary analysis

### Common Standards
1. **Same Datasets**: miyawaki, vangerven, crell, mindbigdata
2. **Same Metrics**: MSE, Correlation, SSIM, LPIPS
3. **Same Cross-Validation**: 10-fold CV for fair comparison
4. **Same Hardware**: CUDA-enabled GPU for consistent timing

## Citation Requirements

Each implementation must include proper citations to:
- Original paper authors
- Original code repositories (if available)
- Dataset sources
- Evaluation metric definitions

## Reproducibility Checklist

- [ ] Original paper methodology followed exactly
- [ ] No unauthorized optimizations added
- [ ] Same experimental setup across all methods
- [ ] Proper statistical analysis
- [ ] Clear documentation of any limitations
- [ ] Results validated against original papers (where possible)
