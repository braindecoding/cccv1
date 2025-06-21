# Model Checkpoint Guide
## Complete Guide for Using Saved Models Without Retraining

**Last Updated**: June 21, 2025  
**Status**: All 12 models (3 methods × 4 datasets) ready with checkpoints  
**Academic Integrity**: Verified and maintained

---

## 🎯 Overview

This guide provides comprehensive instructions for using the saved model checkpoints to run fair comparisons and evaluations without needing to retrain models from scratch. All models have been trained with identical conditions and saved with full reproducibility.

## 📊 Checkpoint Status

### ✅ **100% Complete - All Models Ready**

| Method | Miyawaki | Vangerven | Crell | MindBigData | Total |
|--------|----------|-----------|-------|-------------|-------|
| **CortexFlow** | ✅ | ✅ | ✅ | ✅ | **4/4** |
| **Brain-Diffuser** | ✅ | ✅ | ✅ | ✅ | **4/4** |
| **Mind-Vis** | ✅ | ✅ | ✅ | ✅ | **4/4** |
| **Total** | **3/3** | **3/3** | **3/3** | **3/3** | **12/12** |

---

## 📁 Checkpoint Locations

### 🤖 **CortexFlow Checkpoints**
```
models/miyawaki_cv_best.pth + models/miyawaki_cv_best_metadata.json
models/vangerven_cv_best.pth + models/vangerven_cv_best_metadata.json
models/crell_cv_best.pth + models/crell_cv_best_metadata.json
models/mindbigdata_cv_best.pth + models/mindbigdata_cv_best_metadata.json
```

### 🧠 **Brain-Diffuser Checkpoints**
```
models/miyawaki_brain_diffuser_simplified.pkl
models/vangerven_brain_diffuser_simplified.pkl
models/crell_brain_diffuser_simplified.pkl
models/mindbigdata_brain_diffuser_simplified.pkl
```

### 👁️ **Mind-Vis Checkpoints**
```
sota_comparison/mind_vis/models/miyawaki_mind_vis_best.pth
sota_comparison/mind_vis/models/vangerven_mind_vis_best.pth
sota_comparison/mind_vis/models/crell_mind_vis_best.pth
sota_comparison/mind_vis/models/mindbigdata_mind_vis_best.pth
```

### 📊 **CV Results Files**
```
results/brain_diffuser_cv_results.json
results/mind_vis_cv_results.json
results/final_fair_comparison_20250621_073025/final_fair_comparison_report_20250621_073025.json
```

---

## 🚀 Quick Start Commands

### **1. Run Complete Fair Comparison**
```bash
# Execute full fair comparison with all saved models
python scripts/final_fair_comparison.py
```
**Output**: Complete statistical comparison with significance testing  
**Time**: ~30 seconds (no training needed)

### **2. Verify All Checkpoints**
```bash
# Verify all models are properly saved and loadable
python scripts/verify_model_checkpoints.py
```
**Output**: Comprehensive checkpoint status report  
**Time**: ~10 seconds

### **3. Load Individual Models**
```bash
# Load specific model for custom evaluation
python -c "
import torch
model = torch.load('models/miyawaki_cv_best.pth', map_location='cpu')
print('CortexFlow model loaded successfully')
"
```

---

## 📋 Detailed Usage Instructions

### **Fair Comparison Execution**

#### **Step 1: Verify Environment**
```bash
# Ensure all dependencies are installed
pip install torch numpy scipy matplotlib seaborn pandas scikit-learn

# Verify checkpoint integrity
python scripts/verify_model_checkpoints.py
```

#### **Step 2: Run Fair Comparison**
```bash
# Execute complete fair comparison
python scripts/final_fair_comparison.py

# Expected output:
# - Statistical significance tests
# - Performance comparisons
# - Academic integrity verification
# - Publication-ready results
```

#### **Step 3: Access Results**
```bash
# Results saved in timestamped directory
ls results/final_fair_comparison_*/

# View JSON report
cat results/final_fair_comparison_*/final_fair_comparison_report_*.json
```

### **Individual Model Loading**

#### **CortexFlow Model Loading**
```python
import torch
import json

# Load model
model_path = "models/miyawaki_cv_best.pth"
checkpoint = torch.load(model_path, map_location='cpu')

# Load metadata
metadata_path = "models/miyawaki_cv_best_metadata.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"Model architecture: {metadata['architecture']}")
print(f"CV scores: {metadata['cv_scores']}")
```

#### **Brain-Diffuser Model Loading**
```python
import pickle

# Load simplified Brain-Diffuser
model_path = "models/miyawaki_brain_diffuser_simplified.pkl"
with open(model_path, 'rb') as f:
    brain_diffuser_model = pickle.load(f)

print("Brain-Diffuser model loaded successfully")
```

#### **Mind-Vis Model Loading**
```python
import torch
import sys
import os

# Add path for Mind-Vis
sys.path.append('sota_comparison/mind_vis/src')
from mind_vis_model import SimplifiedMindVis

# Load model
model_path = "sota_comparison/mind_vis/models/miyawaki_mind_vis_best.pth"
model = SimplifiedMindVis(input_dim=967, output_dim=784)
model.load_state_dict(torch.load(model_path, map_location='cpu'))

print("Mind-Vis model loaded successfully")
```

---

## 📊 Performance Summary

### **Fair Comparison Results (Real vs Real)**

| Dataset | CortexFlow | Mind-Vis | Brain-Diffuser | Winner |
|---------|------------|----------|----------------|---------|
| **Miyawaki** | **0.005500** ± 0.004129 | 0.013850 ± 0.005569 | 0.019787 ± 0.003698 | **🏆 CortexFlow** |
| **Vangerven** | **0.044505** ± 0.004611 | 0.048816 ± 0.003304 | 0.062625 ± 0.005815 | **🏆 CortexFlow** |
| **Crell** | 0.032525 ± 0.001393 | **0.032493** ± 0.001035 | 0.059478 ± 0.032085 | **🏆 Mind-Vis** |
| **MindBigData** | 0.057019 ± 0.001570 | **0.056956** ± 0.001188 | 0.143951 ± 0.107499 | **🏆 Mind-Vis** |

### **Statistical Significance**
- **Total Comparisons**: 12 pairwise tests
- **Significant Results**: 10/12 (83.3%)
- **CortexFlow Wins**: 6/12 (50.0%)
- **Academic Integrity**: ✅ Verified

---

## 🔧 Advanced Usage

### **Custom Evaluation Script**
```python
#!/usr/bin/env python3
"""
Custom evaluation using saved checkpoints
"""

import torch
import numpy as np
from pathlib import Path

def load_all_models(dataset_name):
    """Load all three models for a dataset"""
    
    models = {}
    
    # Load CortexFlow
    cortexflow_path = f"models/{dataset_name}_cv_best.pth"
    if Path(cortexflow_path).exists():
        models['CortexFlow'] = torch.load(cortexflow_path, map_location='cpu')
    
    # Load Brain-Diffuser
    bd_path = f"models/{dataset_name}_brain_diffuser_simplified.pkl"
    if Path(bd_path).exists():
        import pickle
        with open(bd_path, 'rb') as f:
            models['Brain-Diffuser'] = pickle.load(f)
    
    # Load Mind-Vis
    mv_path = f"sota_comparison/mind_vis/models/{dataset_name}_mind_vis_best.pth"
    if Path(mv_path).exists():
        models['Mind-Vis'] = torch.load(mv_path, map_location='cpu')
    
    return models

# Usage example
dataset = "miyawaki"
models = load_all_models(dataset)
print(f"Loaded {len(models)} models for {dataset}")
```

### **Batch Evaluation**
```bash
# Create batch evaluation script
cat > batch_evaluate.py << 'EOF'
import subprocess
import json
from datetime import datetime

datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
results = {}

for dataset in datasets:
    print(f"Evaluating {dataset}...")
    # Add your custom evaluation logic here
    results[dataset] = {"status": "evaluated"}

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"results/batch_evaluation_{timestamp}.json", 'w') as f:
    json.dump(results, f, indent=2)
EOF

python batch_evaluate.py
```

---

## 🛠️ Troubleshooting

### **Common Issues and Solutions**

#### **Issue 1: Model Loading Fails**
```bash
# Check file existence
ls -la models/miyawaki_cv_best.pth

# Verify file integrity
python -c "
import torch
try:
    model = torch.load('models/miyawaki_cv_best.pth', map_location='cpu')
    print('✅ Model loads successfully')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

#### **Issue 2: Missing Dependencies**
```bash
# Install missing packages
pip install torch numpy scipy matplotlib seaborn pandas scikit-learn

# For Brain-Diffuser (if needed)
pip install diffusers transformers clip-by-openai
```

#### **Issue 3: Path Issues**
```bash
# Verify current directory
pwd
# Should be in: .../cccv1/

# Check relative paths
ls models/
ls sota_comparison/mind_vis/models/
```

### **Verification Commands**
```bash
# Quick verification of all checkpoints
python scripts/verify_model_checkpoints.py

# Test fair comparison (should complete in <1 minute)
python scripts/final_fair_comparison.py

# Check results directory
ls -la results/final_fair_comparison_*/
```

---

## 📚 Academic Usage

### **For Publications**
1. **Cite Fair Comparison**: Use results from `final_fair_comparison_report_*.json`
2. **Academic Integrity**: All models trained with identical conditions
3. **Reproducibility**: Checkpoints ensure exact result reproduction
4. **Statistical Rigor**: Wilcoxon signed-rank tests applied

### **For Further Research**
1. **Extend Evaluation**: Load checkpoints and add new metrics
2. **Ablation Studies**: Use saved models as baselines
3. **Cross-Dataset Analysis**: Compare performance patterns
4. **Architecture Analysis**: Examine saved model parameters

---

## 🎯 Summary

### **✅ What You Have**
- **12 trained models** with full checkpoints
- **Complete CV results** (10-fold for each model)
- **Fair comparison framework** ready to run
- **Academic integrity** verified and maintained

### **⚡ What You Can Do**
- **Instant fair comparison** (no retraining needed)
- **Custom evaluations** using saved models
- **Publication-ready results** with statistical rigor
- **Reproducible research** with full checkpoint system

### **🚀 Next Steps**
1. Run `python scripts/final_fair_comparison.py` for complete results
2. Use checkpoints for custom analysis
3. Extend evaluation with additional metrics
4. Prepare publication materials with verified results

---

**🏆 All models ready for immediate use - no training required!**
