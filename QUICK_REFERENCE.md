# Quick Reference - Model Checkpoints
## Instant Commands for Fair Comparison

---

## 🚀 **INSTANT FAIR COMPARISON**
```bash
# Run complete fair comparison (30 seconds)
python scripts/final_fair_comparison.py
```

## 🔍 **VERIFY CHECKPOINTS**
```bash
# Check all models are ready (10 seconds)
python scripts/verify_model_checkpoints.py
```

---

## 📊 **CHECKPOINT STATUS: 100% READY**

| Method | Status | Location |
|--------|--------|----------|
| **CortexFlow** | ✅ 4/4 | `models/{dataset}_cv_best.pth` |
| **Brain-Diffuser** | ✅ 4/4 | `models/{dataset}_brain_diffuser_simplified.pkl` |
| **Mind-Vis** | ✅ 4/4 | `sota_comparison/mind_vis/models/{dataset}_mind_vis_best.pth` |

---

## 🏆 **FAIR COMPARISON RESULTS**

### **Winner by Dataset:**
- **Miyawaki**: 🥇 CortexFlow (72% better than Brain-Diffuser)
- **Vangerven**: 🥇 CortexFlow (29% better than Brain-Diffuser)  
- **Crell**: 🥇 Mind-Vis (tie with CortexFlow, not significant)
- **MindBigData**: 🥇 Mind-Vis (tie with CortexFlow, not significant)

### **Overall Performance:**
- **CortexFlow**: 6/12 wins (50%)
- **Mind-Vis**: 4/12 wins (33%)
- **Brain-Diffuser**: 0/12 wins (0%)

---

## 💾 **LOAD INDIVIDUAL MODELS**

### **CortexFlow:**
```python
import torch
model = torch.load('models/miyawaki_cv_best.pth', map_location='cpu')
```

### **Brain-Diffuser:**
```python
import pickle
with open('models/miyawaki_brain_diffuser_simplified.pkl', 'rb') as f:
    model = pickle.load(f)
```

### **Mind-Vis:**
```python
import torch
model = torch.load('sota_comparison/mind_vis/models/miyawaki_mind_vis_best.pth', map_location='cpu')
```

---

## 📁 **KEY FILES**

### **Results:**
- `results/final_fair_comparison_*/final_fair_comparison_report_*.json`
- `results/brain_diffuser_cv_results.json`
- `results/mind_vis_cv_results.json`

### **Documentation:**
- `MODEL_CHECKPOINT_GUIDE.md` (Complete guide)
- `QUICK_REFERENCE.md` (This file)

---

## ⚡ **NO RETRAINING NEEDED!**
**All 12 models ready for instant use** 🎉
