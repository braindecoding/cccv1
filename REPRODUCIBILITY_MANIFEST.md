# Reproducibility Manifest

**Study**: CortexFlow-CLIP-CNN V1: CLIP-Guided Neural Decoding Framework  
**Version**: 1.0  
**Date**: 2025-06-20  
**Author**: [Researcher Name]  
**Institution**: [Institution Name]

## 📋 **OVERVIEW**

This document provides complete information necessary to reproduce all results presented in this study. It follows best practices for computational reproducibility in academic research.

## 🖥️ **COMPUTATIONAL ENVIRONMENT**

### **Hardware Specifications**
- **GPU**: NVIDIA GeForce RTX 3060
- **GPU Memory**: 12GB GDDR6
- **CPU**: [To be specified]
- **RAM**: [To be specified]
- **Storage**: SSD recommended for data loading performance

### **Operating System**
- **OS**: Windows 11
- **Architecture**: x64
- **WSL**: Compatible (if using Linux subsystem)

### **Software Dependencies**

#### **Core Dependencies**
```
Python >= 3.8
PyTorch == 2.7.1+cu128
CUDA == 12.8
NumPy >= 1.21.0
SciPy >= 1.7.0
scikit-learn >= 1.0.0
Matplotlib >= 3.5.0
```

#### **Exact Package Versions**
```bash
# Core ML packages
torch==2.7.1+cu128
torchvision==0.18.1+cu128
torchaudio==2.2.1+cu128

# Scientific computing
numpy==1.24.3
scipy==1.10.1
scikit-learn==1.3.0

# Data handling
pandas==2.0.3
h5py==3.9.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Statistics
statsmodels==0.14.0

# Utilities
tqdm==4.65.0
pathlib2==2.3.7
```

#### **Installation Commands**
```bash
# Create virtual environment
python -m venv cccv1_env
source cccv1_env/bin/activate  # Linux/Mac
# or
cccv1_env\Scripts\activate  # Windows

# Install PyTorch with CUDA support
pip install torch==2.7.1+cu128 torchvision==0.18.1+cu128 torchaudio==2.2.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install numpy==1.24.3 scipy==1.10.1 scikit-learn==1.3.0
pip install matplotlib==3.7.1 seaborn==0.12.2 pandas==2.0.3
pip install statsmodels==0.14.0 tqdm==4.65.0 h5py==3.9.0
```

## 🗂️ **PROJECT STRUCTURE**

```
cccv1/
├── README.md                           # Project overview
├── REPRODUCIBILITY_MANIFEST.md        # This file
├── methodology_registry.json          # Pre-registered methodology
├── requirements.txt                    # Python dependencies
├── 
├── src/                               # Source code
│   ├── models/
│   │   └── cortexflow_clip_cnn_v1.py  # Main model architecture
│   ├── data/
│   │   └── secure_loader.py           # Leak-proof data loading
│   ├── methodology/
│   │   └── preregistration.py         # Methodology pre-registration
│   ├── statistics/
│   │   └── power_analysis.py          # Statistical power analysis
│   └── validation/
│       └── nested_cv.py               # Nested cross-validation
│
├── data/                              # Data directory
│   └── processed/                     # Processed datasets
│       ├── miyawaki_structured_28x28.mat
│       ├── digit69_28x28.mat
│       ├── mindbigdata.mat
│       └── crell.mat
│
├── scripts/                           # Execution scripts
│   ├── train_cccv1.py                # Training script
│   ├── validate_cccv1.py             # Validation script
│   └── enhanced_validation.py        # Enhanced validation
│
├── configs/                           # Configuration files
│   └── optimal_configs.json          # Pre-optimized configurations
│
├── docs/                             # Documentation
│   ├── DATA_PROVENANCE.md            # Data documentation
│   └── architecture.md               # Architecture details
│
└── results/                          # Results directory
    ├── validation_*/                 # Validation results
    └── enhanced_validation_*/        # Enhanced validation results
```

## 🔄 **REPRODUCTION STEPS**

### **Step 1: Environment Setup**
```bash
# Clone repository
git clone [repository_url]
cd cccv1

# Create and activate virtual environment
python -m venv cccv1_env
source cccv1_env/bin/activate  # Linux/Mac
# or
cccv1_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Data Preparation**
```bash
# Verify data integrity
python src/data/secure_loader.py

# Expected output: All datasets loaded successfully with integrity verification
```

### **Step 3: Methodology Verification**
```bash
# Verify pre-registered methodology
python src/methodology/preregistration.py

# Expected output: Methodology integrity verified
```

### **Step 4: Main Experiments**

#### **Single Training (Supplementary)**
```bash
python scripts/train_cccv1.py --dataset all --mode single
```

#### **10-Fold Cross-Validation (Primary)**
```bash
python scripts/validate_cccv1.py --dataset all --folds 10 --statistical_test
```

#### **Enhanced Validation (Supplementary)**
```bash
python scripts/enhanced_validation.py
```

### **Step 5: Statistical Analysis**
```bash
# Power analysis
python src/statistics/power_analysis.py

# Expected output: Comprehensive power analysis report
```

## 🎯 **EXPECTED RESULTS**

### **Primary Results (10-Fold CV)**
| Dataset | CCCV1 Score | Champion Score | Status | p-value |
|---------|-------------|----------------|--------|---------|
| Miyawaki | 0.005500 ± 0.004130 | 0.009845 | 🏆 Win 44.13% | < 0.05 |
| Vangerven | 0.046832 ± 0.004344 | 0.045659 | Gap +2.57% | > 0.05 |
| MindBigData | 0.056971 ± 0.001519 | 0.057348 | 🏆 Win 0.66% | > 0.05 |
| Crell | 0.032527 ± 0.001404 | 0.032525 | Gap +0.01% | > 0.05 |

### **Success Metrics**
- **Primary Success Rate**: 2/4 datasets (50%)
- **Statistical Significance**: 1/4 datasets (Miyawaki)
- **Effect Sizes**: Large for Miyawaki, small for others

## 🔧 **RANDOM SEEDS**

### **Reproducibility Seeds**
```python
# Primary seeds
MAIN_SEED = 42
CV_SEEDS = [42, 43, 44]  # For multiple runs
TORCH_SEED = 42
NUMPY_SEED = 42
```

### **Seed Setting Code**
```python
import torch
import numpy as np
import random

def set_reproducibility_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## 📊 **DATA REQUIREMENTS**

### **Dataset Files**
- `miyawaki_structured_28x28.mat` (119 samples)
- `digit69_28x28.mat` (100 samples)  
- `mindbigdata.mat` (1200 samples)
- `crell.mat` (640 samples)

### **Data Format**
Each .mat file must contain:
- `fmriTrn`: Training fMRI data [N_train, input_dim]
- `stimTrn`: Training visual stimuli [N_train, 784]
- `fmriTest`: Test fMRI data [N_test, input_dim]
- `stimTest`: Test visual stimuli [N_test, 784]

### **Data Integrity Checks**
```python
# Automated verification
python data/loader.py

# Manual verification
from src.data.secure_loader import SecureDataLoader
loader = SecureDataLoader()
loader.load_dataset_secure('miyawaki', 'cpu')
```

## ⚙️ **CONFIGURATION MANAGEMENT**

### **Pre-optimized Configurations**
All hyperparameters are pre-specified in `configs/optimal_configs.json`:

```json
{
  "miyawaki": {
    "architecture": {"dropout_encoder": 0.06, "dropout_decoder": 0.02},
    "training": {"lr": 0.0003, "batch_size": 8, "epochs": 200}
  }
}
```

### **No Further Tuning**
⚠️ **Important**: Hyperparameters are fixed and should NOT be modified during reproduction to prevent p-hacking.

## 🔍 **VALIDATION PROCEDURES**

### **Academic Integrity Checks**
1. **Methodology Pre-registration**: Verified via hash
2. **Data Leakage Prevention**: Automated checks
3. **Reproducibility**: Fixed seeds and deterministic operations
4. **Statistical Power**: Comprehensive power analysis

### **Quality Assurance**
```bash
# Run all integrity checks
python -c "
from src.methodology.preregistration import MethodologyRegistry
from src.data.secure_loader import SecureDataLoader
from src.statistics.power_analysis import PowerAnalysis

# Verify methodology
registry = MethodologyRegistry()
assert registry.verify_methodology()

# Verify data loading
loader = SecureDataLoader()
report = loader.get_preprocessing_report()
assert report['academic_integrity']['train_test_contamination'] == 'prevented'

print('✅ All integrity checks passed!')
"
```

## 📈 **PERFORMANCE BENCHMARKS**

### **Computational Requirements**
- **Training Time**: ~2-5 minutes per dataset (single training)
- **CV Time**: ~20-30 minutes per dataset (10-fold CV)
- **Enhanced Validation**: ~60-90 minutes per dataset
- **Memory Usage**: ~4-6GB GPU memory
- **Total Runtime**: ~4-6 hours for complete reproduction

### **Expected Runtimes**
| Experiment | Miyawaki | Vangerven | MindBigData | Crell | Total |
|------------|----------|-----------|-------------|-------|-------|
| Single Training | 2 min | 3 min | 5 min | 4 min | 14 min |
| 10-Fold CV | 20 min | 25 min | 30 min | 28 min | 103 min |
| Enhanced Validation | 60 min | 75 min | 90 min | 85 min | 310 min |

## 🚨 **TROUBLESHOOTING**

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in configs/optimal_configs.json
   # Or use CPU: python scripts/validate_cccv1.py --device cpu
   ```

2. **Data Loading Errors**
   ```bash
   # Verify data files exist
   ls data/processed/
   
   # Check data integrity
   python src/data/secure_loader.py
   ```

3. **Methodology Verification Failed**
   ```bash
   # Re-create methodology registry
   python src/methodology/preregistration.py
   ```

### **Performance Variations**
- **Acceptable Variance**: ±5% from reported results
- **Hardware Differences**: May cause minor variations
- **Random Seed Issues**: Ensure all seeds are set correctly

## 📞 **SUPPORT**

### **Contact Information**
- **Primary Investigator**: [Name]
- **Email**: [email@institution.edu]
- **Institution**: [Institution Name]
- **GitHub Issues**: [repository_url]/issues

### **Reporting Issues**
When reporting reproduction issues, please include:
1. Complete error messages
2. Hardware specifications
3. Software versions (`pip list`)
4. Steps taken before error
5. Expected vs. actual results

## 📚 **REFERENCES**

1. **Reproducibility Guidelines**: [Relevant standards]
2. **Statistical Methods**: [Statistical references]
3. **Academic Integrity**: [Integrity guidelines]

## ✅ **REPRODUCTION CHECKLIST**

- [ ] Environment setup completed
- [ ] All dependencies installed
- [ ] Data integrity verified
- [ ] Methodology pre-registration verified
- [ ] Random seeds set correctly
- [ ] Primary experiments completed
- [ ] Results within acceptable variance
- [ ] Statistical analysis completed
- [ ] Academic integrity checks passed

---

**Document Version**: 1.0  
**Last Updated**: 2025-06-20  
**Reproducibility Hash**: [To be generated]
