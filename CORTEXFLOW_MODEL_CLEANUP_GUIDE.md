# CortexFlow Model Cleanup Guide
## Analysis and Recommendations for Model File Organization

**Analysis Date**: June 21, 2025  
**Total CortexFlow Files Found**: 217 files (1,480 MB)  
**Potential Space Savings**: 47.6 MB from safe removals

---

## 📊 **CURRENT MODEL INVENTORY**

### **📁 Files by Category:**
| Category | Count | Size (MB) | Description |
|----------|-------|-----------|-------------|
| **PyTorch Models** | 4 | 1,431.1 | Active model checkpoints |
| **Source Code** | 47 | 0.7 | Python model definitions |
| **Metadata** | 103 | 33.6 | JSON configuration files |
| **Unknown** | 63 | 14.5 | Other related files |

### **🔍 Usage Analysis:**
- **Scripts using cortexflow_clip_cnn_v1.py**: 14 scripts
- **Scripts using CortexFlowCLIPCNNV1**: 11 scripts  
- **Other CortexFlow references**: 47 scripts

---

## 🎯 **ACTIVE vs REDUNDANT MODELS**

### **✅ ACTIVE MODELS (KEEP):**

#### **🏆 Primary Checkpoints (Currently Used):**
```
models/miyawaki_cv_best.pth (351.5 MB) ✅ ACTIVE
models/vangerven_cv_best.pth (359.9 MB) ✅ ACTIVE  
models/crell_cv_best.pth (359.8 MB) ✅ ACTIVE
models/mindbigdata_cv_best.pth (359.9 MB) ✅ ACTIVE
```
**Total**: 1,431.1 MB - **These are the models used in fair comparison**

#### **📋 Essential Metadata:**
```
models/miyawaki_cv_best_metadata.json ✅ ACTIVE
models/vangerven_cv_best_metadata.json ✅ ACTIVE
models/crell_cv_best_metadata.json ✅ ACTIVE  
models/mindbigdata_cv_best_metadata.json ✅ ACTIVE
```

#### **🔧 Core Source Code:**
```
src/models/cortexflow_clip_cnn_v1.py ✅ ESSENTIAL
```
**This is the main model definition used by all scripts**

### **⚠️ DUPLICATE MODELS (REVIEW NEEDED):**

#### **🔄 Alternative Format Checkpoints:**
```
models/miyawaki_cccv1_best.pth (351.5 MB) ⚠️ DUPLICATE?
models/vangerven_cccv1_best.pth (359.9 MB) ⚠️ DUPLICATE?
models/crell_cccv1_best.pth (359.8 MB) ⚠️ DUPLICATE?
models/mindbigdata_cccv1_best.pth (359.9 MB) ⚠️ DUPLICATE?
```
**Total**: 1,431.1 MB - **Same size as cv_best versions**

### **🗑️ SAFE TO REMOVE:**

#### **📦 Publication Package Backups (205 files, 47.6 MB):**
- All files in `publication_packages/` directories
- These are backup copies created for academic publication packages
- **Safe to remove** as they duplicate active files

---

## 🔍 **DETAILED ANALYSIS**

### **🤔 Key Questions:**

#### **1. Are `*_cccv1_best.pth` files duplicates of `*_cv_best.pth`?**
- **Same file sizes** suggest they might be identical
- **Need verification** before removal
- **Recommendation**: Compare file checksums

#### **2. Which checkpoint format is actually used?**
- **Current scripts use**: `{dataset}_cv_best.pth` format
- **Alternative format**: `{dataset}_cccv1_best.pth` format
- **Recommendation**: Verify which format is loaded by active scripts

#### **3. Are there any legacy model versions?**
- **Multiple source code files** in different directories
- **Some may be outdated** versions
- **Recommendation**: Identify the canonical version

---

## 🛠️ **CLEANUP RECOMMENDATIONS**

### **🚀 IMMEDIATE ACTIONS (SAFE):**

#### **1. Remove Publication Package Backups:**
```bash
# Safe to remove - these are just backup copies
rm -rf publication_packages/
```
**Space Saved**: 47.6 MB

#### **2. Clean Up Cache Files:**
```bash
# Remove Python cache files
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### **🔍 VERIFICATION NEEDED:**

#### **1. Compare Duplicate Checkpoints:**
```bash
# Check if cccv1_best and cv_best files are identical
md5sum models/miyawaki_cv_best.pth models/miyawaki_cccv1_best.pth
md5sum models/vangerven_cv_best.pth models/vangerven_cccv1_best.pth
md5sum models/crell_cv_best.pth models/crell_cccv1_best.pth
md5sum models/mindbigdata_cv_best.pth models/mindbigdata_cccv1_best.pth
```

#### **2. Verify Active Usage:**
```bash
# Check which checkpoint format is actually loaded
grep -r "cv_best.pth" scripts/
grep -r "cccv1_best.pth" scripts/
```

### **⚠️ CONDITIONAL CLEANUP:**

#### **If `*_cccv1_best.pth` files are duplicates:**
```bash
# Remove duplicate checkpoints (ONLY if verified as duplicates)
rm models/*_cccv1_best.pth
```
**Potential Space Saved**: 1,431.1 MB (50% reduction!)

---

## 📋 **VERIFICATION SCRIPT**

### **🔧 Recommended Verification Process:**

```bash
#!/bin/bash
# Verify model file relationships

echo "🔍 VERIFYING CORTEXFLOW MODEL FILES"
echo "=================================="

# 1. Check file sizes
echo "📊 File sizes:"
ls -lh models/*_cv_best.pth models/*_cccv1_best.pth

# 2. Check checksums
echo "🔐 Checksums:"
for dataset in miyawaki vangerven crell mindbigdata; do
    echo "Dataset: $dataset"
    if [ -f "models/${dataset}_cv_best.pth" ] && [ -f "models/${dataset}_cccv1_best.pth" ]; then
        md5sum "models/${dataset}_cv_best.pth" "models/${dataset}_cccv1_best.pth"
    fi
done

# 3. Check which files are referenced in scripts
echo "📝 Script references:"
echo "cv_best references:"
grep -r "cv_best.pth" scripts/ | wc -l
echo "cccv1_best references:"  
grep -r "cccv1_best.pth" scripts/ | wc -l
```

---

## 🎯 **FINAL RECOMMENDATIONS**

### **✅ IMMEDIATE ACTIONS:**
1. **Remove publication packages** (safe, 47.6 MB saved)
2. **Clean Python cache files** (safe, small space saved)
3. **Run verification script** to check for duplicates

### **🔍 AFTER VERIFICATION:**
1. **If duplicates confirmed**: Remove `*_cccv1_best.pth` files (1.4 GB saved)
2. **Update scripts** to use consistent naming convention
3. **Document** the canonical model file format

### **📚 ORGANIZATION BENEFITS:**
- **Cleaner repository** structure
- **Reduced storage** requirements  
- **Clearer model** versioning
- **Easier maintenance** and deployment

### **⚠️ SAFETY NOTES:**
- **Always backup** before removing large model files
- **Verify functionality** after cleanup
- **Test model loading** with remaining files
- **Keep at least one working copy** of each trained model

---

## 🏆 **EXPECTED OUTCOME**

### **After Complete Cleanup:**
- **Files Removed**: 205+ files
- **Space Saved**: 47.6 MB (guaranteed) + up to 1.4 GB (if duplicates)
- **Repository Size**: Reduced by 50-75%
- **Organization**: Much cleaner and more maintainable

### **Maintained Functionality:**
- ✅ All fair comparison capabilities
- ✅ All visualization scripts  
- ✅ All model loading functionality
- ✅ All academic integrity features

**The cleanup will significantly improve repository organization while maintaining all essential functionality!** 🎉
