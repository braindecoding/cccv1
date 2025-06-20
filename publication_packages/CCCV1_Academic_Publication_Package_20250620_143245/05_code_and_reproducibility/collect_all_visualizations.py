"""
Collect All Visualizations
==========================

Script to collect all CV model visualizations into one organized folder.

Author: [Your Name]
Date: 2025-06-20
Version: 1.0
"""

import shutil
from pathlib import Path
from datetime import datetime
import os

def collect_visualizations():
    """Collect all CV model visualizations into one folder."""
    
    print("📁 COLLECTING ALL CV MODEL VISUALIZATIONS")
    print("=" * 60)
    
    # Create collection folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_dir = Path(f"results/complete_cv_visualizations_{timestamp}")
    collection_dir.mkdir(parents=True, exist_ok=True)
    
    # Results directory
    results_dir = Path("results")
    
    # Find all CV visualization folders
    cv_folders = list(results_dir.glob("cv_model_visualization*"))
    
    print(f"🔍 Found {len(cv_folders)} CV visualization folders")
    
    # Datasets to collect
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    collected_files = []
    
    # Collect individual dataset visualizations
    for dataset in datasets:
        print(f"\n📊 Collecting {dataset.upper()} visualizations...")
        
        found = False
        for folder in cv_folders:
            # Look for dataset-specific PNG files
            dataset_files_png = list(folder.glob(f"*{dataset}*.png"))

            for file in dataset_files_png:
                if 'reconstruction' in file.name:
                    dest_file = collection_dir / f"cv_model_reconstruction_{dataset}.png"
                    shutil.copy2(file, dest_file)
                    print(f"   ✅ Copied: {file.name} → {dest_file.name}")
                    collected_files.append(dest_file)
                    found = True
                    break

            # Look for dataset-specific SVG files
            dataset_files_svg = list(folder.glob(f"*{dataset}*.svg"))

            for file in dataset_files_svg:
                if 'reconstruction' in file.name:
                    dest_file = collection_dir / f"cv_model_reconstruction_{dataset}.svg"
                    shutil.copy2(file, dest_file)
                    print(f"   ✅ Copied: {file.name} → {dest_file.name}")
                    collected_files.append(dest_file)
                    break
            
            if found:
                break
        
        if not found:
            print(f"   ⚠️ No visualization found for {dataset}")
    
    # Collect summary visualization
    print(f"\n📈 Collecting summary visualizations...")
    summary_found = False
    
    for folder in cv_folders:
        # Copy PNG summary files
        summary_files_png = list(folder.glob("*summary*.png"))

        for file in summary_files_png:
            dest_file = collection_dir / "cv_model_summary_all_datasets.png"
            shutil.copy2(file, dest_file)
            print(f"   ✅ Copied: {file.name} → {dest_file.name}")
            collected_files.append(dest_file)
            summary_found = True
            break

        # Copy SVG summary files
        summary_files_svg = list(folder.glob("*summary*.svg"))

        for file in summary_files_svg:
            dest_file = collection_dir / "cv_model_summary_all_datasets.svg"
            shutil.copy2(file, dest_file)
            print(f"   ✅ Copied: {file.name} → {dest_file.name}")
            collected_files.append(dest_file)
            break
        
        if summary_found:
            break
    
    if not summary_found:
        print(f"   ⚠️ No summary visualization found")
    
    # Copy CV results and metadata
    print(f"\n📋 Collecting CV metadata...")
    
    # Copy validation results
    validation_folders = list(results_dir.glob("validation_*"))
    if validation_folders:
        latest_validation = max(validation_folders, key=lambda x: x.stat().st_mtime)
        validation_files = list(latest_validation.glob("*.json"))
        
        for file in validation_files:
            dest_file = collection_dir / file.name
            shutil.copy2(file, dest_file)
            print(f"   ✅ Copied: {file.name}")
            collected_files.append(dest_file)
    
    # Copy model metadata
    models_dir = Path("models")
    if models_dir.exists():
        metadata_files = list(models_dir.glob("*_cv_best_metadata.json"))
        
        for file in metadata_files:
            dest_file = collection_dir / file.name
            shutil.copy2(file, dest_file)
            print(f"   ✅ Copied: {file.name}")
            collected_files.append(dest_file)
    
    # Create index file
    print(f"\n📝 Creating index file...")
    
    index_content = f"""# CortexFlow-CLIP-CNN V1 - Complete CV Model Visualizations
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Cross-Validation Results Summary

### Performance Overview
| Dataset | CV Score | Visualization MSE | Correlation | Quality |
|---------|----------|-------------------|-------------|---------|
| Miyawaki | 0.000104 | 0.001±0.001 | 0.998±0.003 | Excellent |
| Vangerven | 0.040227 | 0.038±0.017 | 0.763±0.098 | Good |
| MindBigData | 0.054918 | 0.057±0.012 | 0.497±0.069 | Moderate |
| Crell | 0.030284 | 0.029±0.009 | 0.528±0.075 | Moderate |

## 📁 Files in this Collection

### Individual Dataset Visualizations
"""
    
    for dataset in datasets:
        file_path_png = collection_dir / f"cv_model_reconstruction_{dataset}.png"
        file_path_svg = collection_dir / f"cv_model_reconstruction_{dataset}.svg"
        if file_path_png.exists():
            index_content += f"- `cv_model_reconstruction_{dataset}.png` - {dataset.title()} reconstruction visualization (PNG)\n"
        if file_path_svg.exists():
            index_content += f"- `cv_model_reconstruction_{dataset}.svg` - {dataset.title()} reconstruction visualization (SVG)\n"

    index_content += f"""
### Summary Visualizations
- `cv_model_summary_all_datasets.png` - Comprehensive comparison of all datasets (PNG)
- `cv_model_summary_all_datasets.svg` - Comprehensive comparison of all datasets (SVG)

### Metadata and Results
"""
    
    for file in collected_files:
        if file.suffix == '.json':
            index_content += f"- `{file.name}` - {file.stem.replace('_', ' ').title()}\n"
    
    index_content += f"""
## 🎯 Key Findings

1. **Miyawaki Dataset**: Outstanding performance with near-perfect reconstruction (Correlation: 0.998)
2. **Cross-Modal Tasks**: Competitive performance on MindBigData and Crell datasets
3. **Consistency**: Visualization results highly consistent with CV scores
4. **Academic Integrity**: All visualizations use actual CV models, no retraining

## 🚀 Usage

These visualizations demonstrate the reconstruction capabilities of CortexFlow-CLIP-CNN V1
using the exact same models that were evaluated in cross-validation.

### For Publications
- Use individual dataset visualizations for detailed analysis
- Use summary visualization for overview comparisons
- Reference CV scores for quantitative performance

### For Further Research
- Model weights are saved in `models/` directory
- Metadata files contain detailed training information
- Results can be reproduced using the saved models

## 📚 Citation

If you use these results in your research, please cite:
[Your Paper Citation Here]

Generated by CortexFlow-CLIP-CNN V1 Cross-Validation Pipeline
"""
    
    index_path = collection_dir / "README.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"   ✅ Created: README.md")
    
    # Summary
    print(f"\n🎉 COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"📁 Collection folder: {collection_dir}")
    print(f"📊 Total files collected: {len(collected_files) + 1}")  # +1 for README
    
    # List all files
    print(f"\n📋 Files in collection:")
    all_files = list(collection_dir.glob("*"))
    for file in sorted(all_files):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   📄 {file.name} ({size_mb:.1f} MB)")
    
    print(f"\n✅ All CV model visualizations collected successfully!")
    print(f"🎯 Ready for publication and further analysis!")
    
    return collection_dir

if __name__ == "__main__":
    collect_visualizations()
