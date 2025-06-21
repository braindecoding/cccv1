#!/usr/bin/env python3
"""
Full Training Cleanup Script
===========================

Clean up all existing results and prepare for fresh training with 100% real data.
Academic Integrity: Remove all potentially fabricated or mixed data.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json

def cleanup_results_directory():
    """Clean up results directory while preserving structure."""
    
    print("🧹 CLEANING UP RESULTS DIRECTORY")
    print("=" * 60)
    
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("✅ Results directory doesn't exist, creating fresh one...")
        results_dir.mkdir(exist_ok=True)
        return
    
    # List all subdirectories to remove
    subdirs_to_remove = []
    files_to_remove = []
    
    for item in results_dir.iterdir():
        if item.is_dir():
            subdirs_to_remove.append(item)
        elif item.is_file():
            files_to_remove.append(item)
    
    print(f"📁 Found {len(subdirs_to_remove)} directories to remove")
    print(f"📄 Found {len(files_to_remove)} files to remove")
    
    # Remove directories
    for subdir in subdirs_to_remove:
        try:
            shutil.rmtree(subdir)
            print(f"🗑️ Removed directory: {subdir.name}")
        except Exception as e:
            print(f"❌ Failed to remove {subdir.name}: {e}")
    
    # Remove files
    for file in files_to_remove:
        try:
            file.unlink()
            print(f"🗑️ Removed file: {file.name}")
        except Exception as e:
            print(f"❌ Failed to remove {file.name}: {e}")
    
    print(f"✅ Results directory cleaned!")

def cleanup_saved_models():
    """Clean up saved model files."""
    
    print("\n🧹 CLEANING UP SAVED MODELS")
    print("=" * 60)
    
    model_dirs = [
        "saved_models",
        "models/saved",
        "checkpoints"
    ]
    
    for model_dir in model_dirs:
        model_path = Path(model_dir)
        if model_path.exists():
            try:
                shutil.rmtree(model_path)
                print(f"🗑️ Removed model directory: {model_dir}")
            except Exception as e:
                print(f"❌ Failed to remove {model_dir}: {e}")
        else:
            print(f"ℹ️ Model directory doesn't exist: {model_dir}")

def cleanup_cache_files():
    """Clean up cache and temporary files."""
    
    print("\n🧹 CLEANING UP CACHE FILES")
    print("=" * 60)
    
    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/.pytest_cache",
        "**/temp_*",
        "**/*.tmp"
    ]
    
    for pattern in cache_patterns:
        for path in Path(".").glob(pattern):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                    print(f"🗑️ Removed cache directory: {path}")
                else:
                    path.unlink()
                    print(f"🗑️ Removed cache file: {path}")
            except Exception as e:
                print(f"❌ Failed to remove {path}: {e}")

def create_fresh_structure():
    """Create fresh directory structure for new training."""
    
    print("\n📁 CREATING FRESH DIRECTORY STRUCTURE")
    print("=" * 60)
    
    directories = [
        "results",
        "results/training_logs",
        "results/cv_results", 
        "results/evaluations",
        "results/visualizations",
        "results/statistical_analysis",
        "saved_models",
        "saved_models/cortexflow",
        "saved_models/mind_vis",
        "saved_models/brain_diffuser"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def create_cleanup_log():
    """Create log of cleanup operation."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"results/cleanup_log_{timestamp}.json")
    
    log_data = {
        "cleanup_timestamp": timestamp,
        "operation": "full_training_cleanup",
        "purpose": "Prepare for 100% real data training",
        "academic_integrity": "All previous results removed to ensure clean slate",
        "next_steps": [
            "Full CortexFlow training on all datasets",
            "Full SOTA models training",
            "Comprehensive cross-validation",
            "Real data evaluation and visualization"
        ]
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n📝 Created cleanup log: {log_file}")

def main():
    """Execute full cleanup for fresh training."""
    
    print("🚀 FULL TRAINING CLEANUP")
    print("=" * 80)
    print("🎯 Goal: Clean slate for 100% real data training")
    print("🏆 Academic Integrity: Remove all potentially mixed data")
    print("=" * 80)
    
    # Confirm cleanup
    print("\n⚠️ WARNING: This will remove ALL existing results and models!")
    print("This ensures 100% academic integrity with real data only.")
    
    # Execute cleanup steps
    cleanup_results_directory()
    cleanup_saved_models()
    cleanup_cache_files()
    create_fresh_structure()
    create_cleanup_log()
    
    print("\n" + "=" * 80)
    print("✅ CLEANUP COMPLETED SUCCESSFULLY!")
    print("🎯 Ready for fresh training with 100% real data")
    print("🏆 Academic integrity ensured - clean slate achieved")
    print("=" * 80)
    
    print("\n📋 NEXT STEPS:")
    print("1. 🧠 Full CortexFlow training on all 4 datasets")
    print("2. 🤖 Full SOTA models training (Mind-Vis, Brain-Diffuser)")
    print("3. 📊 Comprehensive 10-fold cross-validation")
    print("4. 📈 Real data evaluation and statistical analysis")
    print("5. 🖼️ Complete visualization with real data")
    print("6. ✅ Academic integrity verification")

if __name__ == "__main__":
    main()
