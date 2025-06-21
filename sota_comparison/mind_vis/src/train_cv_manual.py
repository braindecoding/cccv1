#!/usr/bin/env python3
"""
Mind-Vis Cross-Validation Training
==================================

Cross-validation training for Mind-Vis using unified framework.
Academic Integrity: Real training for fair comparison.
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys

# Add paths for imports
parent_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / 'src'))

# Import academic seed
from sota_comparison.unified_cv_framework import ACADEMIC_SEED

# Import Mind-Vis model
from sota_comparison.mind_vis.src.mind_vis_manual import (
    create_mind_vis_model, train_mind_vis_model
)

# Import data loader
from data.loader import load_dataset_gpu_optimized

# Set seeds for reproducibility
torch.manual_seed(ACADEMIC_SEED)
np.random.seed(ACADEMIC_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(ACADEMIC_SEED)

def train_mind_vis_single_fold(X_train, y_train, X_val, y_val, input_dim, device='cuda'):
    """Train Mind-Vis for a single fold."""
    
    # Create model
    model = create_mind_vis_model(input_dim, device)
    
    # Train model
    trained_model, val_loss = train_mind_vis_model(
        model, X_train, y_train, X_val, y_val,
        epochs=100, lr=1e-3, batch_size=32, device=device
    )
    
    return trained_model, val_loss

def train_mind_vis_cv(dataset_name, device='cuda', n_folds=10):
    """Train Mind-Vis using cross-validation."""
    
    print(f"🧠 TRAINING MIND-VIS WITH CROSS-VALIDATION")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔄 Folds: {n_folds}")
    print(f"🎯 Academic Seed: {ACADEMIC_SEED}")
    
    # Load dataset
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
    
    if X_train is None:
        print(f"❌ Failed to load dataset: {dataset_name}")
        return None
    
    # Create simple CV framework
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=ACADEMIC_SEED)
    cv_framework = kfold.split(X_train)
    
    cv_scores = []
    fold_models = []
    
    print(f"\n🔄 Starting {n_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(cv_framework):
        print(f"\n📊 Fold {fold + 1}/{n_folds}")
        
        # Get fold data
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]
        
        try:
            # Train model for this fold
            model, val_loss = train_mind_vis_single_fold(
                X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                input_dim, device
            )
            
            cv_scores.append(val_loss)
            fold_models.append(model)
            
            print(f"✅ Fold {fold + 1} complete: Val Loss = {val_loss:.6f}")
            
        except Exception as e:
            print(f"❌ Fold {fold + 1} failed: {e}")
            cv_scores.append(float('inf'))
            fold_models.append(None)
    
    # Find best fold
    valid_scores = [score for score in cv_scores if score != float('inf')]
    
    if not valid_scores:
        print(f"❌ All folds failed for {dataset_name}")
        return None
    
    best_fold = cv_scores.index(min(valid_scores))
    best_model = fold_models[best_fold]
    best_score = cv_scores[best_fold]
    
    print(f"\n✅ Cross-validation complete!")
    print(f"📊 CV Mean: {np.mean(valid_scores):.6f}")
    print(f"📊 CV Std: {np.std(valid_scores):.6f}")
    print(f"🏆 Best Fold: {best_fold + 1} (Score: {best_score:.6f})")
    
    # Save best model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_file = model_dir / f"Mind-Vis-{dataset_name}_cv_best.pth"
    metadata_file = model_dir / f"Mind-Vis-{dataset_name}_cv_best_metadata.json"
    
    # Save model state
    torch.save(best_model.state_dict(), model_file)
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'model_type': 'Mind-Vis',
        'input_dim': input_dim,
        'best_fold': best_fold + 1,
        'best_score': best_score,
        'cv_mean': np.mean(valid_scores),
        'cv_std': np.std(valid_scores),
        'n_folds': n_folds,
        'academic_seed': ACADEMIC_SEED,
        'timestamp': datetime.now().isoformat(),
        'academic_compliant': True
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Model saved: {model_file}")
    print(f"💾 Metadata saved: {metadata_file}")
    
    # Return results in unified format
    return {
        'cv_mean': np.mean(valid_scores),
        'cv_std': np.std(valid_scores),
        'best_fold': best_fold + 1,
        'best_score': best_score,
        'model_file': str(model_file),
        'metadata_file': str(metadata_file),
        'academic_compliant': True
    }

def main():
    """Train Mind-Vis for all datasets."""
    
    print("🧠 MIND-VIS CROSS-VALIDATION TRAINING")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎯 Using device: {device}")
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING MIND-VIS FOR: {dataset.upper()}")
        print(f"{'='*60}")
        
        try:
            results = train_mind_vis_cv(dataset, device)
            
            if results:
                print(f"✅ Mind-Vis training complete for {dataset}")
                print(f"📊 CV Score: {results['cv_mean']:.6f} ± {results['cv_std']:.6f}")
            else:
                print(f"❌ Mind-Vis training failed for {dataset}")
                
        except Exception as e:
            print(f"❌ Error training {dataset}: {e}")
            continue
    
    print(f"\n✅ Mind-Vis training complete for all datasets!")

if __name__ == "__main__":
    main()
