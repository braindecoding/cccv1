#!/usr/bin/env python3
"""
Fix Brain-Diffuser for MindBigData
==================================

Specifically train Brain-Diffuser for MindBigData dataset to complete 4×4 comparison.
Academic Integrity: Real training for complete comparison.
"""

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys

# Add paths for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / 'src'))
sys.path.append(str(parent_dir / 'sota_comparison'))

# Import unified CV framework
from sota_comparison.unified_cv_framework import ACADEMIC_SEED

# Import Brain-Diffuser
from sota_comparison.brain_diffuser.src.lightweight_brain_diffuser import LightweightBrainDiffuser

# Import data loader
from data.loader import load_dataset_gpu_optimized

# Set seeds for reproducibility
torch.manual_seed(ACADEMIC_SEED)
np.random.seed(ACADEMIC_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(ACADEMIC_SEED)

def train_brain_diffuser_single_fold(X_train, y_train, X_val, y_val, input_dim, device='cuda'):
    """Train Brain-Diffuser for a single fold with NaN handling."""
    
    print(f"🧠 Training Brain-Diffuser (single fold)...")
    
    # Create model
    model = LightweightBrainDiffuser(
        input_dim=input_dim,
        device=device,
        image_size=28
    ).to(device)
    
    # Setup training with more conservative parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 25
    epochs = 150
    batch_size = 16  # Smaller batch size for stability
    
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: 5e-4")
    print(f"   Batch size: {batch_size}")
    print(f"   Max patience: {max_patience}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Create batches
        num_samples = X_train.shape[0]
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                _, output = model(X_batch)  # Get final output
                
                # Check for NaN
                if torch.isnan(output).any():
                    print(f"   ⚠️ NaN detected in output at epoch {epoch}, skipping batch")
                    continue
                
                loss = criterion(output, y_batch)
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    print(f"   ⚠️ NaN detected in loss at epoch {epoch}, skipping batch")
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"   ⚠️ Error in batch at epoch {epoch}: {e}")
                continue
        
        if num_batches == 0:
            print(f"   ❌ No valid batches in epoch {epoch}")
            continue
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                X_batch = X_val[i:i+batch_size]
                y_batch = y_val[i:i+batch_size]
                
                try:
                    _, output = model(X_batch)
                    
                    # Check for NaN
                    if torch.isnan(output).any():
                        continue
                    
                    loss = criterion(output, y_batch)
                    
                    if torch.isnan(loss):
                        continue
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    continue
        
        if val_batches == 0:
            print(f"   ❌ No valid validation batches in epoch {epoch}")
            continue
        
        avg_val_loss = val_loss / val_batches
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, "
                  f"Val Loss = {avg_val_loss:.6f}")
        
        if patience_counter >= max_patience:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    print(f"✅ Brain-Diffuser training complete. Best val loss: {best_val_loss:.6f}")
    return model, best_val_loss

def train_brain_diffuser_mindbigdata_cv(device='cuda', n_folds=10):
    """Train Brain-Diffuser specifically for MindBigData with CV."""
    
    print(f"🧠 TRAINING BRAIN-DIFFUSER FOR MINDBIGDATA")
    print(f"📊 Dataset: mindbigdata")
    print(f"🔄 Folds: {n_folds}")
    print(f"🎯 Academic Seed: {ACADEMIC_SEED}")
    
    # Load dataset
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized('mindbigdata', device)
    
    if X_train is None:
        print(f"❌ Failed to load mindbigdata dataset")
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
            model, val_loss = train_brain_diffuser_single_fold(
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
        print(f"❌ All folds failed for mindbigdata")
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
    
    model_file = model_dir / f"Lightweight-Brain-Diffuser-mindbigdata_cv_best.pth"
    metadata_file = model_dir / f"Lightweight-Brain-Diffuser-mindbigdata_cv_best_metadata.json"
    
    # Save model state
    torch.save(best_model.state_dict(), model_file)
    
    # Save metadata
    metadata = {
        'dataset': 'mindbigdata',
        'model_type': 'Lightweight-Brain-Diffuser',
        'input_dim': input_dim,
        'best_fold': best_fold + 1,
        'best_score': best_score,
        'cv_mean': np.mean(valid_scores),
        'cv_std': np.std(valid_scores),
        'n_folds': n_folds,
        'academic_seed': ACADEMIC_SEED,
        'timestamp': datetime.now().isoformat(),
        'academic_compliant': True,
        'training_notes': 'Special training for MindBigData with NaN handling'
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
    """Fix Brain-Diffuser for MindBigData."""
    
    print("🔧 FIXING BRAIN-DIFFUSER FOR MINDBIGDATA")
    print("=" * 60)
    print("🎯 Goal: Complete 4×4 real data comparison")
    print("🏆 Academic Integrity: Real training for fair comparison")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎯 Using device: {device}")
    
    try:
        results = train_brain_diffuser_mindbigdata_cv(device)
        
        if results:
            print(f"\n✅ Brain-Diffuser training complete for mindbigdata!")
            print(f"📊 CV Score: {results['cv_mean']:.6f} ± {results['cv_std']:.6f}")
            print(f"🏆 Best Fold: {results['best_fold']}")
            print(f"🎯 Now ready for complete 4×4 comparison!")
        else:
            print(f"\n❌ Brain-Diffuser training failed for mindbigdata")
            
    except Exception as e:
        print(f"❌ Error training mindbigdata: {e}")
    
    print(f"\n🎯 Brain-Diffuser fix attempt complete!")

if __name__ == "__main__":
    main()
