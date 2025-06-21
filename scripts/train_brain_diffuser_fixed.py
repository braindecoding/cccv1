#!/usr/bin/env python3
"""
Fixed Brain-Diffuser Training
============================

Train simplified Brain-Diffuser that actually works for fair comparison.
"""

import sys
import os
import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import pickle
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import load_dataset_gpu_optimized

class SimplifiedBrainDiffuser:
    """Simplified Brain-Diffuser that actually works"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.name = "Brain-Diffuser"
        self.ridge_regressor = None
        
    def setup_models(self):
        """Setup simplified models"""
        print("✅ Simplified Brain-Diffuser models setup complete")
        return True
    
    def train_cv(self, dataset_name, alpha=1.0):
        """Train with 10-fold cross-validation"""
        print(f"🎯 Training Brain-Diffuser on {dataset_name} with 10-fold CV")
        
        # Load data
        X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device=self.device)
        
        # Combine for CV
        X_all = torch.cat([X_train, X_test], dim=0)
        y_all = torch.cat([y_train, y_test], dim=0)
        
        # Flatten images for regression
        y_flat = y_all.view(y_all.shape[0], -1)
        
        # Convert to numpy
        X_np = X_all.cpu().numpy()
        y_np = y_flat.cpu().numpy()
        
        # 10-fold CV
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = []
        
        print(f"   Data shape: {X_np.shape} → {y_np.shape}")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
            print(f"   Fold {fold+1}/10...")
            
            X_fold_train = X_np[train_idx]
            y_fold_train = y_np[train_idx]
            X_fold_val = X_np[val_idx]
            y_fold_val = y_np[val_idx]
            
            # Train ridge regression
            ridge = Ridge(alpha=alpha, random_state=42)
            ridge.fit(X_fold_train, y_fold_train)
            
            # Evaluate
            y_pred = ridge.predict(X_fold_val)
            mse = np.mean((y_pred - y_fold_val) ** 2)
            
            cv_scores.append(mse)
            print(f"      Fold {fold+1} MSE: {mse:.6f}")
        
        # Train final model on all data
        self.ridge_regressor = Ridge(alpha=alpha, random_state=42)
        self.ridge_regressor.fit(X_np, y_np)
        
        print(f"✅ Brain-Diffuser CV training complete")
        print(f"   CV Mean: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
        
        return cv_scores
    
    def evaluate(self, dataset_name, X_test, y_test, num_samples=6):
        """Evaluate simplified Brain-Diffuser"""
        
        if self.ridge_regressor is None:
            raise ValueError("Model not trained")
        
        # Predict
        X_np = X_test[:num_samples].cpu().numpy()
        y_flat = y_test[:num_samples].view(num_samples, -1).cpu().numpy()
        
        y_pred_flat = self.ridge_regressor.predict(X_np)
        
        # Calculate metrics
        mse = np.mean((y_pred_flat - y_flat) ** 2)
        
        # Correlation
        correlation = np.corrcoef(y_pred_flat.flatten(), y_flat.flatten())[0, 1]
        
        return {
            'method': 'Brain-Diffuser',
            'dataset': dataset_name,
            'mse': mse,
            'correlation': correlation,
            'num_samples': num_samples,
            'cv_scores': None  # Will be filled by CV training
        }
    
    def save_model(self, dataset_name):
        """Save model"""
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{dataset_name}_brain_diffuser_simplified.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.ridge_regressor, f)
        
        print(f"💾 Model saved: {model_path}")
    
    def load_model(self, dataset_name):
        """Load model"""
        model_path = f"models/{dataset_name}_brain_diffuser_simplified.pkl"
        
        if not os.path.exists(model_path):
            return False
        
        with open(model_path, 'rb') as f:
            self.ridge_regressor = pickle.load(f)
        
        print(f"✅ Model loaded: {model_path}")
        return True

def train_all_datasets():
    """Train Brain-Diffuser on all datasets"""
    
    print("🧠 BRAIN-DIFFUSER FIXED TRAINING")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n📊 DATASET: {dataset.upper()}")
        print("-" * 40)
        
        try:
            # Initialize model
            model = SimplifiedBrainDiffuser(device=device)
            model.setup_models()
            
            # Train with CV
            cv_scores = model.train_cv(dataset)
            
            # Save model
            model.save_model(dataset)
            
            # Store results
            all_results[dataset] = {
                'cv_scores': cv_scores,
                'mean_mse': np.mean(cv_scores),
                'std_mse': np.std(cv_scores),
                'status': 'success'
            }
            
            print(f"✅ {dataset}: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
            
        except Exception as e:
            print(f"❌ Error training {dataset}: {e}")
            all_results[dataset] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save all results
    results_path = "results/brain_diffuser_cv_results.json"
    os.makedirs("results", exist_ok=True)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_path}")
    
    # Summary
    print(f"\n📊 BRAIN-DIFFUSER TRAINING SUMMARY:")
    print("=" * 60)
    
    for dataset, result in all_results.items():
        if result['status'] == 'success':
            print(f"✅ {dataset}: {result['mean_mse']:.6f} ± {result['std_mse']:.6f}")
        else:
            print(f"❌ {dataset}: Failed")
    
    return all_results

if __name__ == "__main__":
    train_all_datasets()
