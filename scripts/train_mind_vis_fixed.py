#!/usr/bin/env python3
"""
Fixed Mind-Vis Training
======================

Train Mind-Vis with proper 10-fold CV for fair comparison.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import json

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import load_dataset_gpu_optimized

class SimplifiedMindVis(nn.Module):
    """Simplified Mind-Vis that actually works"""
    
    def __init__(self, input_dim, output_dim, device='cuda'):
        super(SimplifiedMindVis, self).__init__()
        
        self.name = "Mind-Vis"
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple but effective architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        ).to(device)
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, output_dim),
            nn.Sigmoid()
        ).to(device)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
    
    def get_parameter_count(self):
        """Get parameter count"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def train_mind_vis_cv(dataset_name, device='cuda'):
    """Train Mind-Vis with 10-fold cross-validation"""
    
    print(f"🎯 Training Mind-Vis on {dataset_name} with 10-fold CV")
    
    # Load dataset
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device=device)
    
    # Combine for CV
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    
    # Flatten images
    output_dim = y_all.shape[1] * y_all.shape[2] * y_all.shape[3]
    y_flat = y_all.view(y_all.shape[0], -1)
    
    print(f"   Data shape: {X_all.shape} → {y_flat.shape}")
    print(f"   Parameters: Input={input_dim}, Output={output_dim}")
    
    # 10-fold CV
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = []
    best_model = None
    best_score = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        print(f"   Fold {fold+1}/10...")
        
        X_fold_train = X_all[train_idx]
        y_fold_train = y_flat[train_idx]
        X_fold_val = X_all[val_idx]
        y_fold_val = y_flat[val_idx]
        
        # Create model
        model = SimplifiedMindVis(input_dim, output_dim, device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(100):  # More epochs for better training
            # Training step
            optimizer.zero_grad()
            y_pred = model(X_fold_train)
            loss = criterion(y_pred, y_fold_train)
            loss.backward()
            optimizer.step()
            
            # Validation step
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    y_val_pred = model(X_fold_val)
                    val_loss = criterion(y_val_pred, y_fold_val).item()
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 3:  # Early stopping
                    break
                
                model.train()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_fold_val)
            val_loss = criterion(y_val_pred, y_fold_val).item()
        
        cv_scores.append(val_loss)
        print(f"      Fold {fold+1} MSE: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_score:
            best_score = val_loss
            best_model = model.state_dict()
    
    # Save best model
    os.makedirs(f"sota_comparison/mind_vis/models", exist_ok=True)
    model_path = f"sota_comparison/mind_vis/models/{dataset_name}_mind_vis_best.pth"
    
    # Create final model and load best weights
    final_model = SimplifiedMindVis(input_dim, output_dim, device)
    final_model.load_state_dict(best_model)
    torch.save(final_model.state_dict(), model_path)
    
    print(f"✅ Mind-Vis CV training complete")
    print(f"   CV Mean: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
    print(f"   Parameters: {final_model.get_parameter_count():,}")
    print(f"💾 Model saved: {model_path}")
    
    return cv_scores

def train_all_datasets():
    """Train Mind-Vis on all datasets"""
    
    print("🧠 MIND-VIS FIXED TRAINING")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n📊 DATASET: {dataset.upper()}")
        print("-" * 40)
        
        try:
            # Train with CV
            cv_scores = train_mind_vis_cv(dataset, device)
            
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
    results_path = "results/mind_vis_cv_results.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_path}")
    
    # Summary
    print(f"\n📊 MIND-VIS TRAINING SUMMARY:")
    print("=" * 60)
    
    for dataset, result in all_results.items():
        if result['status'] == 'success':
            print(f"✅ {dataset}: {result['mean_mse']:.6f} ± {result['std_mse']:.6f}")
        else:
            print(f"❌ {dataset}: Failed")
    
    return all_results

if __name__ == "__main__":
    train_all_datasets()
