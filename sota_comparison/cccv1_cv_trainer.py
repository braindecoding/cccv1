"""
CCCV1 CV-Compliant Training Script
=================================

Train CCCV1 using unified CV framework for academic-compliant comparison.
Uses same 10-fold CV strategy and seed=42 as other methods.

Academic Compliance:
- Uses unified CV framework with seed=42
- Same 10-fold CV strategy as other methods
- Identical data splits for fair comparison
- Reproducible results
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

# Import unified CV framework
from sota_comparison.unified_cv_framework import create_unified_cv_framework, ACADEMIC_SEED

# Import CCCV1 components
try:
    from src.models.cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized
except ImportError:
    sys.path.append(os.path.join(parent_dir, 'src', 'models'))
    from cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized

# Import data loader
sys.path.append(parent_dir)
from data.loader import load_dataset_gpu_optimized

# Set seeds for reproducibility
torch.manual_seed(ACADEMIC_SEED)
np.random.seed(ACADEMIC_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(ACADEMIC_SEED)


def load_optimal_config(dataset_name):
    """Load optimal configuration for dataset"""
    config_path = Path("cccv1/configs/optimal_configurations.json")
    
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        dataset_config = configs['cccv1_optimal_configurations']['datasets'][dataset_name]
        return dataset_config['optimal_config']
    
    except Exception as e:
        print(f"⚠️ Could not load config for {dataset_name}: {e}")
        # Return default config
        return {
            'architecture': {
                'dropout_encoder': 0.06, 
                'dropout_decoder': 0.02, 
                'clip_residual_weight': 0.1
            },
            'training': {
                'lr': 0.001, 
                'batch_size': 16, 
                'weight_decay': 1e-6, 
                'epochs': 100, 
                'patience': 15
            }
        }


def train_cccv1_fold(train_loader, val_loader, input_dim, device='cuda', 
                    fold=0, dataset_name='miyawaki', **kwargs):
    """
    Train CCCV1 for one CV fold
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input dimensionality
        device: Computing device
        fold: Current fold number
        dataset_name: Name of dataset
        
    Returns:
        Trained model
    """
    # Load optimal configuration
    config = load_optimal_config(dataset_name)
    training_config = config['training']
    
    # Create CCCV1 model
    model = CortexFlowCLIPCNNV1Optimized(input_dim, device, dataset_name).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay']
    )
    criterion = nn.MSELoss()
    
    # Training parameters
    epochs = training_config['epochs']
    patience = training_config['patience']
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (fmri_data, visual_data) in enumerate(train_loader):
            fmri_data = fmri_data.to(device)
            visual_data = visual_data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(fmri_data)
            loss = criterion(reconstructed, visual_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for fmri_data, visual_data in val_loader:
                fmri_data = fmri_data.to(device)
                visual_data = visual_data.to(device)
                
                reconstructed = model(fmri_data)
                loss = criterion(reconstructed, visual_data)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_cccv1_fold(model, val_loader, device='cuda'):
    """
    Evaluate CCCV1 for one CV fold
    
    Args:
        model: Trained CCCV1 model
        val_loader: Validation data loader
        device: Computing device
        
    Returns:
        MSE score
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for fmri_data, visual_data in val_loader:
            fmri_data = fmri_data.to(device)
            visual_data = visual_data.to(device)
            
            reconstructed = model(fmri_data)
            loss = criterion(reconstructed, visual_data)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_cccv1_cv(dataset_name, device='cuda', n_folds=10):
    """
    Train CCCV1 using unified CV framework
    
    Args:
        dataset_name: Name of dataset
        device: Computing device
        n_folds: Number of CV folds
        
    Returns:
        CV results dictionary
    """
    print(f"🧠 Training CCCV1-Optimized with Unified CV Framework")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔄 Folds: {n_folds}")
    print(f"🎯 Academic Seed: {ACADEMIC_SEED}")
    
    # Load dataset
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
    
    if X_train is None:
        print(f"❌ Failed to load dataset: {dataset_name}")
        return None
    
    # Create unified CV framework
    cv_framework = create_unified_cv_framework(n_folds=n_folds)
    
    # Method-specific parameters
    method_kwargs = {
        'dataset_name': dataset_name,
        'batch_size': 16
    }
    
    # Run CV evaluation
    results = cv_framework.evaluate_method_cv(
        method_name=f"CCCV1-Optimized-{dataset_name}",
        train_func=train_cccv1_fold,
        evaluate_func=evaluate_cccv1_fold,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        input_dim=input_dim,
        device=device,
        **method_kwargs
    )
    
    return results


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CCCV1 with Unified CV Framework')
    parser.add_argument('--dataset', type=str, default='miyawaki',
                       choices=['miyawaki', 'vangerven', 'mindbigdata', 'crell'],
                       help='Dataset to train on')
    parser.add_argument('--folds', type=int, default=10,
                       help='Number of CV folds')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computing device')
    
    args = parser.parse_args()
    
    print("🧠 CCCV1-Optimized CV-Compliant Training")
    print("=" * 45)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔄 Folds: {args.folds}")
    print(f"💻 Device: {args.device}")
    print(f"🎯 Academic Seed: {ACADEMIC_SEED}")
    print()
    
    # Train with CV framework
    results = train_cccv1_cv(
        dataset_name=args.dataset,
        device=args.device,
        n_folds=args.folds
    )
    
    if results and results.get('academic_compliant', False):
        print(f"\n✅ CCCV1-Optimized CV Training Complete!")
        print(f"📊 CV Score: {results['cv_mean']:.6f} ± {results['cv_std']:.6f}")
        print(f"🎯 Academic Compliant: {results['academic_compliant']}")
        
        # Save results
        results_dir = Path("sota_comparison/comparison_results")
        results_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"cccv1_cv_results_{args.dataset}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved: {results_file}")
    else:
        print(f"\n❌ CCCV1-Optimized CV Training Failed!")


if __name__ == "__main__":
    main()
