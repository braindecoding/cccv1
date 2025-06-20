"""
Lightweight Brain-Diffuser CV-Compliant Training Script
======================================================

Train Lightweight Brain-Diffuser using unified CV framework for academic-compliant comparison.
Maintains exact two-stage methodology while ensuring fair comparison.

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
from pathlib import Path

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.extend([root_dir, parent_dir, current_dir])

# Import unified CV framework
sys.path.append(root_dir)
from sota_comparison.unified_cv_framework import create_unified_cv_framework, ACADEMIC_SEED

# Import Lightweight Brain-Diffuser model
from lightweight_brain_diffuser import LightweightBrainDiffuser

# Import data loader
sys.path.append(root_dir)
from data.loader import load_dataset_gpu_optimized

# Set seeds for reproducibility
torch.manual_seed(ACADEMIC_SEED)
np.random.seed(ACADEMIC_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(ACADEMIC_SEED)


def train_brain_diffuser_fold(train_loader, val_loader, input_dim, device='cuda', 
                             fold=0, epochs=50, lr=0.001, patience=10, **kwargs):
    """
    Train Lightweight Brain-Diffuser for one CV fold
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input dimensionality
        device: Computing device
        fold: Current fold number
        epochs: Maximum epochs
        lr: Learning rate
        patience: Early stopping patience
        
    Returns:
        Trained model
    """
    # Get image size from first batch
    sample_batch = next(iter(train_loader))
    image_size = sample_batch[1].shape[-1]  # Assuming square images
    
    # Create Lightweight Brain-Diffuser model
    model = LightweightBrainDiffuser(
        input_dim=input_dim,
        device=device,
        image_size=image_size
    ).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()
    
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
            
            # Forward pass (two-stage: VDVAE + Diffusion)
            vae_output, diffusion_output = model(fmri_data, use_diffusion=True)
            
            # Loss on final diffusion output
            loss = criterion(diffusion_output, visual_data)
            
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
                
                # Forward pass with diffusion
                _, diffusion_output = model(fmri_data, use_diffusion=True)
                loss = criterion(diffusion_output, visual_data)
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


def evaluate_brain_diffuser_fold(model, val_loader, device='cuda'):
    """
    Evaluate Lightweight Brain-Diffuser for one CV fold
    
    Args:
        model: Trained Brain-Diffuser model
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
            
            # Forward pass with diffusion
            _, diffusion_output = model(fmri_data, use_diffusion=True)
            loss = criterion(diffusion_output, visual_data)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_brain_diffuser_cv(dataset_name, device='cuda', n_folds=10):
    """
    Train Lightweight Brain-Diffuser using unified CV framework
    
    Args:
        dataset_name: Name of dataset
        device: Computing device
        n_folds: Number of CV folds
        
    Returns:
        CV results dictionary
    """
    print(f"🧠 Training Lightweight Brain-Diffuser with Unified CV Framework")
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
        'epochs': 50,
        'lr': 0.001,
        'patience': 10,
        'batch_size': 16
    }
    
    # Run CV evaluation
    results = cv_framework.evaluate_method_cv(
        method_name=f"Lightweight-Brain-Diffuser-{dataset_name}",
        train_func=train_brain_diffuser_fold,
        evaluate_func=evaluate_brain_diffuser_fold,
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
    
    parser = argparse.ArgumentParser(description='Train Lightweight Brain-Diffuser with Unified CV Framework')
    parser.add_argument('--dataset', type=str, default='miyawaki',
                       choices=['miyawaki', 'vangerven', 'mindbigdata', 'crell'],
                       help='Dataset to train on')
    parser.add_argument('--folds', type=int, default=10,
                       help='Number of CV folds')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computing device')
    
    args = parser.parse_args()
    
    print("🧠 Lightweight Brain-Diffuser CV-Compliant Training")
    print("=" * 50)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔄 Folds: {args.folds}")
    print(f"💻 Device: {args.device}")
    print(f"🎯 Academic Seed: {ACADEMIC_SEED}")
    print()
    
    # Train with CV framework
    results = train_brain_diffuser_cv(
        dataset_name=args.dataset,
        device=args.device,
        n_folds=args.folds
    )
    
    if results and results.get('academic_compliant', False):
        print(f"\n✅ Lightweight Brain-Diffuser CV Training Complete!")
        print(f"📊 CV Score: {results['cv_mean']:.6f} ± {results['cv_std']:.6f}")
        print(f"🎯 Academic Compliant: {results['academic_compliant']}")
        
        # Save results
        results_dir = Path("sota_comparison/brain_diffuser/results")
        results_dir.mkdir(exist_ok=True)
        
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"cv_results_{args.dataset}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved: {results_file}")
    else:
        print(f"\n❌ Lightweight Brain-Diffuser CV Training Failed!")


if __name__ == "__main__":
    main()
