"""
CCCV1 Baseline Training Script
=============================

Standard training procedure without optimizations for fair comparison.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import argparse

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from model import CCCV1Baseline, get_baseline_config
from data.loader import load_dataset


def train_baseline_model(dataset_name, device='cuda', save_model=True):
    """Train CCCV1 baseline model on specified dataset"""
    
    print(f"🎯 CCCV1 BASELINE TRAINING")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Load dataset
    print(f"📊 Loading {dataset_name} dataset...")
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, device=device)
    
    input_dim = X_train.shape[1]
    print(f"✅ Dataset loaded: {X_train.shape[0]} samples, {input_dim} features")
    
    # Get baseline configuration
    config = get_baseline_config()
    
    # Create model
    model = CCCV1Baseline(input_dim, device)
    print(f"✅ Model created: {model.get_parameter_count():,} parameters")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        betas=config['training']['betas']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler_factor'],
        patience=config['training']['patience'] // 3,
        min_lr=1e-8
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Training loop
    print("🚀 Starting training...")
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['training']['gradient_clip']
            )
            
            optimizer.step()
            train_loss += loss.item()
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                test_loss += criterion(output, target).item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            
            if save_model:
                # Save best model
                model_path = f"models/{dataset_name}_baseline_best.pth"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 10 == 0 or patience_counter == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}, Best: {best_test_loss:.6f}")
        
        # Early stopping
        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"✅ Training completed!")
    print(f"Best test loss: {best_test_loss:.6f}")
    
    # Save metadata
    if save_model:
        metadata = {
            'dataset_name': dataset_name,
            'input_dim': input_dim,
            'best_test_loss': best_test_loss,
            'config': config,
            'model_architecture': 'CCCV1Baseline',
            'save_timestamp': datetime.now().isoformat(),
            'device': str(device)
        }
        
        metadata_path = f"models/{dataset_name}_baseline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved: {model_path}")
        print(f"💾 Metadata saved: {metadata_path}")
    
    return model, best_test_loss


def main():
    parser = argparse.ArgumentParser(description='Train CCCV1 Baseline Model')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['miyawaki', 'vangerven', 'crell', 'mindbigdata', 'all'],
                        help='Dataset to train on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save trained model')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    save_model = not args.no_save
    
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
        results = {}
        
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"Training on {dataset.upper()}")
            print(f"{'='*60}")
            
            try:
                model, best_loss = train_baseline_model(dataset, device, save_model)
                results[dataset] = best_loss
                print(f"✅ {dataset}: {best_loss:.6f}")
            except Exception as e:
                print(f"❌ {dataset}: {str(e)}")
                results[dataset] = None
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        for dataset, loss in results.items():
            if loss is not None:
                print(f"{dataset:12}: {loss:.6f}")
            else:
                print(f"{dataset:12}: FAILED")
    
    else:
        train_baseline_model(args.dataset, device, save_model)


if __name__ == "__main__":
    main()
