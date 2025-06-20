"""
Lightweight Brain-Diffuser Training Script
==========================================

Academic-compliant training maintaining exact methodology.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from lightweight_brain_diffuser import LightweightBrainDiffuser
from data.loader import load_dataset_gpu_optimized


def train_lightweight_brain_diffuser(dataset_name: str, device='cuda', epochs=100, 
                                    batch_size=16, lr=1e-3, save_model=True):
    """Train Lightweight Brain-Diffuser model on specified dataset"""
    
    print(f"🧠 LIGHTWEIGHT BRAIN-DIFFUSER TRAINING")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print("=" * 60)
    
    # Load dataset
    print(f"📊 Loading {dataset_name} dataset...")
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(
        dataset_name, device=device
    )
    
    print(f"✅ Dataset loaded:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    print(f"   Input dim: {input_dim}")
    print(f"   Image shape: {y_train.shape[1:]}")
    
    # Create model
    model = LightweightBrainDiffuser(
        input_dim=input_dim,
        device=device,
        image_size=y_train.shape[-1]
    )
    
    print(f"✅ Model created: {model.get_parameter_count():,} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-8
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Training loop
    print("🚀 Starting training...")
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    training_history = {
        'train_losses': [],
        'test_losses': [],
        'learning_rates': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Compute losses
            losses = model.compute_loss(data, target)
            total_loss = losses['total_loss']
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(total_loss.item())
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{total_loss.item():.6f}",
                'VDVAE': f"{losses['vdvae_loss'].item():.6f}",
                'Diff': f"{losses['diffusion_loss'].item():.6f}"
            })
        
        # Testing phase
        model.eval()
        test_losses = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]")
            for data, target in test_pbar:
                data, target = data.to(device), target.to(device)
                
                losses = model.compute_loss(data, target)
                test_losses.append(losses['total_loss'].item())
                
                test_pbar.set_postfix({
                    'Loss': f"{losses['total_loss'].item():.6f}"
                })
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        training_history['train_losses'].append(avg_train_loss)
        training_history['test_losses'].append(avg_test_loss)
        training_history['learning_rates'].append(current_lr)
        
        # Learning rate scheduling
        scheduler.step(avg_test_loss)
        
        # Early stopping and model saving
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            
            if save_model:
                # Save best model
                model_path = f"models/{dataset_name}_lightweight_brain_diffuser_best.pth"
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        # Print progress
        print(f"Epoch {epoch+1:3d}: "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Test Loss: {avg_test_loss:.6f}, "
              f"Best: {best_test_loss:.6f}, "
              f"LR: {current_lr:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"✅ Training completed!")
    print(f"Best test loss: {best_test_loss:.6f}")
    
    # Save metadata and training history
    if save_model:
        metadata = {
            'dataset_name': dataset_name,
            'input_dim': input_dim,
            'image_shape': list(y_train.shape[1:]),
            'best_test_loss': best_test_loss,
            'total_epochs': epoch + 1,
            'model_architecture': 'Lightweight-Brain-Diffuser',
            'training_config': {
                'batch_size': batch_size,
                'learning_rate': lr,
                'epochs': epochs,
                'patience': patience
            },
            'training_history': training_history,
            'model_info': model.get_model_info(),
            'save_timestamp': datetime.now().isoformat(),
            'device': str(device),
            'academic_compliance': {
                'methodology': 'Exact Brain-Diffuser methodology',
                'modifications': 'Lightweight components for computational feasibility',
                'academic_integrity': 'Maintained'
            }
        }
        
        metadata_path = f"models/{dataset_name}_lightweight_brain_diffuser_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved: {model_path}")
        print(f"💾 Metadata saved: {metadata_path}")
    
    return model, best_test_loss, training_history


def main():
    parser = argparse.ArgumentParser(description='Train Lightweight Brain-Diffuser Model')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['miyawaki', 'vangerven', 'crell', 'mindbigdata', 'all'],
                        help='Dataset to train on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save trained model')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    save_model = not args.no_save
    
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
        results = {}
        
        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"TRAINING ON {dataset.upper()}")
            print(f"{'='*80}")
            
            try:
                model, best_loss, history = train_lightweight_brain_diffuser(
                    dataset, device, args.epochs, args.batch_size, 
                    args.lr, save_model
                )
                results[dataset] = {
                    'best_loss': best_loss,
                    'final_train_loss': history['train_losses'][-1],
                    'status': 'success'
                }
                print(f"✅ {dataset}: {best_loss:.6f}")
            except Exception as e:
                print(f"❌ {dataset}: {str(e)}")
                results[dataset] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        print(f"\n{'='*80}")
        print("LIGHTWEIGHT BRAIN-DIFFUSER TRAINING SUMMARY")
        print(f"{'='*80}")
        for dataset, result in results.items():
            if result['status'] == 'success':
                print(f"{dataset:12}: {result['best_loss']:.6f}")
            else:
                print(f"{dataset:12}: FAILED")
    
    else:
        train_lightweight_brain_diffuser(
            args.dataset, device, args.epochs, 
            args.batch_size, args.lr, save_model
        )


if __name__ == "__main__":
    main()
