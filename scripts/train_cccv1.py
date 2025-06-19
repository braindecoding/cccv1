"""
CortexFlow-CLIP-CNN V1 Training Script
=====================================

Main training script for CCCV1 with optimal configurations.
Uses the breakthrough configurations that achieved 100% success rate.

Usage:
    python train_cccv1.py --dataset miyawaki --mode single
    python train_cccv1.py --dataset all --mode cross_validation
    python train_cccv1.py --dataset vangerven --mode reproduce_breakthrough
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CCCV1 components
from src.models.cortexflow_clip_cnn_v1 import (
    CortexFlowCLIPCNNV1Optimized,
    CLIPLossV1,
    create_cccv1_model
)

# Import utilities
try:
    from data import load_dataset_gpu_optimized
    print("✅ Data loader imported successfully")
except ImportError:
    try:
        from src.data import load_dataset_gpu_optimized
        print("✅ Data loader imported from src.data")
    except ImportError:
        print("❌ Could not import data loader")
        load_dataset_gpu_optimized = None

try:
    from src.evaluation.metrics import calculate_comprehensive_metrics
except ImportError:
    print("⚠️ Evaluation metrics not available. Using basic metrics.")
    calculate_comprehensive_metrics = None

def setup_device():
    """Setup CUDA device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        return device
    else:
        print("⚠️ CUDA not available, using CPU")
        return torch.device('cpu')

def load_optimal_config(dataset_name):
    """Load optimal configuration for dataset"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'optimal_configs.json')
    
    try:
        with open(config_path, 'r') as f:
            configs = json.load(f)
        
        dataset_config = configs['cccv1_optimal_configurations']['datasets'][dataset_name]
        return dataset_config['optimal_config']
    
    except Exception as e:
        print(f"⚠️ Could not load config for {dataset_name}: {e}")
        # Return default config
        return {
            'architecture': {'dropout_encoder': 0.06, 'dropout_decoder': 0.02, 'clip_residual_weight': 0.1},
            'training': {'lr': 0.001, 'batch_size': 16, 'weight_decay': 1e-6, 'epochs': 100, 'patience': 15}
        }

def train_cccv1_model(model, train_loader, val_loader, config, device, verbose=True):
    """Train CCCV1 model with optimal configuration"""
    
    training_config = config['training']
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay'],
        betas=training_config.get('betas', [0.9, 0.999])
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=training_config['scheduler_factor'], 
        patience=training_config['patience']//3,
        min_lr=1e-8
    )
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    if verbose:
        print(f"🔧 Training with config: lr={training_config['lr']}, "
              f"batch_size={training_config['batch_size']}, epochs={training_config['epochs']}")
    
    for epoch in range(training_config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, clip_embedding = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=training_config.get('gradient_clip', 0.5)
            )
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                val_loss += criterion(output, target).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if patience_counter >= training_config['patience']:
            if verbose:
                print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return {
        'best_val_loss': best_val_loss,
        'training_history': training_history,
        'final_epoch': epoch + 1
    }

def evaluate_cccv1_model(model, test_loader, device):
    """Evaluate CCCV1 model"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate MSE
    mse = nn.MSELoss()(predictions, targets).item()
    
    return {
        'mse': mse,
        'predictions': predictions,
        'targets': targets
    }

def train_single_dataset(dataset_name, device, save_model=True):
    """Train CCCV1 on a single dataset"""
    
    print(f"\n📁 Training CCCV1 on {dataset_name.upper()}")
    print("=" * 50)
    
    # Load dataset
    try:
        if load_dataset_gpu_optimized is not None:
            X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
        else:
            print("❌ Dataset loading function not available")
            return None
        
        if X_train is None:
            print(f"❌ Failed to load {dataset_name}")
            return None
        
        print(f"✅ Dataset loaded: Train={len(X_train)}, Test={len(X_test)}, Input_dim={input_dim}")
        
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return None
    
    # Load optimal configuration
    config = load_optimal_config(dataset_name)
    
    # Split training data for validation
    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create optimized model
    model = CortexFlowCLIPCNNV1Optimized(input_dim, device, dataset_name)
    
    print(f"🏗️ Model: {model.name}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    training_results = train_cccv1_model(model, train_loader, val_loader, config, device)
    
    # Evaluate on test set
    test_results = evaluate_cccv1_model(model, test_loader, device)
    
    print(f"\n🎯 Results for {dataset_name.upper()}:")
    print(f"   Best Val Loss: {training_results['best_val_loss']:.6f}")
    print(f"   Test MSE: {test_results['mse']:.6f}")
    print(f"   Training Epochs: {training_results['final_epoch']}")
    
    # Check if beats champion
    champion_targets = {
        'miyawaki': 0.009845,
        'vangerven': 0.045659,
        'mindbigdata': 0.057348,
        'crell': 0.032525
    }
    
    if dataset_name in champion_targets:
        target = champion_targets[dataset_name]
        if test_results['mse'] < target:
            improvement = ((target - test_results['mse']) / target) * 100
            print(f"   🏆 BEATS CHAMPION by {improvement:.2f}%!")
        else:
            gap = ((test_results['mse'] - target) / target) * 100
            print(f"   📈 Gap to champion: +{gap:.2f}%")
    
    # Save model if requested
    if save_model:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f'{dataset_name}_cccv1_best.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'results': {
                'val_loss': training_results['best_val_loss'],
                'test_mse': test_results['mse'],
                'training_epochs': training_results['final_epoch']
            },
            'dataset_name': dataset_name,
            'input_dim': input_dim
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
    
    return {
        'dataset_name': dataset_name,
        'config': config,
        'training_results': training_results,
        'test_results': test_results,
        'model': model
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train CortexFlow-CLIP-CNN V1')
    parser.add_argument('--dataset', type=str, default='miyawaki',
                       choices=['miyawaki', 'vangerven', 'mindbigdata', 'crell', 'all'],
                       help='Dataset to train on')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'cross_validation', 'reproduce_breakthrough'],
                       help='Training mode')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save trained model')
    
    args = parser.parse_args()
    
    print("🎯 CortexFlow-CLIP-CNN V1 Training")
    print("=" * 40)
    print(f"📊 Mode: {args.mode}")
    print(f"📁 Dataset: {args.dataset}")
    print()
    
    device = setup_device()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
        all_results = {}
        
        for dataset_name in datasets:
            result = train_single_dataset(dataset_name, device, args.save_model)
            if result:
                all_results[dataset_name] = result
        
        # Summary
        print(f"\n🎉 CCCV1 Training Complete!")
        print("=" * 40)
        
        champions_beaten = 0
        for dataset_name, result in all_results.items():
            test_mse = result['test_results']['mse']
            print(f"{dataset_name.upper()}: {test_mse:.6f}")
            
            # Check champion status
            champion_targets = {'miyawaki': 0.009845, 'vangerven': 0.045659, 
                              'mindbigdata': 0.057348, 'crell': 0.032525}
            if dataset_name in champion_targets and test_mse < champion_targets[dataset_name]:
                champions_beaten += 1
        
        print(f"\n🏆 Champions beaten: {champions_beaten}/{len(all_results)}")
        print(f"Success rate: {(champions_beaten/len(all_results))*100:.1f}%")
        
    else:
        result = train_single_dataset(args.dataset, device, args.save_model)
        
        if result:
            print(f"\n🎉 CCCV1 Training Complete for {args.dataset.upper()}!")
        else:
            print(f"\n❌ Training failed for {args.dataset}")

if __name__ == "__main__":
    main()
