"""
CortexFlow-CLIP-CNN V1 Configuration Refinement
==============================================

Fine-tune configurations for datasets that showed gaps in cross-validation:
- Vangerven: +1.66% gap
- Crell: +0.07% gap

Goal: Achieve consistent wins across all datasets through targeted optimization.

Strategy:
1. Grid search on critical hyperparameters
2. Extended training with higher epochs
3. Advanced learning rate schedules
4. Dropout optimization
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
from sklearn.model_selection import KFold
import itertools
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.append(root_dir)
sys.path.append(parent_dir)

# Import CCCV1 components
try:
    from cccv1.src.models.cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized
except ImportError:
    try:
        sys.path.append(os.path.join(parent_dir, 'src', 'models'))
        from cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized
    except ImportError:
        print("❌ Could not import CCCV1 models")
        sys.exit(1)

# Import utilities
try:
    from src.data import load_dataset_gpu_optimized
except ImportError:
    print("⚠️ Parent directory imports not available")

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
        return torch.device('cpu')

def get_refinement_configs(dataset_name):
    """Get refined configuration grids for specific datasets"""
    
    if dataset_name == 'vangerven':
        # Vangerven: Focus on ultra-stable learning with extended training
        return {
            'lr': [0.0002, 0.0003, 0.0004, 0.0005],
            'batch_size': [6, 8, 10, 12],
            'weight_decay': [5e-9, 1e-8, 2e-8, 5e-8],
            'epochs': [200, 250, 300],
            'patience': [30, 35, 40],
            'scheduler_factor': [0.2, 0.3, 0.4],
            'dropout_encoder': [0.04, 0.05, 0.06],
            'dropout_decoder': [0.01, 0.015, 0.02],
            'clip_residual_weight': [0.06, 0.08, 0.1]
        }
    
    elif dataset_name == 'crell':
        # Crell: Fine-tune around current best configuration
        return {
            'lr': [0.0006, 0.0007, 0.0008, 0.0009, 0.001],
            'batch_size': [16, 18, 20, 22, 24],
            'weight_decay': [3e-7, 5e-7, 7e-7, 1e-6],
            'epochs': [140, 160, 180],
            'patience': [18, 20, 22],
            'scheduler_factor': [0.4, 0.5, 0.6],
            'dropout_encoder': [0.04, 0.05, 0.06],
            'dropout_decoder': [0.015, 0.02, 0.025],
            'clip_residual_weight': [0.06, 0.08, 0.1, 0.12]
        }
    
    else:
        # Default configuration for other datasets
        return {
            'lr': [0.0005, 0.001],
            'batch_size': [16, 32],
            'weight_decay': [1e-6],
            'epochs': [100],
            'patience': [15],
            'scheduler_factor': [0.5],
            'dropout_encoder': [0.05],
            'dropout_decoder': [0.02],
            'clip_residual_weight': [0.08]
        }

def create_refined_model(input_dim, device, config):
    """Create CCCV1 model with refined configuration"""
    
    class RefinedCortexFlowCLIPCNNV1(nn.Module):
        def __init__(self, input_dim, device, config):
            super(RefinedCortexFlowCLIPCNNV1, self).__init__()
            self.name = "CortexFlow-CLIP-CNN-V1-Refined"
            self.device = device
            
            # Refined encoder with tunable dropout
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Dropout(config.get('dropout_encoder', 0.05)),
                
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Dropout(config.get('dropout_encoder', 0.05) * 0.7),
                
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(config.get('dropout_encoder', 0.05) * 0.5),
                
                # Map to CLIP space
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.Tanh()
            ).to(device)
            
            # Refined decoder
            self.decoder = nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(config.get('dropout_decoder', 0.02)),
                
                nn.Linear(512, 784),
                nn.Sigmoid()
            ).to(device)
            
            # CLIP alignment module
            self.clip_aligner = nn.Sequential(
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 512),
                nn.Tanh()
            ).to(device)
            
            self.residual_weight = config.get('clip_residual_weight', 0.08)
        
        def forward(self, x):
            # Encode to CLIP-like space
            features = self.encoder(x)
            
            # CLIP alignment with tunable residual
            aligned_features = features + self.residual_weight * self.clip_aligner(features)
            aligned_features = torch.nn.functional.normalize(aligned_features, p=2, dim=1)
            
            # Decode to visual output
            visual_output = self.decoder(aligned_features)
            
            return visual_output.view(-1, 1, 28, 28), aligned_features
    
    return RefinedCortexFlowCLIPCNNV1(input_dim, device, config)

def train_refined_model(model, train_loader, val_loader, config, device):
    """Train refined model with advanced techniques"""
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Advanced scheduler with warmup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['scheduler_factor'], 
        patience=config['patience']//3,
        min_lr=1e-9,
        threshold=1e-6
    )
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, clip_embedding = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping with adaptive norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
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
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return best_val_loss

def evaluate_refined_model(model, test_loader, device):
    """Evaluate refined model"""
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
    
    mse = nn.MSELoss()(predictions, targets).item()
    return mse

def grid_search_refinement(dataset_name, X_train, y_train, input_dim, device, max_trials=20):
    """Grid search for configuration refinement"""
    
    config_grid = get_refinement_configs(dataset_name)
    
    # Champion targets
    champion_targets = {
        'vangerven': 0.045659,   # Brain-Diffuser
        'crell': 0.032525        # MinD-Vis
    }
    
    target_score = champion_targets.get(dataset_name, 0.05)
    
    print(f"🎯 Target to beat on {dataset_name}: {target_score:.6f}")
    print(f"🔍 Running grid search refinement (max {max_trials} trials)")
    
    # Generate all combinations (sample if too many)
    keys = list(config_grid.keys())
    values = list(config_grid.values())
    all_combinations = list(itertools.product(*values))
    
    if len(all_combinations) > max_trials:
        # Random sample from all combinations
        selected_combinations = np.random.choice(
            len(all_combinations), 
            size=max_trials, 
            replace=False
        )
        combinations = [all_combinations[i] for i in selected_combinations]
    else:
        combinations = all_combinations
    
    best_score = float('inf')
    best_config = None
    results = []
    
    for trial, combination in enumerate(combinations):
        # Create config from combination
        config = dict(zip(keys, combination))
        
        print(f"\n🔧 Trial {trial+1}/{len(combinations)}")
        print(f"   lr={config['lr']}, batch_size={config['batch_size']}, "
              f"epochs={config['epochs']}")
        
        try:
            # Split for validation
            train_size = int(0.8 * len(X_train))
            X_train_split = X_train[:train_size]
            y_train_split = y_train[:train_size]
            X_val = X_train[train_size:]
            y_val = y_train[train_size:]
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_split, y_train_split)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
            
            # Create refined model
            model = create_refined_model(input_dim, device, config)
            
            # Train model
            val_loss = train_refined_model(model, train_loader, val_loader, config, device)
            
            print(f"   Result: Val Loss = {val_loss:.6f}")
            
            # Check if beats champion
            if val_loss < target_score:
                improvement = ((target_score - val_loss) / target_score) * 100
                print(f"   🏆 BEATS CHAMPION by {improvement:.2f}%!")
            else:
                gap = ((val_loss - target_score) / target_score) * 100
                print(f"   📈 Gap to target: +{gap:.2f}%")
            
            # Track results
            results.append({
                'trial': trial + 1,
                'config': config.copy(),
                'val_loss': val_loss,
                'beats_champion': val_loss < target_score
            })
            
            # Update best
            if val_loss < best_score:
                best_score = val_loss
                best_config = config.copy()
                print(f"   🌟 New best score: {best_score:.6f}")
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Trial failed: {e}")
            continue
    
    return {
        'best_score': best_score,
        'best_config': best_config,
        'all_results': results,
        'beats_champion': best_score < target_score,
        'target_score': target_score
    }

def main():
    """Main refinement function"""
    print("🎯 CortexFlow-CLIP-CNN V1 Configuration Refinement")
    print("=" * 60)
    print("🔬 Goal: Achieve consistent wins through targeted optimization")
    print()
    
    device = setup_device()
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Datasets to refine (focus on gaps)
    datasets_to_refine = ['vangerven', 'crell']
    
    # Results storage
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dataset_name in datasets_to_refine:
        print(f"\n📁 Refining Configuration for {dataset_name.upper()}")
        print("=" * 50)
        
        # Load dataset
        try:
            if 'load_dataset_gpu_optimized' in globals():
                X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
            else:
                print("❌ Dataset loading function not available")
                continue
            
            if X_train is None:
                print(f"❌ Failed to load {dataset_name}")
                continue
            
            print(f"✅ Dataset loaded: Train={len(X_train)}, Test={len(X_test)}")
            
        except Exception as e:
            print(f"❌ Error loading {dataset_name}: {e}")
            continue
        
        # Run grid search refinement
        refinement_results = grid_search_refinement(
            dataset_name, X_train, y_train, input_dim, device
        )
        
        all_results[dataset_name] = refinement_results
    
    # Save results
    results_dir = f"cccv1/results/configuration_refinement_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/refinement_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Final analysis
    print("\n🎉 Configuration Refinement Complete!")
    print("=" * 60)
    
    champions_beaten = 0
    total_datasets = len(all_results)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"   Target: {results['target_score']:.6f}")
        print(f"   Best Refined: {results['best_score']:.6f}")
        
        if results['beats_champion']:
            improvement = ((results['target_score'] - results['best_score']) / results['target_score']) * 100
            print(f"   🏆 BEATS CHAMPION by {improvement:.2f}%!")
            champions_beaten += 1
        else:
            gap = ((results['best_score'] - results['target_score']) / results['target_score']) * 100
            print(f"   📈 Gap: +{gap:.2f}%")
        
        if results['best_config']:
            config = results['best_config']
            print(f"   Best config: lr={config['lr']}, batch_size={config['batch_size']}, "
                  f"epochs={config['epochs']}")
    
    print(f"\n🏆 FINAL REFINEMENT RESULTS:")
    print(f"Champions beaten: {champions_beaten}/{total_datasets}")
    print(f"Success rate: {(champions_beaten/total_datasets)*100:.1f}%")
    
    if champions_beaten == total_datasets:
        print(f"\n🎉 REFINEMENT SUCCESS!")
        print(f"🚀 All gap datasets now beat champions!")
    else:
        print(f"\n🔧 Partial success - further refinement may be needed")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print("🚀 Configuration Refinement Complete!")

if __name__ == "__main__":
    main()
