"""
CortexFlow-CLIP-CNN V1 Enhanced Validation
==========================================

Enhanced validation with refined configurations and increased statistical power:
1. 10-fold cross-validation for better statistical power
2. Use refined configurations from optimization
3. Multiple runs with different seeds
4. Comprehensive statistical analysis

Goal: Validate refined breakthrough with rigorous statistical testing.
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
from scipy import stats
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
    from data import load_dataset_gpu_optimized
    print("✅ Data loader imported successfully")
except ImportError:
    try:
        from src.data import load_dataset_gpu_optimized
        print("✅ Data loader imported from src.data")
    except ImportError:
        print("❌ Could not import data loader")
        load_dataset_gpu_optimized = None

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

def get_refined_configs():
    """Get refined configurations from optimization results"""
    return {
        'miyawaki': {
            'architecture': {'dropout_encoder': 0.06, 'dropout_decoder': 0.02, 'clip_residual_weight': 0.1},
            'training': {'lr': 0.0003, 'batch_size': 8, 'weight_decay': 1e-8, 'epochs': 200, 'patience': 25, 'scheduler_factor': 0.3}
        },
        'vangerven': {
            'architecture': {'dropout_encoder': 0.05, 'dropout_decoder': 0.015, 'clip_residual_weight': 0.08},
            'training': {'lr': 0.0003, 'batch_size': 8, 'weight_decay': 1e-8, 'epochs': 250, 'patience': 35, 'scheduler_factor': 0.3}
        },
        'mindbigdata': {
            'architecture': {'dropout_encoder': 0.04, 'dropout_decoder': 0.02, 'clip_residual_weight': 0.05},
            'training': {'lr': 0.001, 'batch_size': 32, 'weight_decay': 1e-6, 'epochs': 100, 'patience': 12, 'scheduler_factor': 0.5}
        },
        'crell': {
            'architecture': {'dropout_encoder': 0.05, 'dropout_decoder': 0.02, 'clip_residual_weight': 0.08},
            'training': {'lr': 0.0007, 'batch_size': 16, 'weight_decay': 5e-7, 'epochs': 160, 'patience': 20, 'scheduler_factor': 0.5}
        }
    }

def create_enhanced_model(input_dim, device, config):
    """Create enhanced CCCV1 model with refined configuration"""
    
    class EnhancedCortexFlowCLIPCNNV1(nn.Module):
        def __init__(self, input_dim, device, config):
            super(EnhancedCortexFlowCLIPCNNV1, self).__init__()
            self.name = "CortexFlow-CLIP-CNN-V1-Enhanced"
            self.device = device
            
            arch_config = config.get('architecture', {})
            
            # Enhanced encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Dropout(arch_config.get('dropout_encoder', 0.05)),
                
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.SiLU(),
                nn.Dropout(arch_config.get('dropout_encoder', 0.05) * 0.7),
                
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(arch_config.get('dropout_encoder', 0.05) * 0.5),
                
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.Tanh()
            ).to(device)
            
            # Enhanced decoder
            self.decoder = nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(arch_config.get('dropout_decoder', 0.02)),
                
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
            
            self.residual_weight = arch_config.get('clip_residual_weight', 0.08)
        
        def forward(self, x):
            features = self.encoder(x)
            aligned_features = features + self.residual_weight * self.clip_aligner(features)
            aligned_features = torch.nn.functional.normalize(aligned_features, p=2, dim=1)
            visual_output = self.decoder(aligned_features)
            return visual_output.view(-1, 1, 28, 28), aligned_features
    
    return EnhancedCortexFlowCLIPCNNV1(input_dim, device, config)

def train_enhanced_model(model, train_loader, val_loader, config, device):
    """Train enhanced model with refined configuration"""
    
    training_config = config['training']
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=training_config['scheduler_factor'], 
        patience=training_config['patience']//3,
        min_lr=1e-9
    )
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    
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
        
        if patience_counter >= training_config['patience']:
            break
    
    model.load_state_dict(best_model_state)
    return best_val_loss

def evaluate_enhanced_model(model, test_loader, device):
    """Evaluate enhanced model"""
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

def enhanced_cross_validation(dataset_name, X_train, y_train, input_dim, device, n_folds=10, n_runs=3):
    """Enhanced cross-validation with multiple runs"""
    
    print(f"🔄 Enhanced {n_folds}-fold CV with {n_runs} runs: {dataset_name.upper()}")
    
    config = get_refined_configs()[dataset_name]
    all_scores = []
    
    for run in range(n_runs):
        print(f"   Run {run+1}/{n_runs}:")
        
        # Set different seed for each run
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)
        
        # Setup cross-validation
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42 + run)
        run_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            print(f"     Fold {fold+1}/{n_folds}...", end=" ")
            
            # Split data for this fold
            X_train_fold = X_train[train_idx]
            y_train_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            # Create data loaders
            batch_size = config['training']['batch_size']
            train_dataset = TensorDataset(X_train_fold, y_train_fold)
            val_dataset = TensorDataset(X_val_fold, y_val_fold)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            model = create_enhanced_model(input_dim, device, config)
            
            try:
                # Train model
                train_enhanced_model(model, train_loader, val_loader, config, device)
                
                # Evaluate model
                fold_score = evaluate_enhanced_model(model, val_loader, device)
                run_scores.append(fold_score)
                
                print(f"MSE: {fold_score:.6f}")
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
        
        if run_scores:
            run_scores = np.array(run_scores)
            all_scores.extend(run_scores)
            print(f"     Run {run+1} Results: {run_scores.mean():.6f} ± {run_scores.std():.6f}")
    
    if all_scores:
        all_scores = np.array(all_scores)
        print(f"   📊 Enhanced CV Results: {all_scores.mean():.6f} ± {all_scores.std():.6f}")
        return all_scores
    else:
        print(f"   ❌ No successful folds")
        return np.array([])

def comprehensive_statistical_analysis(cccv1_scores, champion_score, dataset_name):
    """Comprehensive statistical analysis"""
    
    if len(cccv1_scores) == 0:
        return None
    
    # One-sample t-test against champion
    t_stat, p_value = stats.ttest_1samp(cccv1_scores, champion_score)
    
    # Effect size (Cohen's d)
    cohens_d = (champion_score - cccv1_scores.mean()) / cccv1_scores.std()
    
    # Confidence interval
    confidence_interval = stats.t.interval(
        0.95, len(cccv1_scores)-1, 
        loc=cccv1_scores.mean(), 
        scale=stats.sem(cccv1_scores)
    )
    
    # Win rate
    wins = np.sum(cccv1_scores < champion_score)
    win_rate = wins / len(cccv1_scores)
    
    return {
        'test_type': 'One-sample t-test',
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'confidence_interval': confidence_interval,
        'win_rate': win_rate,
        'wins': wins,
        'total_folds': len(cccv1_scores),
        'cccv1_mean': cccv1_scores.mean(),
        'cccv1_std': cccv1_scores.std(),
        'champion_score': champion_score,
        'significant': p_value < 0.05
    }

def main():
    """Main enhanced validation function"""
    print("🎯 CortexFlow-CLIP-CNN V1 Enhanced Validation")
    print("=" * 60)
    print("🔬 Enhanced validation with refined configurations")
    print("📊 10-fold CV with multiple runs for statistical power")
    print()
    
    device = setup_device()
    
    # Champion targets
    champion_targets = {
        'miyawaki': 0.009845,    # Brain-Diffuser
        'vangerven': 0.045659,   # Brain-Diffuser  
        'mindbigdata': 0.057348, # MinD-Vis
        'crell': 0.032525        # MinD-Vis
    }
    
    champion_methods = {
        'miyawaki': 'Brain-Diffuser',
        'vangerven': 'Brain-Diffuser',
        'mindbigdata': 'MinD-Vis',
        'crell': 'MinD-Vis'
    }
    
    # Datasets to validate
    datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    
    # Results storage
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for dataset_name in datasets:
        print(f"\n📁 Enhanced Validation for {dataset_name.upper()}")
        print("=" * 50)
        
        # Load dataset
        try:
            if load_dataset_gpu_optimized is not None:
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
        
        # Enhanced cross-validation
        cccv1_scores = enhanced_cross_validation(
            dataset_name, X_train, y_train, input_dim, device, n_folds=10, n_runs=3
        )
        
        if len(cccv1_scores) == 0:
            print(f"❌ Enhanced validation failed for {dataset_name}")
            continue
        
        # Statistical analysis
        champion_score = champion_targets[dataset_name]
        champion_method = champion_methods[dataset_name]
        
        stats_result = comprehensive_statistical_analysis(cccv1_scores, champion_score, dataset_name)
        
        print(f"\n🎯 Comparison with {champion_method}:")
        print(f"   Champion: {champion_score:.6f}")
        print(f"   CCCV1 Enhanced: {stats_result['cccv1_mean']:.6f} ± {stats_result['cccv1_std']:.6f}")
        print(f"   95% CI: [{stats_result['confidence_interval'][0]:.6f}, {stats_result['confidence_interval'][1]:.6f}]")
        
        if stats_result['cccv1_mean'] < champion_score:
            improvement = ((champion_score - stats_result['cccv1_mean']) / champion_score) * 100
            print(f"   🏆 CCCV1 WINS by {improvement:.2f}%!")
        else:
            gap = ((stats_result['cccv1_mean'] - champion_score) / champion_score) * 100
            print(f"   📈 Gap: +{gap:.2f}%")
        
        print(f"   📊 Win Rate: {stats_result['wins']}/{stats_result['total_folds']} ({stats_result['win_rate']*100:.1f}%)")
        
        print(f"\n📊 Statistical Analysis:")
        print(f"   Test: {stats_result['test_type']}")
        print(f"   p-value: {stats_result['p_value']:.6f}")
        print(f"   Cohen's d: {stats_result['cohens_d']:.3f}")
        
        if stats_result['significant']:
            print(f"   ✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
        else:
            print(f"   ⚠️ Not statistically significant (p ≥ 0.05)")
        
        all_results[dataset_name] = {
            'cccv1_scores': cccv1_scores,
            'champion_target': champion_score,
            'champion_method': champion_method,
            'statistical_result': stats_result
        }
    
    # Save results
    results_dir = f"cccv1/results/enhanced_validation_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/enhanced_validation_results.json", 'w') as f:
        json.dump({k: {
            'cccv1_scores': v['cccv1_scores'].tolist(),
            'champion_target': v['champion_target'],
            'champion_method': v['champion_method'],
            'statistical_result': v['statistical_result']
        } for k, v in all_results.items()}, f, indent=2, default=str)
    
    # Final summary
    print("\n🎉 Enhanced Validation Complete!")
    print("=" * 60)
    
    consistent_wins = 0
    significant_wins = 0
    total_datasets = len(all_results)
    
    for dataset_name, result in all_results.items():
        stats_result = result['statistical_result']
        cccv1_mean = stats_result['cccv1_mean']
        champion_target = result['champion_target']
        
        print(f"\n{dataset_name.upper()}:")
        print(f"   CCCV1: {cccv1_mean:.6f} ± {stats_result['cccv1_std']:.6f}")
        print(f"   Champion: {champion_target:.6f}")
        print(f"   Win Rate: {stats_result['win_rate']*100:.1f}%")
        print(f"   p-value: {stats_result['p_value']:.6f}")
        
        if cccv1_mean < champion_target:
            improvement = ((champion_target - cccv1_mean) / champion_target) * 100
            print(f"   🏆 WINS by {improvement:.2f}%")
            consistent_wins += 1
        else:
            gap = ((cccv1_mean - champion_target) / champion_target) * 100
            print(f"   📈 Gap: +{gap:.2f}%")
        
        if stats_result['significant']:
            print(f"   ✅ Statistically significant")
            significant_wins += 1
        else:
            print(f"   ⚠️ Not significant")
    
    print(f"\n🏆 FINAL ENHANCED VALIDATION RESULTS:")
    print(f"Consistent wins: {consistent_wins}/{total_datasets}")
    print(f"Statistically significant: {significant_wins}/{total_datasets}")
    print(f"Success rate: {(consistent_wins/total_datasets)*100:.1f}%")
    print(f"Statistical power: {(significant_wins/total_datasets)*100:.1f}%")
    
    if consistent_wins >= total_datasets * 0.75:
        print(f"\n🎉 ENHANCED VALIDATION SUCCESS!")
        print(f"🚀 CCCV1 demonstrates strong performance with refined configurations!")
    
    print(f"\n💾 Results saved to: {results_dir}/")
    print("🚀 Enhanced Validation Complete!")

if __name__ == "__main__":
    main()
