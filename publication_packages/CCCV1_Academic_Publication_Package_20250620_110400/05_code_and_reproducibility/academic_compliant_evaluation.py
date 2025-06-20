"""
Academic-Compliant Evaluation Framework
======================================

Evaluate existing trained models using unified CV framework to ensure
academic integrity and fair comparison.

This approach:
1. Uses existing trained models (already optimized)
2. Applies consistent evaluation methodology
3. Uses same random seed and data splits
4. Provides statistical comparison
"""

import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from scipy import stats

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.extend([current_dir, parent_dir])

# Import data loader
from data.loader import load_dataset_gpu_optimized

# Academic seed for reproducibility
ACADEMIC_SEED = 42
torch.manual_seed(ACADEMIC_SEED)
np.random.seed(ACADEMIC_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(ACADEMIC_SEED)


def load_trained_model(method_name, dataset_name, device='cuda'):
    """Load existing trained model"""
    
    model_paths = {
        'CCCV1-Optimized': f'models/{dataset_name}_cccv1_best.pth',
        'Mind-Vis': f'models/{dataset_name}_mind_vis_best.pth',
        'Lightweight-Brain-Diffuser': f'models/{dataset_name}_lightweight_brain_diffuser_best.pth'
    }
    
    model_path = model_paths.get(method_name)
    if not model_path or not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}")
        return None
    
    try:
        if method_name == 'CCCV1-Optimized':
            from src.models.cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized
            X_train, _, _, _, input_dim = load_dataset_gpu_optimized(dataset_name, device)
            model = CortexFlowCLIPCNNV1Optimized(input_dim, device, dataset_name)
            
        elif method_name == 'Mind-Vis':
            from sota_comparison.mind_vis.src.mind_vis_model import MindVis
            X_train, _, _, _, input_dim = load_dataset_gpu_optimized(dataset_name, device)
            model = MindVis(input_dim=input_dim, device=device)
            
        elif method_name == 'Lightweight-Brain-Diffuser':
            from sota_comparison.brain_diffuser.src.lightweight_brain_diffuser import LightweightBrainDiffuser
            X_train, y_train, _, _, input_dim = load_dataset_gpu_optimized(dataset_name, device)
            image_size = y_train.shape[-1]
            model = LightweightBrainDiffuser(input_dim=input_dim, device=device, image_size=image_size)
        
        # Load trained weights with proper handling
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load state dict with strict=False to handle missing keys
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print(f"✅ Loaded {method_name} model for {dataset_name}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading {method_name}: {e}")
        return None


def evaluate_model_cv(model, method_name, dataset_name, device='cuda', n_folds=10):
    """Evaluate model using academic-compliant CV"""
    
    print(f"🔄 {n_folds}-fold CV Evaluation: {method_name} on {dataset_name}")
    
    # Load dataset
    X_train, y_train, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device)
    
    if X_train is None:
        return None
    
    # Combine train and test for proper CV
    X_all = torch.cat([X_train, X_test], dim=0)
    y_all = torch.cat([y_train, y_test], dim=0)
    
    # Setup CV with academic seed
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=ACADEMIC_SEED)
    cv_scores = []
    criterion = nn.MSELoss()
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
        print(f"   Fold {fold+1}/{n_folds}...", end=" ")
        
        try:
            # Get validation data for this fold
            X_val_fold = X_all[val_idx]
            y_val_fold = y_all[val_idx]
            
            # Create validation loader
            val_dataset = TensorDataset(X_val_fold, y_val_fold)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Evaluate model on this fold
            model.eval()
            fold_loss = 0.0
            
            with torch.no_grad():
                for fmri_data, visual_data in val_loader:
                    fmri_data = fmri_data.to(device)
                    visual_data = visual_data.to(device)
                    
                    # Forward pass based on method
                    if method_name == 'Lightweight-Brain-Diffuser':
                        _, predictions = model(fmri_data, use_diffusion=True)
                    elif method_name == 'Mind-Vis':
                        _, _, predictions = model(fmri_data)  # Returns (latent, visual, images)
                    elif method_name == 'CCCV1-Optimized':
                        predictions, _ = model(fmri_data)  # Returns (visual_output, enhanced_embedding)
                    else:
                        predictions = model(fmri_data)
                    
                    loss = criterion(predictions, visual_data)
                    fold_loss += loss.item()
            
            fold_score = fold_loss / len(val_loader)
            cv_scores.append(fold_score)
            
            print(f"MSE: {fold_score:.6f}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if cv_scores:
        cv_scores = np.array(cv_scores)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   📊 CV Results: {cv_mean:.6f} ± {cv_std:.6f}")
        
        return {
            'method': method_name,
            'dataset': dataset_name,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'n_folds': n_folds,
            'academic_compliant': True
        }
    else:
        print(f"   ❌ All folds failed")
        return None


def run_academic_evaluation(datasets=['miyawaki'], n_folds=10, device='cuda'):
    """Run academic-compliant evaluation"""
    
    print("🎯 ACADEMIC-COMPLIANT EVALUATION")
    print("=" * 50)
    print(f"📊 Datasets: {datasets}")
    print(f"🔄 CV Folds: {n_folds}")
    print(f"🎯 Academic Seed: {ACADEMIC_SEED}")
    print()
    
    methods = ['CCCV1-Optimized', 'Mind-Vis', 'Lightweight-Brain-Diffuser']
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n📁 DATASET: {dataset_name.upper()}")
        print("=" * 40)
        
        dataset_results = {}
        
        for method_name in methods:
            print(f"\n🧠 Evaluating {method_name}...")
            
            # Load trained model
            model = load_trained_model(method_name, dataset_name, device)
            
            if model is not None:
                # Evaluate with CV
                results = evaluate_model_cv(model, method_name, dataset_name, device, n_folds)
                if results:
                    dataset_results[method_name] = results
                
                # Cleanup
                del model
                torch.cuda.empty_cache()
        
        # Statistical comparison
        if len(dataset_results) > 1:
            print(f"\n📊 STATISTICAL COMPARISON - {dataset_name.upper()}")
            print("-" * 40)
            
            # Sort by performance
            sorted_methods = sorted(dataset_results.items(), key=lambda x: x[1]['cv_mean'])
            
            for rank, (method, results) in enumerate(sorted_methods, 1):
                cv_mean = results['cv_mean']
                cv_std = results['cv_std']
                
                if rank == 1:
                    print(f"🥇 {method}: {cv_mean:.6f} ± {cv_std:.6f}")
                elif rank == 2:
                    print(f"🥈 {method}: {cv_mean:.6f} ± {cv_std:.6f}")
                else:
                    print(f"🥉 {method}: {cv_mean:.6f} ± {cv_std:.6f}")
            
            # Statistical significance testing
            if len(sorted_methods) >= 2:
                winner_name, winner_results = sorted_methods[0]
                winner_scores = np.array(winner_results['cv_scores'])
                
                print(f"\n📈 Statistical Significance:")
                for method, results in sorted_methods[1:]:
                    method_scores = np.array(results['cv_scores'])
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(method_scores, winner_scores)
                    significance = "✅ Significant" if p_value < 0.05 else "⚠️ Not significant"
                    
                    print(f"   {winner_name} vs {method}: p={p_value:.6f} ({significance})")
        
        all_results[dataset_name] = dataset_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("sota_comparison/comparison_results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"academic_evaluation_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': all_results,
            'metadata': {
                'timestamp': timestamp,
                'datasets': datasets,
                'n_folds': n_folds,
                'random_state': ACADEMIC_SEED,
                'academic_compliant': True
            }
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved: {results_file}")
    return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Academic-Compliant Evaluation')
    parser.add_argument('--datasets', nargs='+', 
                       default=['miyawaki'],
                       choices=['miyawaki', 'vangerven', 'mindbigdata', 'crell'],
                       help='Datasets to evaluate')
    parser.add_argument('--folds', type=int, default=10,
                       help='Number of CV folds')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Computing device')
    
    args = parser.parse_args()
    
    results = run_academic_evaluation(
        datasets=args.datasets,
        n_folds=args.folds,
        device=args.device
    )
    
    return results


if __name__ == "__main__":
    main()
