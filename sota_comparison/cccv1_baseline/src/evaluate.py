"""
CCCV1 Baseline Evaluation Script
===============================

Standard evaluation metrics for fair comparison with SOTA methods.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from model import CCCV1Baseline
from data.loader import load_dataset


def calculate_metrics(y_true, y_pred):
    """Calculate standard evaluation metrics"""
    
    # Flatten arrays for metric calculation
    y_true_flat = y_true.cpu().numpy().flatten()
    y_pred_flat = y_pred.cpu().numpy().flatten()
    
    # MSE
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    
    # Correlation
    correlation, _ = pearsonr(y_true_flat, y_pred_flat)
    
    # SSIM (simplified version)
    def ssim_simple(img1, img2):
        """Simplified SSIM calculation"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    # Calculate SSIM for each image pair
    ssim_scores = []
    for i in range(y_true.shape[0]):
        img_true = y_true[i].cpu().numpy().squeeze()
        img_pred = y_pred[i].cpu().numpy().squeeze()
        ssim_score = ssim_simple(img_true, img_pred)
        ssim_scores.append(ssim_score)
    
    ssim_mean = np.mean(ssim_scores)
    ssim_std = np.std(ssim_scores)
    
    return {
        'mse': mse,
        'correlation': correlation,
        'ssim_mean': ssim_mean,
        'ssim_std': ssim_std
    }


def evaluate_baseline_model(dataset_name, model_path=None, device='cuda', 
                          num_samples=6, save_results=True):
    """Evaluate CCCV1 baseline model"""
    
    print(f"🎯 CCCV1 BASELINE EVALUATION")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Load dataset
    print(f"📊 Loading {dataset_name} dataset...")
    X_train, y_train, X_test, y_test = load_dataset(dataset_name, device=device)
    
    input_dim = X_train.shape[1]
    print(f"✅ Dataset loaded: {X_test.shape[0]} test samples")
    
    # Create model
    model = CCCV1Baseline(input_dim, device)
    
    # Load trained weights
    if model_path is None:
        model_path = f"models/{dataset_name}_baseline_best.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model loaded from {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(0, X_test.shape[0], 32):  # Process in batches
            batch_x = X_test[i:i+32]
            batch_y = y_test[i:i+32]
            
            predictions, _ = model(batch_x)
            
            all_predictions.append(predictions)
            all_targets.append(batch_y)
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    print("📊 Calculating metrics...")
    metrics = calculate_metrics(all_targets, all_predictions)
    
    print(f"✅ Evaluation Results:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   Correlation: {metrics['correlation']:.6f}")
    print(f"   SSIM: {metrics['ssim_mean']:.6f} ± {metrics['ssim_std']:.6f}")
    
    # Generate sample reconstructions
    if num_samples > 0:
        print(f"🎨 Generating {num_samples} sample reconstructions...")
        
        # Select random samples
        indices = np.random.choice(X_test.shape[0], min(num_samples, X_test.shape[0]), replace=False)
        sample_x = X_test[indices]
        sample_y = y_test[indices]
        
        with torch.no_grad():
            sample_pred, _ = model(sample_x)
        
        # Create visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(sample_y[i].cpu().numpy().squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstruction
            axes[1, i].imshow(sample_pred[i].cpu().numpy().squeeze(), cmap='gray')
            axes[1, i].set_title(f'Baseline {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_results:
            # Save visualization
            os.makedirs("results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_path = f"results/{dataset_name}_baseline_evaluation_{timestamp}.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved: {viz_path}")
        
        plt.show()
    
    # Save results
    if save_results:
        results = {
            'dataset_name': dataset_name,
            'model_architecture': 'CCCV1Baseline',
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'num_test_samples': X_test.shape[0],
            'device': str(device)
        }
        
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/{dataset_name}_baseline_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate CCCV1 Baseline Model')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['miyawaki', 'vangerven', 'crell', 'mindbigdata', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (default: models/{dataset}_baseline_best.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of sample reconstructions to generate')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    save_results = not args.no_save
    
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
        all_results = {}
        
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"Evaluating {dataset.upper()}")
            print(f"{'='*60}")
            
            try:
                metrics = evaluate_baseline_model(
                    dataset, args.model, device, args.samples, save_results
                )
                if metrics:
                    all_results[dataset] = metrics
                    print(f"✅ {dataset}: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
            except Exception as e:
                print(f"❌ {dataset}: {str(e)}")
                all_results[dataset] = None
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        for dataset, metrics in all_results.items():
            if metrics is not None:
                print(f"{dataset:12}: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
            else:
                print(f"{dataset:12}: FAILED")
    
    else:
        evaluate_baseline_model(args.dataset, args.model, device, args.samples, save_results)


if __name__ == "__main__":
    main()
