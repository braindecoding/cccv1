"""
Simple SOTA Comparison Test
==========================

Simplified test of comparison framework focusing on available methods.
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cccv1_optimized_adapter import CCCV1OptimizedAdapter
from data.loader import load_dataset_gpu_optimized


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Calculate basic evaluation metrics"""
    
    # Flatten for correlation calculation
    y_true_flat = y_true.cpu().numpy().flatten()
    y_pred_flat = y_pred.cpu().numpy().flatten()
    
    # MSE
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    
    # Correlation
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # SSIM (simplified)
    def calculate_ssim_batch(imgs_true, imgs_pred):
        ssim_scores = []
        for i in range(imgs_true.shape[0]):
            img1 = imgs_true[i].cpu().numpy().squeeze()
            img2 = imgs_pred[i].cpu().numpy().squeeze()
            
            mu1 = np.mean(img1)
            mu2 = np.mean(img2)
            sigma1 = np.var(img1)
            sigma2 = np.var(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
            
            ssim_scores.append(ssim)
        
        return ssim_scores
    
    ssim_scores = calculate_ssim_batch(y_true, y_pred)
    ssim_mean = np.mean(ssim_scores)
    ssim_std = np.std(ssim_scores)
    
    return {
        'mse': float(mse),
        'correlation': float(correlation),
        'ssim_mean': float(ssim_mean),
        'ssim_std': float(ssim_std)
    }


def evaluate_cccv1_optimized(dataset_name: str, device='cuda', num_samples=6):
    """Evaluate CCCV1 Optimized method"""
    
    print(f"📊 Evaluating CCCV1-Optimized on {dataset_name}...")
    
    try:
        # Create adapter
        adapter = CCCV1OptimizedAdapter(dataset_name, device)
        
        # Load test data
        _, _, X_test, y_test, _ = load_dataset_gpu_optimized(dataset_name, device=device)
        
        if num_samples and num_samples < X_test.shape[0]:
            indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        # Generate predictions
        with torch.no_grad():
            predictions, _ = adapter.model(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, predictions)
        
        result = {
            'method': 'CCCV1-Optimized',
            'dataset': dataset_name,
            'num_samples': X_test.shape[0],
            'metrics': metrics,
            'status': 'success'
        }
        
        print(f"✅ CCCV1-Optimized: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
        return result
        
    except Exception as e:
        print(f"❌ CCCV1-Optimized failed: {str(e)}")
        return {
            'method': 'CCCV1-Optimized',
            'dataset': dataset_name,
            'status': 'failed',
            'error': str(e)
        }


def evaluate_mind_vis(dataset_name: str, device='cuda', num_samples=6):
    """Evaluate Mind-Vis method"""
    
    print(f"📊 Evaluating Mind-Vis on {dataset_name}...")
    
    try:
        # Load test data
        _, _, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device=device)
        
        if num_samples and num_samples < X_test.shape[0]:
            indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        # Check for trained model
        model_path = f"models/{dataset_name}_mind_vis_best.pth"
        
        if not os.path.exists(model_path):
            print(f"⚠️  No trained Mind-Vis model for {dataset_name}")
            return {
                'method': 'Mind-Vis',
                'dataset': dataset_name,
                'status': 'no_model',
                'note': 'Model not trained'
            }
        
        # Import Mind-Vis model
        from mind_vis.src.mind_vis_model import MindVis
        
        # Create and load model
        model = MindVis(
            input_dim=input_dim,
            device=device,
            image_size=y_test.shape[-1],
            channels=y_test.shape[1]
        )
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Generate predictions
        with torch.no_grad():
            _, _, predictions = model(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, predictions)
        
        result = {
            'method': 'Mind-Vis',
            'dataset': dataset_name,
            'num_samples': X_test.shape[0],
            'metrics': metrics,
            'status': 'success'
        }
        
        print(f"✅ Mind-Vis: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
        return result
        
    except Exception as e:
        print(f"❌ Mind-Vis failed: {str(e)}")
        return {
            'method': 'Mind-Vis',
            'dataset': dataset_name,
            'status': 'failed',
            'error': str(e)
        }


def evaluate_lightweight_brain_diffuser(dataset_name: str, device='cuda', num_samples=6):
    """Evaluate Lightweight Brain-Diffuser method"""

    print(f"📊 Evaluating Lightweight-Brain-Diffuser on {dataset_name}...")

    try:
        # Load test data
        _, _, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device=device)

        if num_samples and num_samples < X_test.shape[0]:
            indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]

        # Check for trained model
        model_path = f"models/{dataset_name}_lightweight_brain_diffuser_best.pth"

        if not os.path.exists(model_path):
            print(f"⚠️  No trained Lightweight Brain-Diffuser model for {dataset_name}")
            return {
                'method': 'Lightweight-Brain-Diffuser',
                'dataset': dataset_name,
                'status': 'no_model',
                'note': 'Model not trained'
            }

        # Import Lightweight Brain-Diffuser model
        from brain_diffuser.src.lightweight_brain_diffuser import LightweightBrainDiffuser

        # Create and load model
        model = LightweightBrainDiffuser(
            input_dim=input_dim,
            device=device,
            image_size=y_test.shape[-1]
        )

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Generate predictions
        with torch.no_grad():
            _, predictions = model(X_test, use_diffusion=True)

        # Calculate metrics
        metrics = calculate_metrics(y_test, predictions)

        result = {
            'method': 'Lightweight-Brain-Diffuser',
            'dataset': dataset_name,
            'num_samples': X_test.shape[0],
            'metrics': metrics,
            'status': 'success'
        }

        print(f"✅ Lightweight-Brain-Diffuser: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
        return result

    except Exception as e:
        print(f"❌ Lightweight-Brain-Diffuser failed: {str(e)}")
        return {
            'method': 'Lightweight-Brain-Diffuser',
            'dataset': dataset_name,
            'status': 'failed',
            'error': str(e)
        }


def run_simple_comparison(datasets=['miyawaki'], num_samples=6, device='cuda'):
    """Run simple comparison between available methods"""
    
    print(f"🎯 SIMPLE SOTA COMPARISON")
    print(f"Datasets: {datasets}")
    print(f"Samples: {num_samples}")
    print(f"Device: {device}")
    print("=" * 60)
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*50}")
        
        dataset_results = {}
        
        # Evaluate CCCV1-Optimized
        dataset_results['CCCV1-Optimized'] = evaluate_cccv1_optimized(dataset, device, num_samples)
        
        # Evaluate Mind-Vis
        dataset_results['Mind-Vis'] = evaluate_mind_vis(dataset, device, num_samples)
        
        # Evaluate Lightweight Brain-Diffuser
        dataset_results['Lightweight-Brain-Diffuser'] = evaluate_lightweight_brain_diffuser(dataset, device, num_samples)
        
        all_results[dataset] = dataset_results
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for dataset, methods in all_results.items():
        print(f"\n{dataset.upper()}:")
        for method, result in methods.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                print(f"  {method:15}: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}, SSIM={metrics['ssim_mean']:.6f}")
            elif result['status'] == 'mock':
                metrics = result['metrics']
                print(f"  {method:15}: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}, SSIM={metrics['ssim_mean']:.6f} (MOCK)")
            else:
                print(f"  {method:15}: {result['status'].upper()}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"comparison_results/simple_comparison_{timestamp}.json"
    
    os.makedirs("comparison_results", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved: {results_path}")
    
    return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple SOTA Comparison Test')
    parser.add_argument('--datasets', nargs='+', default=['miyawaki'],
                        choices=['miyawaki', 'vangerven', 'crell', 'mindbigdata'],
                        help='Datasets to evaluate')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of samples per dataset')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Run comparison
    results = run_simple_comparison(args.datasets, args.samples, device)
    
    print(f"\n🎉 Simple comparison complete!")


if __name__ == "__main__":
    main()
