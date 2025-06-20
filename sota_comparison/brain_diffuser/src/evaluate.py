"""
Brain-Diffuser Evaluation Script
===============================

Evaluate trained Brain-Diffuser models with comprehensive metrics.
Academic Integrity: Standard evaluation following original paper.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from brain_diffuser import BrainDiffuser
from data.loader import load_dataset_gpu_optimized


def calculate_comprehensive_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Calculate comprehensive evaluation metrics"""
    
    # Flatten for correlation calculation
    y_true_flat = y_true.cpu().numpy().flatten()
    y_pred_flat = y_pred.cpu().numpy().flatten()
    
    # MSE
    mse = torch.nn.functional.mse_loss(y_pred, y_true).item()
    
    # Correlation
    correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # SSIM (simplified)
    def calculate_ssim(img1, img2):
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
        ssim_score = calculate_ssim(img_true, img_pred)
        ssim_scores.append(ssim_score)
    
    ssim_mean = np.mean(ssim_scores)
    ssim_std = np.std(ssim_scores)
    
    return {
        'mse': mse,
        'correlation': correlation,
        'ssim_mean': ssim_mean,
        'ssim_std': ssim_std,
        'ssim_scores': ssim_scores
    }


def visualize_reconstructions(original: torch.Tensor, 
                            initial: torch.Tensor,
                            final: torch.Tensor,
                            dataset_name: str,
                            save_path: str = None) -> None:
    """Visualize reconstruction results"""
    
    num_samples = min(6, original.shape[0])
    
    fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(original[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Initial guess (VDVAE)
        initial_resized = torch.nn.functional.interpolate(
            initial[i:i+1], size=(28, 28), mode='bilinear', align_corners=False
        )
        if initial_resized.shape[1] == 3:
            initial_resized = torch.mean(initial_resized, dim=1, keepdim=True)
        
        axes[1, i].imshow(initial_resized[0].cpu().numpy().squeeze(), cmap='gray')
        axes[1, i].set_title(f'VDVAE {i+1}')
        axes[1, i].axis('off')
        
        # Final (Brain-Diffuser)
        final_resized = torch.nn.functional.interpolate(
            final[i:i+1], size=(28, 28), mode='bilinear', align_corners=False
        )
        if final_resized.shape[1] == 3:
            final_resized = torch.mean(final_resized, dim=1, keepdim=True)
        
        axes[2, i].imshow(final_resized[0].cpu().numpy().squeeze(), cmap='gray')
        axes[2, i].set_title(f'Brain-Diffuser {i+1}')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Brain-Diffuser Reconstructions - {dataset_name.upper()}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved: {save_path}")
    
    plt.show()


def evaluate_brain_diffuser_comprehensive(dataset_name: str, device='cuda', 
                                        num_samples=6, save_results=True):
    """Comprehensive evaluation of Brain-Diffuser"""
    
    print(f"📊 BRAIN-DIFFUSER COMPREHENSIVE EVALUATION")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Samples: {num_samples}")
    print("=" * 60)
    
    # Initialize Brain-Diffuser
    brain_diffuser = BrainDiffuser(device=device)
    
    # Setup models
    print("🔧 Setting up models...")
    if not brain_diffuser.setup_models():
        print("❌ Failed to setup models")
        return None
    
    # Load trained models
    print("📥 Loading trained models...")
    if not brain_diffuser.load_trained_models(dataset_name):
        print("❌ Failed to load trained models")
        print("   Please train the model first using train.py")
        return None
    
    # Load test data
    print("📊 Loading test data...")
    _, _, X_test, y_test, _ = load_dataset_gpu_optimized(dataset_name, device=device)
    
    print(f"✅ Test data loaded: {X_test.shape[0]} samples")
    
    # Limit samples if requested
    if num_samples and num_samples < X_test.shape[0]:
        indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
        print(f"   Using {num_samples} random samples")
    
    # Reconstruct images
    print("🎨 Reconstructing images...")
    try:
        initial_images, final_images = brain_diffuser.reconstruct(X_test)
        
        print(f"✅ Reconstruction complete:")
        print(f"   Initial images: {initial_images.shape}")
        print(f"   Final images: {final_images.shape}")
        
    except Exception as e:
        print(f"❌ Reconstruction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Prepare images for evaluation
    # Resize final images to match ground truth
    final_resized = torch.nn.functional.interpolate(
        final_images, size=(28, 28), mode='bilinear', align_corners=False
    )
    
    # Convert to grayscale if needed
    if final_resized.shape[1] == 3:
        final_resized = torch.mean(final_resized, dim=1, keepdim=True)
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    metrics = calculate_comprehensive_metrics(y_test, final_resized)
    
    # Also evaluate initial guess (VDVAE stage)
    initial_resized = torch.nn.functional.interpolate(
        initial_images, size=(28, 28), mode='bilinear', align_corners=False
    )
    if initial_resized.shape[1] == 3:
        initial_resized = torch.mean(initial_resized, dim=1, keepdim=True)
    
    initial_metrics = calculate_comprehensive_metrics(y_test, initial_resized)
    
    # Print results
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"{'='*50}")
    print(f"VDVAE (Initial Guess):")
    print(f"   MSE: {initial_metrics['mse']:.6f}")
    print(f"   Correlation: {initial_metrics['correlation']:.6f}")
    print(f"   SSIM: {initial_metrics['ssim_mean']:.6f} ± {initial_metrics['ssim_std']:.6f}")
    
    print(f"\nBrain-Diffuser (Final):")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   Correlation: {metrics['correlation']:.6f}")
    print(f"   SSIM: {metrics['ssim_mean']:.6f} ± {metrics['ssim_std']:.6f}")
    
    # Improvement calculation
    mse_improvement = ((initial_metrics['mse'] - metrics['mse']) / initial_metrics['mse']) * 100
    corr_improvement = ((metrics['correlation'] - initial_metrics['correlation']) / abs(initial_metrics['correlation'])) * 100
    
    print(f"\nImprovement (Final vs Initial):")
    print(f"   MSE: {mse_improvement:+.2f}%")
    print(f"   Correlation: {corr_improvement:+.2f}%")
    
    # Create visualization
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"results/{dataset_name}_brain_diffuser_visualization_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        
        visualize_reconstructions(
            y_test, initial_images, final_images, 
            dataset_name, viz_path
        )
    
    # Compile results
    results = {
        'dataset_name': dataset_name,
        'method': 'Brain-Diffuser',
        'num_samples': X_test.shape[0],
        'evaluation_timestamp': datetime.now().isoformat(),
        'vdvae_metrics': initial_metrics,
        'brain_diffuser_metrics': metrics,
        'improvements': {
            'mse_improvement_percent': mse_improvement,
            'correlation_improvement_percent': corr_improvement
        },
        'model_info': brain_diffuser.get_model_info()
    }
    
    # Save results
    if save_results:
        results_path = f"results/{dataset_name}_brain_diffuser_comprehensive_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        for metric_set in [results['vdvae_metrics'], results['brain_diffuser_metrics']]:
            if 'ssim_scores' in metric_set:
                metric_set['ssim_scores'] = [float(x) for x in metric_set['ssim_scores']]
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Brain-Diffuser Comprehensive Evaluation')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['miyawaki', 'vangerven', 'crell', 'mindbigdata', 'all'],
                        help='Dataset to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    save_results = not args.no_save
    
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
        all_results = {}
        
        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"EVALUATING: {dataset.upper()}")
            print(f"{'='*80}")
            
            try:
                results = evaluate_brain_diffuser_comprehensive(
                    dataset, device, args.samples, save_results
                )
                if results:
                    all_results[dataset] = results
                    print(f"✅ {dataset}: MSE={results['brain_diffuser_metrics']['mse']:.6f}")
            except Exception as e:
                print(f"❌ {dataset}: {str(e)}")
                all_results[dataset] = None
        
        # Summary
        print(f"\n{'='*80}")
        print("BRAIN-DIFFUSER EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        for dataset, results in all_results.items():
            if results is not None:
                metrics = results['brain_diffuser_metrics']
                print(f"{dataset:12}: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
            else:
                print(f"{dataset:12}: FAILED")
    
    else:
        evaluate_brain_diffuser_comprehensive(args.dataset, device, args.samples, save_results)


if __name__ == "__main__":
    main()
