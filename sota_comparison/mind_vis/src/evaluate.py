"""
Mind-Vis Evaluation Script
=========================

Evaluate trained Mind-Vis models with comprehensive metrics.
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
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from mind_vis_model import MindVis
from data.loader import load_dataset_gpu_optimized


def calculate_comprehensive_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """Calculate comprehensive evaluation metrics"""
    
    # Flatten for correlation calculation
    y_true_flat = y_true.cpu().numpy().flatten()
    y_pred_flat = y_pred.cpu().numpy().flatten()
    
    # MSE
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    
    # Correlation
    correlation, _ = pearsonr(y_true_flat, y_pred_flat)
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


def visualize_mind_vis_results(original: torch.Tensor, reconstructed: torch.Tensor,
                              latent_features: torch.Tensor, visual_features: torch.Tensor,
                              dataset_name: str, save_path: str = None) -> None:
    """Visualize Mind-Vis reconstruction results"""
    
    num_samples = min(6, original.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(original[i].cpu().numpy().squeeze(), cmap='gray')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Mind-Vis reconstruction
        axes[1, i].imshow(reconstructed[i].cpu().numpy().squeeze(), cmap='gray')
        axes[1, i].set_title(f'Mind-Vis {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Mind-Vis Reconstructions - {dataset_name.upper()}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved: {save_path}")
    
    plt.show()
    
    # Feature analysis plot
    if latent_features is not None and visual_features is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Latent features distribution
        latent_flat = latent_features.cpu().numpy().flatten()
        ax1.hist(latent_flat, bins=50, alpha=0.7, color='blue')
        ax1.set_title('Latent Features Distribution')
        ax1.set_xlabel('Feature Value')
        ax1.set_ylabel('Frequency')
        
        # Visual features distribution
        visual_flat = visual_features.cpu().numpy().flatten()
        ax2.hist(visual_flat, bins=50, alpha=0.7, color='green')
        ax2.set_title('Visual Features Distribution')
        ax2.set_xlabel('Feature Value')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            feature_path = save_path.replace('.png', '_features.png')
            plt.savefig(feature_path, dpi=300, bbox_inches='tight')
            print(f"💾 Feature analysis saved: {feature_path}")
        
        plt.show()


def evaluate_mind_vis_comprehensive(dataset_name: str, device='cuda', 
                                   num_samples=6, save_results=True):
    """Comprehensive evaluation of Mind-Vis"""
    
    print(f"📊 MIND-VIS COMPREHENSIVE EVALUATION")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Samples: {num_samples}")
    print("=" * 60)
    
    # Load test data
    print("📊 Loading test data...")
    _, _, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device=device)
    
    print(f"✅ Test data loaded: {X_test.shape[0]} samples")
    
    # Load trained model
    model_path = f"models/{dataset_name}_mind_vis_best.pth"
    metadata_path = f"models/{dataset_name}_mind_vis_metadata.json"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please train the model first using train.py")
        return None
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model with same configuration
    model = MindVis(
        input_dim=input_dim,
        device=device,
        image_size=y_test.shape[-1],
        channels=y_test.shape[1]
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Model loaded: {model_path}")
    print(f"   Training info: {metadata['total_epochs']} epochs, "
          f"best loss: {metadata['best_test_loss']:.6f}")
    
    # Limit samples if requested
    if num_samples and num_samples < X_test.shape[0]:
        indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
        X_test = X_test[indices]
        y_test = y_test[indices]
        print(f"   Using {num_samples} random samples")
    
    # Reconstruct images
    print("🎨 Reconstructing images...")
    with torch.no_grad():
        latent_features, visual_features, reconstructed_images = model(X_test)
    
    print(f"✅ Reconstruction complete:")
    print(f"   Latent features: {latent_features.shape}")
    print(f"   Visual features: {visual_features.shape}")
    print(f"   Reconstructed images: {reconstructed_images.shape}")
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    metrics = calculate_comprehensive_metrics(y_test, reconstructed_images)
    
    # Print results
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"{'='*50}")
    print(f"Mind-Vis Performance:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   Correlation: {metrics['correlation']:.6f}")
    print(f"   SSIM: {metrics['ssim_mean']:.6f} ± {metrics['ssim_std']:.6f}")
    
    # Create visualization
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"results/{dataset_name}_mind_vis_evaluation_{timestamp}.png"
        os.makedirs("results", exist_ok=True)
        
        visualize_mind_vis_results(
            y_test, reconstructed_images, latent_features, visual_features,
            dataset_name, viz_path
        )
    
    # Compile results
    results = {
        'dataset_name': dataset_name,
        'method': 'Mind-Vis',
        'num_samples': X_test.shape[0],
        'evaluation_timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'model_info': model.get_model_info(),
        'training_metadata': metadata
    }
    
    # Save results
    if save_results:
        results_path = f"results/{dataset_name}_mind_vis_comprehensive_{timestamp}.json"
        
        # Convert numpy arrays and tensors to lists for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj

        results = convert_to_json_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Mind-Vis Comprehensive Evaluation')
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
                results = evaluate_mind_vis_comprehensive(
                    dataset, device, args.samples, save_results
                )
                if results:
                    all_results[dataset] = results
                    print(f"✅ {dataset}: MSE={results['metrics']['mse']:.6f}")
            except Exception as e:
                print(f"❌ {dataset}: {str(e)}")
                all_results[dataset] = None
        
        # Summary
        print(f"\n{'='*80}")
        print("MIND-VIS EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        for dataset, results in all_results.items():
            if results is not None:
                metrics = results['metrics']
                print(f"{dataset:12}: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
            else:
                print(f"{dataset:12}: FAILED")
    
    else:
        evaluate_mind_vis_comprehensive(args.dataset, device, args.samples, save_results)


if __name__ == "__main__":
    main()
