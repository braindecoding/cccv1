"""
Visualization of Neural Decoding Reconstructions
===============================================

Script to visualize target stimuli vs reconstructed outputs for all datasets.
This provides qualitative assessment of reconstruction quality.

Author: [Your Name]
Date: 2025-06-20
Version: 1.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

# Import our modules
try:
    from data import load_dataset_gpu_optimized
    print("✅ Data loader imported successfully")
except ImportError:
    try:
        from src.data.secure_loader import load_dataset_secure
        load_dataset_gpu_optimized = load_dataset_secure
        print("✅ Secure data loader imported")
    except ImportError:
        print("❌ Could not import data loader")
        exit(1)

try:
    from src.models.cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized
    print("✅ Model imported successfully")
except ImportError:
    print("❌ Could not import model")
    exit(1)


def set_visualization_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set font sizes for publication
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def load_trained_model(dataset_name, device='cuda'):
    """Load trained model for a specific dataset."""

    # Load dataset to get input dimensions
    try:
        result = load_dataset_gpu_optimized(dataset_name, device)
        if len(result) == 6:
            X_train, y_train, X_test, y_test, input_dim, metadata = result
        else:
            X_train, y_train, X_test, y_test, input_dim = result
        print(f"✅ Dataset {dataset_name} loaded: input_dim={input_dim}")
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        return None, None, None, None, None
    
    # Create model
    try:
        model = CortexFlowCLIPCNNV1Optimized(input_dim=input_dim, device=device)
        print(f"✅ Model created for {dataset_name}")
        
        # Try to load saved model weights
        model_paths = [
            Path(f"models/{dataset_name}_cccv1_best.pth"),
            Path(f"results/validation_*/best_model_{dataset_name}.pth"),
            Path(f"results/enhanced_validation_*/best_model_{dataset_name}.pth")
        ]

        model_loaded = False
        for model_path in model_paths:
            model_files = list(Path(".").glob(str(model_path)))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                try:
                    model.load_state_dict(torch.load(latest_model, map_location=device))
                    model.eval()
                    print(f"✅ Loaded trained model from {latest_model}")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Could not load model from {latest_model}: {e}")

        if not model_loaded:
            print(f"⚠️ No saved model found for {dataset_name}, using random weights")
            
    except Exception as e:
        print(f"❌ Failed to create model for {dataset_name}: {e}")
        return None, None, None, None, None
    
    return model, X_train, y_train, X_test, y_test


def generate_reconstructions(model, X_test, y_test, n_samples=8):
    """Generate reconstructions for visualization."""
    
    model.eval()
    with torch.no_grad():
        # Select random samples for visualization
        indices = torch.randperm(len(X_test))[:n_samples]
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        
        # Generate reconstructions
        reconstructions, _ = model(X_sample)
        
        # Convert to numpy for visualization
        targets = y_sample.cpu().numpy()
        recons = reconstructions.cpu().numpy()
        
        return targets, recons, indices.cpu().numpy()


def create_reconstruction_comparison(targets, reconstructions, dataset_name, sample_indices):
    """Create comparison visualization between targets and reconstructions."""
    
    n_samples = len(targets)
    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
    
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    # Dataset-specific visualization parameters
    dataset_configs = {
        'miyawaki': {
            'title': 'Miyawaki Visual Patterns',
            'cmap': 'gray',
            'description': 'Complex geometric patterns'
        },
        'vangerven': {
            'title': 'Vangerven Handwritten Digits', 
            'cmap': 'gray',
            'description': 'Digit recognition (0-9)'
        },
        'mindbigdata': {
            'title': 'MindBigData Cross-Modal',
            'cmap': 'viridis',
            'description': 'EEG→fMRI→Visual translation'
        },
        'crell': {
            'title': 'Crell Cross-Modal',
            'cmap': 'viridis', 
            'description': 'EEG→fMRI→Visual translation'
        }
    }
    
    config = dataset_configs.get(dataset_name, {
        'title': dataset_name.title(),
        'cmap': 'gray',
        'description': 'Neural decoding'
    })
    
    for i in range(n_samples):
        # Original target
        target_img = targets[i].squeeze()
        axes[0, i].imshow(target_img, cmap=config['cmap'])
        axes[0, i].set_title(f'Target #{sample_indices[i]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstruction
        recon_img = reconstructions[i].squeeze()
        axes[1, i].imshow(recon_img, cmap=config['cmap'])
        axes[1, i].set_title(f'Reconstruction', fontsize=10)
        axes[1, i].axis('off')
        
        # Difference map
        diff_img = np.abs(target_img - recon_img)
        im = axes[2, i].imshow(diff_img, cmap='Reds')
        axes[2, i].set_title(f'|Difference|', fontsize=10)
        axes[2, i].axis('off')
        
        # Add MSE for this sample
        mse = np.mean((target_img - recon_img) ** 2)
        axes[2, i].text(0.5, -0.1, f'MSE: {mse:.4f}', 
                       transform=axes[2, i].transAxes, 
                       ha='center', fontsize=8)
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Target', rotation=90, 
                   transform=axes[0, 0].transAxes, 
                   ha='center', va='center', fontsize=12, weight='bold')
    axes[1, 0].text(-0.1, 0.5, 'Reconstruction', rotation=90,
                   transform=axes[1, 0].transAxes,
                   ha='center', va='center', fontsize=12, weight='bold')
    axes[2, 0].text(-0.1, 0.5, 'Difference', rotation=90,
                   transform=axes[2, 0].transAxes,
                   ha='center', va='center', fontsize=12, weight='bold')
    
    # Overall title
    fig.suptitle(f'{config["title"]}\n{config["description"]}', 
                fontsize=14, weight='bold')
    
    plt.tight_layout()
    return fig


def calculate_reconstruction_metrics(targets, reconstructions):
    """Calculate quantitative metrics for reconstructions."""
    
    # Flatten for calculations
    targets_flat = targets.reshape(len(targets), -1)
    recons_flat = reconstructions.reshape(len(reconstructions), -1)
    
    # Calculate metrics
    mse_per_sample = np.mean((targets_flat - recons_flat) ** 2, axis=1)
    
    # Structural similarity (simplified)
    correlations = []
    for i in range(len(targets)):
        target_flat = targets_flat[i]
        recon_flat = recons_flat[i]
        
        # Pearson correlation
        if np.std(target_flat) > 0 and np.std(recon_flat) > 0:
            corr = np.corrcoef(target_flat, recon_flat)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)
    
    correlations = np.array(correlations)
    
    metrics = {
        'mse_mean': np.mean(mse_per_sample),
        'mse_std': np.std(mse_per_sample),
        'mse_per_sample': mse_per_sample,
        'correlation_mean': np.mean(correlations),
        'correlation_std': np.std(correlations),
        'correlations': correlations
    }
    
    return metrics


def create_metrics_summary(all_metrics, dataset_names):
    """Create summary visualization of reconstruction metrics."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE comparison
    mse_means = [all_metrics[dataset]['mse_mean'] for dataset in dataset_names]
    mse_stds = [all_metrics[dataset]['mse_std'] for dataset in dataset_names]

    bars1 = axes[0,0].bar(dataset_names, mse_means, yerr=mse_stds,
                       capsize=5, alpha=0.7, color='skyblue')
    axes[0,0].set_title('Reconstruction MSE by Dataset')
    axes[0,0].set_ylabel('Mean Squared Error')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, mean_val in zip(bars1, mse_means):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{mean_val:.4f}', ha='center', va='bottom', fontsize=10)

    # Correlation comparison
    corr_means = [all_metrics[dataset]['correlation_mean'] for dataset in dataset_names]
    corr_stds = [all_metrics[dataset]['correlation_std'] for dataset in dataset_names]

    bars2 = axes[0,1].bar(dataset_names, corr_means, yerr=corr_stds,
                       capsize=5, alpha=0.7, color='lightcoral')
    axes[0,1].set_title('Reconstruction Correlation by Dataset')
    axes[0,1].set_ylabel('Pearson Correlation')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_ylim(0, 1)

    # Add value labels on bars
    for bar, mean_val in zip(bars2, corr_means):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)

    # MSE distribution
    mse_data = [all_metrics[dataset]['mse_per_sample'] for dataset in dataset_names]
    axes[1,0].boxplot(mse_data, labels=dataset_names)
    axes[1,0].set_title('MSE Distribution per Dataset')
    axes[1,0].set_ylabel('MSE per Sample')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Correlation distribution
    corr_data = [all_metrics[dataset]['correlations'] for dataset in dataset_names]
    axes[1,1].boxplot(corr_data, labels=dataset_names)
    axes[1,1].set_title('Correlation Distribution per Dataset')
    axes[1,1].set_ylabel('Correlation per Sample')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].set_ylim(0, 1)

    plt.tight_layout()
    return fig


def visualize_dataset_reconstructions(dataset_name, device='cuda', n_samples=6):
    """Visualize reconstructions for a specific dataset."""
    
    print(f"\n🎨 VISUALIZING RECONSTRUCTIONS: {dataset_name.upper()}")
    print("=" * 50)
    
    # Load model and data
    model, X_train, y_train, X_test, y_test = load_trained_model(dataset_name, device)
    
    if model is None:
        print(f"❌ Failed to load model for {dataset_name}")
        return None, None
    
    # Generate reconstructions
    print(f"🔄 Generating reconstructions for {n_samples} samples...")
    targets, reconstructions, sample_indices = generate_reconstructions(
        model, X_test, y_test, n_samples
    )
    
    # Calculate metrics
    metrics = calculate_reconstruction_metrics(targets, reconstructions)
    
    print(f"📊 Reconstruction Metrics:")
    print(f"   MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"   Correlation: {metrics['correlation_mean']:.3f} ± {metrics['correlation_std']:.3f}")
    
    # Create visualization
    fig = create_reconstruction_comparison(targets, reconstructions, dataset_name, sample_indices)
    
    return fig, metrics


def main():
    """Main visualization function."""
    
    parser = argparse.ArgumentParser(description='Visualize neural decoding reconstructions')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['miyawaki', 'vangerven', 'mindbigdata', 'crell', 'all'],
                       help='Dataset to visualize')
    parser.add_argument('--samples', type=int, default=6,
                       help='Number of samples to visualize per dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save', action='store_true',
                       help='Save visualizations to file')
    
    args = parser.parse_args()
    
    # Set up visualization style
    set_visualization_style()
    
    print("🎨 NEURAL DECODING RECONSTRUCTION VISUALIZATION")
    print("=" * 60)
    print(f"📊 Samples per dataset: {args.samples}")
    print(f"🖥️ Device: {args.device}")
    
    # Determine datasets to visualize
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']
    else:
        datasets = [args.dataset]
    
    # Store results
    all_figures = {}
    all_metrics = {}
    
    # Visualize each dataset
    for dataset in datasets:
        try:
            fig, metrics = visualize_dataset_reconstructions(
                dataset, args.device, args.samples
            )
            
            if fig is not None:
                all_figures[dataset] = fig
                all_metrics[dataset] = metrics
                
                # Show plot
                plt.show()
                
                # Save if requested
                if args.save:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = Path(f"results/visualizations_{timestamp}")
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    fig_path = save_dir / f"reconstruction_{dataset}.png"
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"💾 Saved: {fig_path}")
                
        except Exception as e:
            print(f"❌ Error visualizing {dataset}: {e}")
    
    # Create summary if multiple datasets
    if len(all_metrics) > 1:
        print(f"\n📊 CREATING METRICS SUMMARY")
        print("=" * 40)
        
        summary_fig = create_metrics_summary(all_metrics, list(all_metrics.keys()))
        plt.show()
        
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(f"results/visualizations_{timestamp}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            summary_path = save_dir / "metrics_summary.png"
            summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
            print(f"💾 Summary saved: {summary_path}")
    
    print(f"\n✅ Visualization complete!")
    print(f"📊 Datasets visualized: {len(all_figures)}")
    
    return all_figures, all_metrics


if __name__ == "__main__":
    main()
