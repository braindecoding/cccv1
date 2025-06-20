"""
Visualize Reconstructions Using Saved CV Model
==============================================

Script to visualize reconstructions using the best model saved from cross-validation.
This is the CORRECT approach - no need to retrain!

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
import json

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


def load_cv_saved_model(dataset_name, device='cuda'):
    """Load the best model saved from cross-validation."""
    
    print(f"🔍 Loading CV saved model for {dataset_name}...")
    
    # Check for saved model
    models_dir = Path("models")
    model_path = models_dir / f"{dataset_name}_cv_best.pth"
    metadata_path = models_dir / f"{dataset_name}_cv_best_metadata.json"
    
    if not model_path.exists():
        print(f"❌ No saved CV model found at {model_path}")
        print(f"💡 Please run cross-validation first: python scripts/validate_cccv1.py --dataset {dataset_name}")
        return None, None, None
    
    if not metadata_path.exists():
        print(f"❌ No metadata found at {metadata_path}")
        return None, None, None
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded:")
        print(f"   Best fold: {metadata['best_fold']}")
        print(f"   Best score: {metadata['best_score']:.6f}")
        print(f"   Save time: {metadata['save_timestamp']}")
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return None, None, None
    
    # Load dataset to get input dimensions
    try:
        result = load_dataset_gpu_optimized(dataset_name, device)
        if len(result) == 6:
            X_train, y_train, X_test, y_test, input_dim, _ = result
        else:
            X_train, y_train, X_test, y_test, input_dim = result
        print(f"✅ Dataset loaded: input_dim={input_dim}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None, None, None
    
    # Create model with same architecture
    try:
        model = CortexFlowCLIPCNNV1Optimized(input_dim=input_dim, device=device)
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None, None, None
    
    # Load saved state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Model weights loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        return None, None, None
    
    return model, (X_train, y_train, X_test, y_test), metadata


def generate_reconstructions(model, X_test, y_test, n_samples=6):
    """Generate reconstructions using the loaded CV model."""
    
    model.eval()
    with torch.no_grad():
        # Select samples for visualization
        if len(X_test) >= n_samples:
            indices = torch.randperm(len(X_test))[:n_samples]
        else:
            indices = torch.arange(len(X_test))
            
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        
        # Generate reconstructions
        reconstructions, _ = model(X_sample)
        
        # Convert to numpy for visualization
        targets = y_sample.cpu().numpy()
        recons = reconstructions.cpu().numpy()
        
        return targets, recons, indices.cpu().numpy()


def create_cv_model_visualization(targets, reconstructions, dataset_name, sample_indices, metadata):
    """Create visualization using CV model results."""
    
    n_samples = len(targets)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Main grid for images
    gs_main = fig.add_gridspec(3, n_samples, height_ratios=[1, 1, 1], 
                              left=0.05, right=0.7, top=0.85, bottom=0.15,
                              hspace=0.3, wspace=0.1)
    
    # Side panel for CV info and metrics
    gs_side = fig.add_gridspec(3, 1, left=0.75, right=0.98, top=0.85, bottom=0.15)
    
    # Dataset configuration
    dataset_configs = {
        'miyawaki': {'title': 'Miyawaki Visual Patterns', 'cmap': 'gray'},
        'vangerven': {'title': 'Vangerven Handwritten Digits', 'cmap': 'gray'},
        'mindbigdata': {'title': 'MindBigData Cross-Modal', 'cmap': 'viridis'},
        'crell': {'title': 'Crell Cross-Modal', 'cmap': 'viridis'}
    }
    
    config = dataset_configs.get(dataset_name, {'title': dataset_name.title(), 'cmap': 'gray'})
    
    # Calculate metrics
    sample_mses = []
    sample_corrs = []
    
    # Create image grid
    for i in range(n_samples):
        # Target
        ax_target = fig.add_subplot(gs_main[0, i])
        target_img = targets[i].squeeze()
        ax_target.imshow(target_img, cmap=config['cmap'])
        ax_target.set_title(f'Target #{sample_indices[i]}', fontsize=11, fontweight='bold')
        ax_target.axis('off')
        
        # Reconstruction
        ax_recon = fig.add_subplot(gs_main[1, i])
        recon_img = reconstructions[i].squeeze()
        ax_recon.imshow(recon_img, cmap=config['cmap'])
        ax_recon.set_title('CV Model Reconstruction', fontsize=11, fontweight='bold')
        ax_recon.axis('off')
        
        # Difference
        ax_diff = fig.add_subplot(gs_main[2, i])
        diff_img = np.abs(target_img - recon_img)
        ax_diff.imshow(diff_img, cmap='Reds')
        ax_diff.set_title('|Difference|', fontsize=11, fontweight='bold')
        ax_diff.axis('off')
        
        # Calculate sample metrics
        sample_mse = np.mean((target_img - recon_img) ** 2)
        sample_mses.append(sample_mse)
        
        target_flat = target_img.flatten()
        recon_flat = recon_img.flatten()
        if np.std(target_flat) > 0 and np.std(recon_flat) > 0:
            sample_corr = np.corrcoef(target_flat, recon_flat)[0, 1]
        else:
            sample_corr = 0
        sample_corrs.append(sample_corr)
        
        # Add sample metrics
        ax_diff.text(0.5, -0.15, f'MSE: {sample_mse:.4f}\nCorr: {sample_corr:.3f}', 
                    transform=ax_diff.transAxes, ha='center', va='top', fontsize=9)
    
    # Row labels
    fig.text(0.02, 0.7, 'Target', rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'CV Model\nReconstruction', rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.3, 'Difference', rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Side panel - CV Model Info
    ax_cv_info = fig.add_subplot(gs_side[0, 0])
    ax_cv_info.axis('off')
    
    cv_info_text = f"""
    🏆 CROSS-VALIDATION MODEL
    
    {config['title']}
    
    📊 CV Training Results:
    • Best Fold: {metadata['best_fold']}/10
    • Best CV Score: {metadata['best_score']:.6f}
    • Model: {metadata['model_architecture']}
    
    🎯 This is the ACTUAL model from CV evaluation!
    ✅ No retraining needed
    ✅ Same model that achieved CV performance
    """
    
    ax_cv_info.text(0.05, 0.95, cv_info_text, transform=ax_cv_info.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Side panel - Reconstruction Metrics
    ax_metrics = fig.add_subplot(gs_side[1, 0])
    
    x = np.arange(n_samples)
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, sample_mses, width, label='MSE', alpha=0.7, color='skyblue')
    ax_metrics_twin = ax_metrics.twinx()
    bars2 = ax_metrics_twin.bar(x + width/2, sample_corrs, width, label='Correlation', alpha=0.7, color='lightcoral')
    
    ax_metrics.set_xlabel('Sample')
    ax_metrics.set_ylabel('MSE', color='blue')
    ax_metrics_twin.set_ylabel('Correlation', color='red')
    ax_metrics.set_title('Per-Sample Quality Metrics')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels([f'#{idx}' for idx in sample_indices])
    
    # Side panel - Quality Summary
    ax_summary = fig.add_subplot(gs_side[2, 0])
    ax_summary.axis('off')
    
    mean_mse = np.mean(sample_mses)
    mean_corr = np.mean(sample_corrs)
    
    # Determine quality level
    if mean_corr > 0.8:
        quality_level = "Excellent"
        quality_color = "green"
    elif mean_corr > 0.6:
        quality_level = "Good"
        quality_color = "orange"
    elif mean_corr > 0.4:
        quality_level = "Moderate"
        quality_color = "yellow"
    else:
        quality_level = "Poor"
        quality_color = "red"
    
    summary_text = f"""
    📈 RECONSTRUCTION QUALITY: {quality_level}
    
    📊 Overall Metrics:
    • Mean MSE: {mean_mse:.6f} ± {np.std(sample_mses):.6f}
    • Mean Correlation: {mean_corr:.3f} ± {np.std(sample_corrs):.3f}
    
    🎯 Best Sample: #{sample_indices[np.argmin(sample_mses)]}
    📉 Worst Sample: #{sample_indices[np.argmax(sample_mses)]}
    
    ✨ Model saved: {metadata['save_timestamp'][:10]}
    """
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor=quality_color, alpha=0.3))
    
    # Overall title
    fig.suptitle(f'{config["title"]} - Reconstruction using Cross-Validation Best Model\n'
                f'✅ Using ACTUAL CV model (Fold {metadata["best_fold"]}, Score: {metadata["best_score"]:.6f})', 
                fontsize=16, fontweight='bold')
    
    return fig, {
        'mse_mean': mean_mse,
        'mse_std': np.std(sample_mses),
        'correlation_mean': mean_corr,
        'correlation_std': np.std(sample_corrs),
        'cv_metadata': metadata
    }


def create_all_datasets_summary(all_metrics, dataset_names):
    """Create summary comparison visualization for all datasets."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data for plotting
    mse_means = [all_metrics[dataset]['mse_mean'] for dataset in dataset_names]
    mse_stds = [all_metrics[dataset]['mse_std'] for dataset in dataset_names]
    corr_means = [all_metrics[dataset]['correlation_mean'] for dataset in dataset_names]
    corr_stds = [all_metrics[dataset]['correlation_std'] for dataset in dataset_names]
    cv_scores = [all_metrics[dataset]['cv_metadata']['best_score'] for dataset in dataset_names]

    # Colors for each dataset
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(dataset_names)]

    # 1. MSE Comparison
    bars1 = axes[0,0].bar(dataset_names, mse_means, yerr=mse_stds, capsize=5,
                         alpha=0.8, color=colors)
    axes[0,0].set_title('Reconstruction MSE by Dataset', fontweight='bold')
    axes[0,0].set_ylabel('MSE')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, mean_val in zip(bars1, mse_means):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                      f'{mean_val:.4f}', ha='center', va='bottom', fontweight='bold')

    # 2. Correlation Comparison
    bars2 = axes[0,1].bar(dataset_names, corr_means, yerr=corr_stds, capsize=5,
                         alpha=0.8, color=colors)
    axes[0,1].set_title('Reconstruction Correlation by Dataset', fontweight='bold')
    axes[0,1].set_ylabel('Pearson Correlation')
    axes[0,1].set_ylim(0, 1)
    axes[0,1].tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, mean_val in zip(bars2, corr_means):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

    # 3. CV Score vs Visualization MSE
    scatter = axes[1,0].scatter(cv_scores, mse_means, s=200, alpha=0.8, c=colors)

    # Add dataset labels
    for i, dataset in enumerate(dataset_names):
        axes[1,0].annotate(dataset.title(), (cv_scores[i], mse_means[i]),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=10, fontweight='bold')

    axes[1,0].set_title('CV Score vs Visualization MSE', fontweight='bold')
    axes[1,0].set_xlabel('CV Score (MSE)')
    axes[1,0].set_ylabel('Visualization MSE')

    # 4. Quality Summary Table
    axes[1,1].axis('off')

    # Create summary table
    table_data = []
    for i, dataset in enumerate(dataset_names):
        quality_level = "Excellent" if corr_means[i] > 0.8 else "Good" if corr_means[i] > 0.6 else "Moderate"
        table_data.append([
            dataset.title(),
            f"{mse_means[i]:.4f}±{mse_stds[i]:.4f}",
            f"{corr_means[i]:.3f}±{corr_stds[i]:.3f}",
            f"{cv_scores[i]:.6f}",
            quality_level
        ])

    table = axes[1,1].table(cellText=table_data,
                           colLabels=['Dataset', 'MSE', 'Correlation', 'CV Score', 'Quality'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(dataset_names) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#E8E8E8')
                cell.set_text_props(weight='bold')
            else:
                # Color code by quality
                if j == 4:  # Quality column
                    quality = table_data[i-1][4]
                    if quality == "Excellent":
                        cell.set_facecolor('#D5F4E6')
                    elif quality == "Good":
                        cell.set_facecolor('#FFF3CD')
                    else:
                        cell.set_facecolor('#F8D7DA')

    axes[1,1].set_title('Summary Statistics', fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Cross-Validation Model Reconstruction Summary\nAll Datasets Comparison',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    return fig


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Visualize reconstructions using saved CV model')
    parser.add_argument('--dataset', type=str, default='miyawaki',
                       choices=['miyawaki', 'vangerven', 'mindbigdata', 'crell', 'all'],
                       help='Dataset to visualize (or "all" for all datasets)')
    parser.add_argument('--samples', type=int, default=6,
                       help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save', action='store_true',
                       help='Save visualizations to file')
    
    args = parser.parse_args()
    
    # Set visualization style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    print("🎨 NEURAL DECODING VISUALIZATION WITH SAVED CV MODEL")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"📈 Samples: {args.samples}")
    print(f"🖥️ Device: {args.device}")
    print(f"🏆 Method: Using BEST model from cross-validation")
    print()

    # Handle multiple datasets
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'mindbigdata', 'crell']

        # Storage for results
        all_figures = {}
        all_metrics = {}
        successful_datasets = []

        # Process each dataset
        for dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"🎨 PROCESSING: {dataset_name.upper()}")
            print(f"{'='*60}")

            # Load CV model and visualize
            model, data, metadata = load_cv_saved_model(dataset_name, args.device)

            if model is None:
                print(f"❌ Failed to load CV model for {dataset_name}")
                print(f"💡 Run cross-validation first: python scripts/validate_cccv1.py --dataset {dataset_name}")
                continue

            X_train, y_train, X_test, y_test = data

            # Generate reconstructions
            print(f"🎨 Generating reconstructions for {args.samples} samples...")
            targets, reconstructions, sample_indices = generate_reconstructions(
                model, X_test, y_test, args.samples
            )

            # Create visualization
            fig, metrics = create_cv_model_visualization(
                targets, reconstructions, dataset_name, sample_indices, metadata
            )

            print(f"📊 Reconstruction Quality (CV Model):")
            print(f"   MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
            print(f"   Correlation: {metrics['correlation_mean']:.3f} ± {metrics['correlation_std']:.3f}")
            print(f"   CV Score: {metadata['best_score']:.6f} (Fold {metadata['best_fold']})")

            # Store results
            all_figures[dataset_name] = fig
            all_metrics[dataset_name] = metrics
            successful_datasets.append(dataset_name)

            # Show plot
            plt.show()

            # Save if requested
            if args.save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path(f"results/cv_model_visualization_all_{timestamp}")
                save_dir.mkdir(parents=True, exist_ok=True)

                fig_path = save_dir / f"cv_model_reconstruction_{dataset_name}.png"
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved: {fig_path}")

        # Summary for all datasets
        print(f"\n🎉 ALL DATASETS VISUALIZATION COMPLETE!")
        print("=" * 60)
        print(f"✅ Successfully processed: {len(successful_datasets)}/{len(datasets)} datasets")

        for dataset_name in successful_datasets:
            metrics = all_metrics[dataset_name]
            print(f"\n{dataset_name.upper()}:")
            print(f"   MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
            print(f"   Correlation: {metrics['correlation_mean']:.3f} ± {metrics['correlation_std']:.3f}")
            print(f"   CV Score: {metrics['cv_metadata']['best_score']:.6f}")

        # Create summary visualization if multiple datasets processed
        if len(successful_datasets) > 1:
            print(f"\n📊 Creating summary comparison...")
            summary_fig = create_all_datasets_summary(all_metrics, successful_datasets)
            plt.show()

            if args.save:
                summary_path = save_dir / "cv_model_summary_all_datasets.png"
                summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
                print(f"💾 Summary saved: {summary_path}")

        if args.save and successful_datasets:
            print(f"\n💾 All visualizations saved to: results/cv_model_visualization_all_{timestamp}/")

        print(f"\n🏆 All visualizations show reconstructions from ACTUAL cross-validation models!")

    else:
        # Single dataset processing (original logic)
        model, data, metadata = load_cv_saved_model(args.dataset, args.device)

        if model is None:
            print(f"\n❌ Failed to load CV model for {args.dataset}")
            print(f"💡 Run cross-validation first: python scripts/validate_cccv1.py --dataset {args.dataset}")
            return

        X_train, y_train, X_test, y_test = data

        # Generate reconstructions
        print(f"🎨 Generating reconstructions for {args.samples} samples...")
        targets, reconstructions, sample_indices = generate_reconstructions(
            model, X_test, y_test, args.samples
        )

        # Create visualization
        fig, metrics = create_cv_model_visualization(
            targets, reconstructions, args.dataset, sample_indices, metadata
        )

        print(f"📊 Reconstruction Quality (CV Model):")
        print(f"   MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
        print(f"   Correlation: {metrics['correlation_mean']:.3f} ± {metrics['correlation_std']:.3f}")
        print(f"   CV Score: {metadata['best_score']:.6f} (Fold {metadata['best_fold']})")

        # Show plot
        plt.show()

        # Save if requested
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(f"results/cv_model_visualization_{timestamp}")
            save_dir.mkdir(parents=True, exist_ok=True)

            fig_path = save_dir / f"cv_model_reconstruction_{args.dataset}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {fig_path}")

        print(f"\n✅ CV model visualization complete for {args.dataset}!")
        print(f"🏆 This shows reconstructions from the ACTUAL cross-validation model!")


if __name__ == "__main__":
    main()
