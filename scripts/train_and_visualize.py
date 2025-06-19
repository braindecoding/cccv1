"""
Train Model and Visualize Reconstructions
=========================================

Script to train a model and immediately visualize the reconstructions.
This provides both quantitative and qualitative assessment.

Author: [Your Name]
Date: 2025-06-20
Version: 1.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def train_model_quick(model, X_train, y_train, X_test, y_test, dataset_name, device='cuda', epochs=50):
    """Quick training for visualization purposes."""
    
    print(f"🔄 Quick training for {dataset_name} ({epochs} epochs)...")
    
    # Training configuration
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    criterion = nn.MSELoss()
    
    # Create data loader
    batch_size = min(16, len(X_train) // 4)
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_output, _ = model(X_test)
                test_loss = criterion(test_output, y_test).item()
            model.train()
            
            print(f"   Epoch {epoch+1:2d}: Train={avg_loss:.6f}, Test={test_loss:.6f}")
    
    print(f"✅ Training complete for {dataset_name}")
    return train_losses


def generate_reconstructions(model, X_test, y_test, n_samples=6):
    """Generate reconstructions for visualization."""
    
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


def create_reconstruction_visualization(targets, reconstructions, dataset_name, sample_indices):
    """Create comprehensive reconstruction visualization."""
    
    n_samples = len(targets)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Main comparison grid
    gs_main = fig.add_gridspec(3, n_samples, height_ratios=[1, 1, 1], 
                              left=0.05, right=0.75, top=0.9, bottom=0.1)
    
    # Side panel for metrics
    gs_side = fig.add_gridspec(2, 1, left=0.8, right=0.95, top=0.9, bottom=0.1)
    
    # Dataset-specific configuration
    dataset_configs = {
        'miyawaki': {'title': 'Miyawaki Visual Patterns', 'cmap': 'gray'},
        'vangerven': {'title': 'Vangerven Handwritten Digits', 'cmap': 'gray'},
        'mindbigdata': {'title': 'MindBigData Cross-Modal', 'cmap': 'viridis'},
        'crell': {'title': 'Crell Cross-Modal', 'cmap': 'viridis'}
    }
    
    config = dataset_configs.get(dataset_name, {'title': dataset_name.title(), 'cmap': 'gray'})
    
    # Calculate metrics
    mse_per_sample = []
    correlations = []
    
    for i in range(n_samples):
        # Create subplots for this sample
        ax_target = fig.add_subplot(gs_main[0, i])
        ax_recon = fig.add_subplot(gs_main[1, i])
        ax_diff = fig.add_subplot(gs_main[2, i])
        
        # Get images
        target_img = targets[i].squeeze()
        recon_img = reconstructions[i].squeeze()
        diff_img = np.abs(target_img - recon_img)
        
        # Display images
        ax_target.imshow(target_img, cmap=config['cmap'])
        ax_target.set_title(f'Target #{sample_indices[i]}', fontsize=10)
        ax_target.axis('off')
        
        ax_recon.imshow(recon_img, cmap=config['cmap'])
        ax_recon.set_title(f'Reconstruction', fontsize=10)
        ax_recon.axis('off')
        
        ax_diff.imshow(diff_img, cmap='Reds')
        ax_diff.set_title(f'|Difference|', fontsize=10)
        ax_diff.axis('off')
        
        # Calculate metrics for this sample
        mse = np.mean((target_img - recon_img) ** 2)
        mse_per_sample.append(mse)
        
        # Correlation
        target_flat = target_img.flatten()
        recon_flat = recon_img.flatten()
        if np.std(target_flat) > 0 and np.std(recon_flat) > 0:
            corr = np.corrcoef(target_flat, recon_flat)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)
        
        # Add metrics text
        ax_diff.text(0.5, -0.15, f'MSE: {mse:.4f}\nCorr: {correlations[-1]:.3f}', 
                    transform=ax_diff.transAxes, ha='center', fontsize=8)
    
    # Add row labels
    if n_samples > 0:
        fig.text(0.02, 0.75, 'Target', rotation=90, ha='center', va='center', 
                fontsize=12, weight='bold')
        fig.text(0.02, 0.5, 'Reconstruction', rotation=90, ha='center', va='center', 
                fontsize=12, weight='bold')
        fig.text(0.02, 0.25, 'Difference', rotation=90, ha='center', va='center', 
                fontsize=12, weight='bold')
    
    # Side panel - Metrics summary
    ax_metrics1 = fig.add_subplot(gs_side[0, 0])
    ax_metrics2 = fig.add_subplot(gs_side[1, 0])
    
    # MSE distribution
    ax_metrics1.bar(range(len(mse_per_sample)), mse_per_sample, alpha=0.7, color='skyblue')
    ax_metrics1.set_title('MSE per Sample')
    ax_metrics1.set_ylabel('MSE')
    ax_metrics1.set_xlabel('Sample')
    ax_metrics1.axhline(np.mean(mse_per_sample), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(mse_per_sample):.4f}')
    ax_metrics1.legend()
    
    # Correlation distribution
    ax_metrics2.bar(range(len(correlations)), correlations, alpha=0.7, color='lightcoral')
    ax_metrics2.set_title('Correlation per Sample')
    ax_metrics2.set_ylabel('Correlation')
    ax_metrics2.set_xlabel('Sample')
    ax_metrics2.set_ylim(0, 1)
    ax_metrics2.axhline(np.mean(correlations), color='red', linestyle='--',
                       label=f'Mean: {np.mean(correlations):.3f}')
    ax_metrics2.legend()
    
    # Overall title
    fig.suptitle(f'{config["title"]} - Neural Decoding Reconstruction', 
                fontsize=16, weight='bold')
    
    # Summary statistics
    summary_text = f"""
    Dataset: {dataset_name.upper()}
    Samples: {n_samples}
    
    MSE: {np.mean(mse_per_sample):.6f} ± {np.std(mse_per_sample):.6f}
    Correlation: {np.mean(correlations):.3f} ± {np.std(correlations):.3f}
    
    Best Sample: #{sample_indices[np.argmin(mse_per_sample)]} (MSE: {np.min(mse_per_sample):.4f})
    Worst Sample: #{sample_indices[np.argmax(mse_per_sample)]} (MSE: {np.max(mse_per_sample):.4f})
    """
    
    fig.text(0.8, 0.05, summary_text, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    return fig, {
        'mse_mean': np.mean(mse_per_sample),
        'mse_std': np.std(mse_per_sample),
        'mse_per_sample': mse_per_sample,
        'correlation_mean': np.mean(correlations),
        'correlation_std': np.std(correlations),
        'correlations': correlations
    }


def train_and_visualize_dataset(dataset_name, device='cuda', epochs=50, n_samples=6):
    """Train model and visualize reconstructions for a dataset."""
    
    print(f"\n🎨 TRAIN & VISUALIZE: {dataset_name.upper()}")
    print("=" * 50)
    
    # Load dataset
    try:
        result = load_dataset_gpu_optimized(dataset_name, device)
        if len(result) == 6:
            X_train, y_train, X_test, y_test, input_dim, metadata = result
        else:
            X_train, y_train, X_test, y_test, input_dim = result
        print(f"✅ Dataset loaded: Train={len(X_train)}, Test={len(X_test)}, Input_dim={input_dim}")
    except Exception as e:
        print(f"❌ Failed to load dataset {dataset_name}: {e}")
        return None, None
    
    # Create model
    try:
        model = CortexFlowCLIPCNNV1Optimized(input_dim=input_dim, device=device)
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None, None
    
    # Train model
    train_losses = train_model_quick(model, X_train, y_train, X_test, y_test, 
                                   dataset_name, device, epochs)
    
    # Generate reconstructions
    print(f"🎨 Generating reconstructions for {n_samples} samples...")
    targets, reconstructions, sample_indices = generate_reconstructions(
        model, X_test, y_test, n_samples
    )
    
    # Create visualization
    fig, metrics = create_reconstruction_visualization(
        targets, reconstructions, dataset_name, sample_indices
    )
    
    print(f"📊 Reconstruction Quality:")
    print(f"   MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"   Correlation: {metrics['correlation_mean']:.3f} ± {metrics['correlation_std']:.3f}")
    
    return fig, metrics


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Train and visualize neural decoding reconstructions')
    parser.add_argument('--dataset', type=str, default='miyawaki',
                       choices=['miyawaki', 'vangerven', 'mindbigdata', 'crell'],
                       help='Dataset to train and visualize')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=6,
                       help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--save', action='store_true',
                       help='Save visualizations to file')
    
    args = parser.parse_args()
    
    # Set up visualization style
    set_visualization_style()
    
    print("🎨 TRAIN & VISUALIZE NEURAL DECODING RECONSTRUCTIONS")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Samples: {args.samples}")
    print(f"🖥️ Device: {args.device}")
    
    # Train and visualize
    fig, metrics = train_and_visualize_dataset(
        args.dataset, args.device, args.epochs, args.samples
    )
    
    if fig is not None:
        # Show plot
        plt.show()
        
        # Save if requested
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(f"results/train_visualize_{timestamp}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            fig_path = save_dir / f"train_visualize_{args.dataset}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {fig_path}")
        
        print(f"\n✅ Train & visualize complete for {args.dataset}!")
    else:
        print(f"\n❌ Train & visualize failed for {args.dataset}")


if __name__ == "__main__":
    main()
