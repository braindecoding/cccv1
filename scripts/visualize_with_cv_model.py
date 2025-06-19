"""
Visualize Reconstructions Using Cross-Validation Trained Model
=============================================================

Script to visualize reconstructions using the same model architecture and training
procedure as the cross-validation, but trained on the full training set for better
visualization quality.

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


def load_optimal_config(dataset_name):
    """Load optimal configuration for dataset."""
    try:
        with open('configs/optimal_configs.json', 'r') as f:
            configs = json.load(f)
        return configs.get(dataset_name, {})
    except:
        # Default config if file not found
        return {
            'architecture': {'dropout_encoder': 0.05, 'dropout_decoder': 0.02},
            'training': {'lr': 0.001, 'batch_size': 16, 'epochs': 100}
        }


def train_cv_style_model(model, X_train, y_train, config, device='cuda'):
    """Train model using the same procedure as cross-validation."""
    
    print("🔄 Training model with CV-style procedure...")
    
    # Training configuration
    training_config = config.get('training', {})
    lr = training_config.get('lr', 0.001)
    batch_size = training_config.get('batch_size', 16)
    epochs = training_config.get('epochs', 100)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1)
    
    # Create data loaders (use 80% for training, 20% for validation)
    n_train = int(0.8 * len(X_train))
    indices = torch.randperm(len(X_train))
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train_fold = X_train[train_indices]
    y_train_fold = y_train[train_indices]
    X_val_fold = X_train[val_indices]
    y_val_fold = y_train[val_indices]
    
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    val_dataset = TensorDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    model.train()
    
    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output, _ = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output, _ = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item()
        
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
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        model.train()
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"✅ Training complete. Best validation loss: {best_val_loss:.6f}")
    
    return model


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


def create_publication_visualization(targets, reconstructions, dataset_name, sample_indices, metrics):
    """Create publication-quality visualization."""
    
    n_samples = len(targets)
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    
    # Main grid for images
    gs_main = fig.add_gridspec(3, n_samples, height_ratios=[1, 1, 1], 
                              left=0.05, right=0.7, top=0.85, bottom=0.15,
                              hspace=0.3, wspace=0.1)
    
    # Side panel for metrics and analysis
    gs_side = fig.add_gridspec(3, 1, left=0.75, right=0.98, top=0.85, bottom=0.15)
    
    # Dataset configuration
    dataset_configs = {
        'miyawaki': {'title': 'Miyawaki Visual Patterns', 'cmap': 'gray', 'description': 'Complex geometric shapes from fMRI signals'},
        'vangerven': {'title': 'Vangerven Handwritten Digits', 'cmap': 'gray', 'description': 'Digit reconstruction from visual cortex'},
        'mindbigdata': {'title': 'MindBigData Cross-Modal', 'cmap': 'viridis', 'description': 'EEG→fMRI→Visual translation'},
        'crell': {'title': 'Crell Cross-Modal', 'cmap': 'viridis', 'description': 'EEG→fMRI→Visual translation'}
    }
    
    config = dataset_configs.get(dataset_name, {'title': dataset_name.title(), 'cmap': 'gray', 'description': 'Neural decoding'})
    
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
        ax_recon.set_title('Reconstruction', fontsize=11, fontweight='bold')
        ax_recon.axis('off')
        
        # Difference
        ax_diff = fig.add_subplot(gs_main[2, i])
        diff_img = np.abs(target_img - recon_img)
        ax_diff.imshow(diff_img, cmap='Reds')
        ax_diff.set_title('|Difference|', fontsize=11, fontweight='bold')
        ax_diff.axis('off')
        
        # Add sample metrics
        sample_mse = np.mean((target_img - recon_img) ** 2)
        target_flat = target_img.flatten()
        recon_flat = recon_img.flatten()
        if np.std(target_flat) > 0 and np.std(recon_flat) > 0:
            sample_corr = np.corrcoef(target_flat, recon_flat)[0, 1]
        else:
            sample_corr = 0
        
        ax_diff.text(0.5, -0.15, f'MSE: {sample_mse:.4f}\nCorr: {sample_corr:.3f}', 
                    transform=ax_diff.transAxes, ha='center', va='top', fontsize=9)
    
    # Row labels
    fig.text(0.02, 0.7, 'Target', rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Reconstruction', rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.3, 'Difference', rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Side panel - Dataset info
    ax_info = fig.add_subplot(gs_side[0, 0])
    ax_info.axis('off')
    
    info_text = f"""
    {config['title']}
    
    {config['description']}
    
    📊 Reconstruction Quality:
    • MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}
    • Correlation: {metrics['correlation_mean']:.3f} ± {metrics['correlation_std']:.3f}
    • Samples: {n_samples}
    
    🎯 Model: CortexFlow-CLIP-CNN V1
    🔬 Training: CV-style procedure
    """
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Side panel - Metrics distribution
    ax_metrics = fig.add_subplot(gs_side[1, 0])
    
    sample_mses = []
    sample_corrs = []
    for i in range(n_samples):
        target_img = targets[i].squeeze()
        recon_img = reconstructions[i].squeeze()
        sample_mse = np.mean((target_img - recon_img) ** 2)
        sample_mses.append(sample_mse)
        
        target_flat = target_img.flatten()
        recon_flat = recon_img.flatten()
        if np.std(target_flat) > 0 and np.std(recon_flat) > 0:
            sample_corr = np.corrcoef(target_flat, recon_flat)[0, 1]
        else:
            sample_corr = 0
        sample_corrs.append(sample_corr)
    
    x = np.arange(n_samples)
    width = 0.35
    
    bars1 = ax_metrics.bar(x - width/2, sample_mses, width, label='MSE', alpha=0.7, color='skyblue')
    ax_metrics_twin = ax_metrics.twinx()
    bars2 = ax_metrics_twin.bar(x + width/2, sample_corrs, width, label='Correlation', alpha=0.7, color='lightcoral')
    
    ax_metrics.set_xlabel('Sample')
    ax_metrics.set_ylabel('MSE', color='blue')
    ax_metrics_twin.set_ylabel('Correlation', color='red')
    ax_metrics.set_title('Per-Sample Metrics')
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels([f'#{idx}' for idx in sample_indices])
    
    # Side panel - Quality assessment
    ax_quality = fig.add_subplot(gs_side[2, 0])
    ax_quality.axis('off')
    
    # Determine quality level
    if metrics['correlation_mean'] > 0.8:
        quality_level = "Excellent"
        quality_color = "green"
    elif metrics['correlation_mean'] > 0.6:
        quality_level = "Good"
        quality_color = "orange"
    elif metrics['correlation_mean'] > 0.4:
        quality_level = "Moderate"
        quality_color = "yellow"
    else:
        quality_level = "Poor"
        quality_color = "red"
    
    quality_text = f"""
    🏆 Reconstruction Quality: {quality_level}
    
    📈 Performance Analysis:
    • Best sample: #{sample_indices[np.argmin(sample_mses)]} (MSE: {np.min(sample_mses):.4f})
    • Worst sample: #{sample_indices[np.argmax(sample_mses)]} (MSE: {np.max(sample_mses):.4f})
    • Consistency: {np.std(sample_mses)/np.mean(sample_mses)*100:.1f}% CV
    
    ✨ This visualization uses the same model architecture
    and training procedure as the cross-validation evaluation.
    """
    
    ax_quality.text(0.05, 0.95, quality_text, transform=ax_quality.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor=quality_color, alpha=0.3))
    
    # Overall title
    fig.suptitle(f'{config["title"]} - Neural Decoding Reconstruction\nUsing Cross-Validation Model Architecture', 
                fontsize=16, fontweight='bold')
    
    return fig


def visualize_cv_model(dataset_name, device='cuda', n_samples=6, epochs=100):
    """Visualize reconstructions using CV-style trained model."""
    
    print(f"\n🎨 VISUALIZING WITH CV-STYLE MODEL: {dataset_name.upper()}")
    print("=" * 60)
    
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
    
    # Load optimal configuration
    config = load_optimal_config(dataset_name)
    print(f"✅ Configuration loaded for {dataset_name}")
    
    # Create model with same architecture as CV
    try:
        model = CortexFlowCLIPCNNV1Optimized(input_dim=input_dim, device=device)
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None, None
    
    # Train model using CV-style procedure
    model = train_cv_style_model(model, X_train, y_train, config, device)
    
    # Generate reconstructions
    print(f"🎨 Generating reconstructions for {n_samples} samples...")
    targets, reconstructions, sample_indices = generate_reconstructions(model, X_test, y_test, n_samples)
    
    # Calculate metrics
    targets_flat = targets.reshape(len(targets), -1)
    recons_flat = reconstructions.reshape(len(reconstructions), -1)
    mse_per_sample = np.mean((targets_flat - recons_flat) ** 2, axis=1)
    
    correlations = []
    for i in range(len(targets)):
        target_flat = targets_flat[i]
        recon_flat = recons_flat[i]
        if np.std(target_flat) > 0 and np.std(recon_flat) > 0:
            corr = np.corrcoef(target_flat, recon_flat)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        else:
            correlations.append(0)
    
    metrics = {
        'mse_mean': np.mean(mse_per_sample),
        'mse_std': np.std(mse_per_sample),
        'correlation_mean': np.mean(correlations),
        'correlation_std': np.std(correlations)
    }
    
    print(f"📊 Reconstruction Quality (CV-style model):")
    print(f"   MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"   Correlation: {metrics['correlation_mean']:.3f} ± {metrics['correlation_std']:.3f}")
    
    # Create visualization
    fig = create_publication_visualization(targets, reconstructions, dataset_name, sample_indices, metrics)
    
    return fig, metrics


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='Visualize reconstructions using CV-style model')
    parser.add_argument('--dataset', type=str, default='miyawaki',
                       choices=['miyawaki', 'vangerven', 'mindbigdata', 'crell'],
                       help='Dataset to visualize')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
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
    
    print("🎨 NEURAL DECODING VISUALIZATION WITH CV-STYLE MODEL")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📈 Samples: {args.samples}")
    print(f"🖥️ Device: {args.device}")
    print(f"🔬 Method: Same architecture and training as cross-validation")
    
    # Visualize
    fig, metrics = visualize_cv_model(args.dataset, args.device, args.samples, args.epochs)
    
    if fig is not None:
        # Show plot
        plt.show()
        
        # Save if requested
        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(f"results/cv_style_visualization_{timestamp}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            fig_path = save_dir / f"cv_style_reconstruction_{args.dataset}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {fig_path}")
        
        print(f"\n✅ CV-style visualization complete for {args.dataset}!")
        print(f"🎯 This uses the SAME model architecture and training procedure as cross-validation")
    else:
        print(f"\n❌ Visualization failed for {args.dataset}")


if __name__ == "__main__":
    main()
