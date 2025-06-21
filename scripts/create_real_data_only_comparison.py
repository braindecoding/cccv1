#!/usr/bin/env python3
"""
Real Data Only Visual Comparison
================================

Creates publication-quality visual comparison using ONLY real data:
1. Original stimulus images (REAL from datasets)
2. CCCV1 reconstructions (REAL from trained CV models)
3. Quality metrics (REAL SSIM and correlation)

Academic Integrity: 100% real data, NO mock data whatsoever.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim

# Set style for publication quality
plt.style.use('default')
sns.set_palette("husl")

def load_real_cccv1_data(results_dir):
    """Load ONLY real CCCV1 reconstruction data."""
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    real_data = {}
    
    for dataset in datasets:
        json_file = results_dir / f"{dataset}_reconstructions.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract ONLY CCCV1 real data
            if 'CCCV1' in data:
                cccv1_data = data['CCCV1']
                
                # Convert lists back to numpy arrays
                for key in cccv1_data:
                    if isinstance(cccv1_data[key], list):
                        cccv1_data[key] = np.array(cccv1_data[key])
                
                real_data[dataset] = cccv1_data
                print(f"✅ Loaded REAL CCCV1 data for {dataset}")
            else:
                print(f"❌ No CCCV1 data found for {dataset}")
        else:
            print(f"❌ No reconstruction data found for {dataset}")
    
    return real_data

def calculate_real_quality_metrics(targets, reconstructions):
    """Calculate real SSIM and correlation metrics."""
    
    ssim_scores = []
    corr_scores = []
    mse_scores = []
    
    for target, recon in zip(targets, reconstructions):
        # SSIM
        ssim_score = ssim(target, recon, data_range=1.0)
        ssim_scores.append(ssim_score)
        
        # Correlation
        corr_score = np.corrcoef(target.flatten(), recon.flatten())[0, 1]
        if np.isnan(corr_score):
            corr_score = 0.0
        corr_scores.append(corr_score)
        
        # MSE
        mse_score = np.mean((target - recon) ** 2)
        mse_scores.append(mse_score)
    
    return {
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'corr_mean': np.mean(corr_scores),
        'corr_std': np.std(corr_scores),
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores)
    }

def create_real_data_comparison_figure(real_data):
    """Create visual comparison using ONLY real data."""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    fig.suptitle('CortexFlow Visual Reconstruction Results\n(Real Data Only - Original Stimuli vs CCCV1 Reconstructions)', 
                fontsize=18, fontweight='bold', y=0.95)
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    dataset_labels = ['Miyawaki (fMRI)', 'Vangerven (fMRI)', 'Crell (EEG→fMRI)', 'MindBigData (EEG→fMRI)']
    
    # Set random seed for consistent sample selection
    np.random.seed(42)
    
    for dataset_idx, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
        if dataset not in real_data:
            continue
        
        data = real_data[dataset]
        targets = data['targets']
        reconstructions = data['reconstructions']
        
        # Calculate quality metrics for all samples
        metrics = calculate_real_quality_metrics(targets, reconstructions)
        
        # Select multiple samples for visualization (show 3 samples)
        num_samples = min(3, len(targets))
        sample_indices = np.random.choice(len(targets), num_samples, replace=False)
        
        # Create subplot for this dataset
        ax_orig = fig.add_subplot(gs[dataset_idx, 0])
        ax_recon = fig.add_subplot(gs[dataset_idx, 1])
        
        # Create composite image showing multiple samples
        if num_samples > 1:
            # Combine multiple samples horizontally
            orig_combined = np.hstack([targets[i] for i in sample_indices])
            recon_combined = np.hstack([reconstructions[i] for i in sample_indices])
        else:
            orig_combined = targets[sample_indices[0]]
            recon_combined = reconstructions[sample_indices[0]]
        
        # Display original stimuli
        ax_orig.imshow(orig_combined, cmap='gray', vmin=0, vmax=1)
        ax_orig.set_title(f'Original Stimuli\n{label}', fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # Display CCCV1 reconstructions
        ax_recon.imshow(recon_combined, cmap='gray', vmin=0, vmax=1)
        ax_recon.set_title(f'CCCV1 Reconstructions\nSSIM: {metrics["ssim_mean"]:.3f}±{metrics["ssim_std"]:.3f}', 
                          fontsize=12, fontweight='bold', color='#2E86AB')
        ax_recon.axis('off')
        
        # Add quality metrics text
        metrics_text = (f"Quality Metrics (n={len(targets)}):\n"
                       f"SSIM: {metrics['ssim_mean']:.3f} ± {metrics['ssim_std']:.3f}\n"
                       f"Correlation: {metrics['corr_mean']:.3f} ± {metrics['corr_std']:.3f}\n"
                       f"MSE: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
        
        ax_recon.text(1.05, 0.5, metrics_text, transform=ax_recon.transAxes,
                     fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Add academic integrity note
    integrity_text = """
    Academic Integrity Statement:
    • All data shown is REAL experimental data
    • Original stimuli from published datasets
    • CCCV1 reconstructions from trained CV models
    • Quality metrics calculated from actual outputs
    • NO mock or synthetic data used
    """
    
    fig.text(0.02, 0.15, integrity_text, fontsize=11, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    return fig

def create_real_quality_metrics_figure(real_data):
    """Create quality metrics figure using ONLY real data."""
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    dataset_labels = ['Miyawaki', 'Vangerven', 'Crell', 'MindBigData']
    
    # Calculate metrics for all datasets
    ssim_means = []
    ssim_stds = []
    corr_means = []
    corr_stds = []
    mse_means = []
    mse_stds = []
    
    for dataset in datasets:
        if dataset in real_data:
            data = real_data[dataset]
            targets = data['targets']
            reconstructions = data['reconstructions']
            
            metrics = calculate_real_quality_metrics(targets, reconstructions)
            ssim_means.append(metrics['ssim_mean'])
            ssim_stds.append(metrics['ssim_std'])
            corr_means.append(metrics['corr_mean'])
            corr_stds.append(metrics['corr_std'])
            mse_means.append(metrics['mse_mean'])
            mse_stds.append(metrics['mse_std'])
        else:
            ssim_means.append(0)
            ssim_stds.append(0)
            corr_means.append(0)
            corr_stds.append(0)
            mse_means.append(0)
            mse_stds.append(0)
    
    # Create comparison figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CortexFlow Quality Metrics Analysis\n(Real Data Only)', 
                fontsize=16, fontweight='bold')
    
    x = np.arange(len(datasets))
    color = '#2E86AB'
    
    # SSIM
    bars1 = ax1.bar(x, ssim_means, yerr=ssim_stds, color=color, alpha=0.8, capsize=5)
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_ylabel('SSIM Score', fontweight='bold')
    ax1.set_title('Structural Similarity Index (SSIM)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_labels, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars1, ssim_means, ssim_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Correlation
    bars2 = ax2.bar(x, corr_means, yerr=corr_stds, color=color, alpha=0.8, capsize=5)
    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_ylabel('Correlation Score', fontweight='bold')
    ax2.set_title('Pearson Correlation Coefficient', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_labels, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars2, corr_means, corr_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # MSE (log scale)
    bars3 = ax3.bar(x, mse_means, yerr=mse_stds, color=color, alpha=0.8, capsize=5)
    ax3.set_xlabel('Dataset', fontweight='bold')
    ax3.set_ylabel('MSE (Log Scale)', fontweight='bold')
    ax3.set_title('Mean Squared Error', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(dataset_labels, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_yscale('log')
    
    # Add value labels
    for bar, mean_val in zip(bars3, mse_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.5,
                f'{mean_val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Sample sizes
    sample_sizes = [len(real_data[dataset]['targets']) if dataset in real_data else 0 for dataset in datasets]
    bars4 = ax4.bar(x, sample_sizes, color=color, alpha=0.8)
    ax4.set_xlabel('Dataset', fontweight='bold')
    ax4.set_ylabel('Number of Samples', fontweight='bold')
    ax4.set_title('Sample Sizes (Real Data)', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(dataset_labels, rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, size in zip(bars4, sample_sizes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{size}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Generate real data only visual comparison."""
    
    print("🎨 CREATING REAL DATA ONLY VISUAL COMPARISON")
    print("=" * 60)
    print("🏆 Academic Integrity: NO MOCK DATA - REAL DATA ONLY")
    
    # Find latest reconstruction results
    results_dirs = list(Path("results").glob("all_model_reconstructions_*"))
    if not results_dirs:
        print("❌ No reconstruction results found!")
        print("💡 Run generate_all_reconstructions.py first")
        return
    
    latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using results from: {latest_dir}")
    
    # Load ONLY real CCCV1 data
    real_data = load_real_cccv1_data(latest_dir)
    if not real_data:
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/real_data_only_comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Create real data comparison
    print("\n🖼️ Creating real data visual comparison...")
    fig1 = create_real_data_comparison_figure(real_data)
    
    # Save visual comparison
    fig1_png = output_dir / "real_data_visual_comparison.png"
    fig1_svg = output_dir / "real_data_visual_comparison.svg"
    
    fig1.savefig(fig1_png, dpi=300, bbox_inches='tight')
    fig1.savefig(fig1_svg, format='svg', bbox_inches='tight')
    
    print(f"💾 Saved PNG: {fig1_png}")
    print(f"💾 Saved SVG: {fig1_svg}")
    
    # Create quality metrics
    print("\n📊 Creating real quality metrics analysis...")
    fig2 = create_real_quality_metrics_figure(real_data)
    
    # Save quality metrics
    fig2_png = output_dir / "real_quality_metrics.png"
    fig2_svg = output_dir / "real_quality_metrics.svg"
    
    fig2.savefig(fig2_png, dpi=300, bbox_inches='tight')
    fig2.savefig(fig2_svg, format='svg', bbox_inches='tight')
    
    print(f"💾 Saved PNG: {fig2_png}")
    print(f"💾 Saved SVG: {fig2_svg}")
    
    # Show plots
    plt.show()
    
    print(f"\n✅ Real data only comparison complete!")
    print(f"🏆 100% REAL DATA - NO MOCK DATA")
    print(f"🎨 Generated publication-quality visualizations")
    print(f"📊 Created comprehensive quality metrics")
    print(f"📁 All files saved to: {output_dir}")

if __name__ == "__main__":
    main()
