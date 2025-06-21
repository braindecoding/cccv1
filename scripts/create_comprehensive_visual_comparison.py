#!/usr/bin/env python3
"""
Comprehensive Visual Comparison
===============================

Creates publication-quality visual comparison showing:
1. Original stimulus images (REAL)
2. CCCV1 reconstructions (REAL from trained models)
3. Mind-Vis reconstructions (Mock based on performance characteristics)
4. Brain-Diffuser reconstructions (Mock based on performance characteristics)

Academic Integrity: Uses real CCCV1 data, clearly labeled mock data for SOTA.
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

def load_reconstruction_data(results_dir):
    """Load reconstruction data from generated results."""
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    all_data = {}
    
    for dataset in datasets:
        json_file = results_dir / f"{dataset}_reconstructions.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            for method in data:
                for key in data[method]:
                    if isinstance(data[method][key], list):
                        data[method][key] = np.array(data[method][key])
            
            all_data[dataset] = data
            print(f"✅ Loaded reconstruction data for {dataset}")
        else:
            print(f"❌ No reconstruction data found for {dataset}")
    
    return all_data

def calculate_quality_metrics(targets, reconstructions):
    """Calculate SSIM and correlation metrics."""
    
    ssim_scores = []
    corr_scores = []
    
    for target, recon in zip(targets, reconstructions):
        # SSIM
        ssim_score = ssim(target, recon, data_range=1.0)
        ssim_scores.append(ssim_score)
        
        # Correlation
        corr_score = np.corrcoef(target.flatten(), recon.flatten())[0, 1]
        if np.isnan(corr_score):
            corr_score = 0.0
        corr_scores.append(corr_score)
    
    return np.mean(ssim_scores), np.mean(corr_scores)

def create_comprehensive_comparison_figure(all_data):
    """Create comprehensive visual comparison figure."""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Comprehensive Visual Reconstruction Comparison\n(Original Stimuli vs Method Outputs)', 
                fontsize=20, fontweight='bold', y=0.95)
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    dataset_labels = ['Miyawaki (fMRI)', 'Vangerven (fMRI)', 'Crell (EEG→fMRI)', 'MindBigData (EEG→fMRI)']
    methods = ['Original', 'CCCV1', 'Mind-Vis', 'Brain-Diffuser']
    method_colors = ['black', '#2E86AB', '#A23B72', '#F18F01']
    
    # Set random seed for consistent sample selection
    np.random.seed(42)
    
    for dataset_idx, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
        if dataset not in all_data:
            continue
        
        data = all_data[dataset]
        
        # Get one sample for visualization
        sample_idx = 0  # Use first sample
        
        targets = data['CCCV1']['targets']
        cccv1_recons = data['CCCV1']['reconstructions']
        mindvis_recons = data['Mind-Vis']['reconstructions']
        braindiff_recons = data['Brain-Diffuser']['reconstructions']
        
        # Plot for this dataset (one row)
        for method_idx, (method, color) in enumerate(zip(methods, method_colors)):
            ax = fig.add_subplot(gs[dataset_idx, method_idx])
            
            if method == 'Original':
                img = targets[sample_idx]
                title_suffix = ""
            elif method == 'CCCV1':
                img = cccv1_recons[sample_idx]
                # Calculate quality metrics
                ssim_score, corr_score = calculate_quality_metrics(
                    [targets[sample_idx]], [img]
                )
                title_suffix = f"\nSSIM: {ssim_score:.3f}"
            elif method == 'Mind-Vis':
                img = mindvis_recons[sample_idx]
                ssim_score, corr_score = calculate_quality_metrics(
                    [targets[sample_idx]], [img]
                )
                title_suffix = f"\nSSIM: {ssim_score:.3f} (Mock)"
            elif method == 'Brain-Diffuser':
                img = braindiff_recons[sample_idx]
                ssim_score, corr_score = calculate_quality_metrics(
                    [targets[sample_idx]], [img]
                )
                title_suffix = f"\nSSIM: {ssim_score:.3f} (Mock)"
            
            # Display image
            im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{method}{title_suffix}', fontsize=11, fontweight='bold', color=color)
            ax.axis('off')
        
        # Add dataset label on the left
        fig.text(0.02, 0.85 - dataset_idx * 0.22, label, 
                rotation=90, fontsize=14, fontweight='bold', 
                verticalalignment='center', horizontalalignment='center')
    
    # Add legend and notes
    legend_text = """
    Data Sources:
    • Original: Real stimulus images from datasets
    • CCCV1: Real reconstructions from trained CV models
    • Mind-Vis: Mock reconstructions (SOTA not implemented)
    • Brain-Diffuser: Mock reconstructions (SOTA not implemented)
    
    SSIM: Structural Similarity Index (Higher = Better)
    Mock data based on reported performance characteristics
    """
    
    fig.text(0.02, 0.25, legend_text, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    return fig

def create_quality_metrics_summary(all_data):
    """Create quality metrics summary figure."""
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    dataset_labels = ['Miyawaki', 'Vangerven', 'Crell', 'MindBigData']
    methods = ['CCCV1', 'Mind-Vis', 'Brain-Diffuser']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Calculate metrics for all datasets
    ssim_data = {method: [] for method in methods}
    corr_data = {method: [] for method in methods}
    
    for dataset in datasets:
        if dataset not in all_data:
            for method in methods:
                ssim_data[method].append(0)
                corr_data[method].append(0)
            continue
        
        data = all_data[dataset]
        targets = data['CCCV1']['targets']
        
        for method in methods:
            if method in data:
                recons = data[method]['reconstructions']
                ssim_score, corr_score = calculate_quality_metrics(targets, recons)
                ssim_data[method].append(ssim_score)
                corr_data[method].append(corr_score)
            else:
                ssim_data[method].append(0)
                corr_data[method].append(0)
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Quantitative Quality Metrics Comparison\n(CCCV1: Real Data, SOTA: Mock Data)', 
                fontsize=16, fontweight='bold')
    
    x = np.arange(len(datasets))
    width = 0.25
    
    # SSIM comparison
    for i, (method, color) in enumerate(zip(methods, colors)):
        bars = ax1.bar(x + i*width, ssim_data[method], width, label=method, color=color, alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, ssim_data[method]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Dataset', fontweight='bold')
    ax1.set_ylabel('SSIM Score', fontweight='bold')
    ax1.set_title('Structural Similarity Index (SSIM)', fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(dataset_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Correlation comparison
    for i, (method, color) in enumerate(zip(methods, colors)):
        bars = ax2.bar(x + i*width, corr_data[method], width, label=method, color=color, alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, corr_data[method]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Dataset', fontweight='bold')
    ax2.set_ylabel('Correlation Score', fontweight='bold')
    ax2.set_title('Pearson Correlation Coefficient', fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(dataset_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.0)
    
    plt.tight_layout()
    return fig

def main():
    """Generate comprehensive visual comparison."""
    
    print("🎨 CREATING COMPREHENSIVE VISUAL COMPARISON")
    print("=" * 60)
    
    # Find latest reconstruction results
    results_dirs = list(Path("results").glob("all_model_reconstructions_*"))
    if not results_dirs:
        print("❌ No reconstruction results found!")
        print("💡 Run generate_all_reconstructions.py first")
        return
    
    latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using results from: {latest_dir}")
    
    # Load reconstruction data
    all_data = load_reconstruction_data(latest_dir)
    if not all_data:
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/comprehensive_visual_comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    
    # Create comprehensive comparison
    print("\n🖼️ Creating comprehensive visual comparison...")
    fig1 = create_comprehensive_comparison_figure(all_data)
    
    # Save comprehensive comparison
    fig1_png = output_dir / "comprehensive_visual_comparison.png"
    fig1_svg = output_dir / "comprehensive_visual_comparison.svg"
    
    fig1.savefig(fig1_png, dpi=300, bbox_inches='tight')
    fig1.savefig(fig1_svg, format='svg', bbox_inches='tight')
    
    print(f"💾 Saved PNG: {fig1_png}")
    print(f"💾 Saved SVG: {fig1_svg}")
    
    # Create quality metrics summary
    print("\n📊 Creating quality metrics summary...")
    fig2 = create_quality_metrics_summary(all_data)
    
    # Save quality metrics
    fig2_png = output_dir / "quality_metrics_summary.png"
    fig2_svg = output_dir / "quality_metrics_summary.svg"
    
    fig2.savefig(fig2_png, dpi=300, bbox_inches='tight')
    fig2.savefig(fig2_svg, format='svg', bbox_inches='tight')
    
    print(f"💾 Saved PNG: {fig2_png}")
    print(f"💾 Saved SVG: {fig2_svg}")
    
    # Show plots
    plt.show()
    
    print(f"\n✅ Comprehensive visual comparison complete!")
    print(f"🎨 Generated publication-quality visual comparison")
    print(f"📊 Created quantitative quality metrics")
    print(f"🏆 CCCV1 uses REAL data, SOTA uses labeled mock data")
    print(f"📁 All files saved to: {output_dir}")

if __name__ == "__main__":
    main()
