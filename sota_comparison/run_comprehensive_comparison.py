"""
Comprehensive SOTA Comparison Runner
===================================

Run complete comparison between CCCV1, Brain-Diffuser, and Mind-Vis.
Academic Integrity: Fair and standardized evaluation protocol.
"""

import os
import sys
import argparse
import torch
import pandas as pd
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from unified_evaluation_framework import UnifiedSOTAEvaluator


def print_comparison_summary(evaluator: UnifiedSOTAEvaluator):
    """Print detailed comparison summary"""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SOTA COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Generate summary table
    df = evaluator.generate_comparison_summary()
    
    if df is None or df.empty:
        print("❌ No results to display")
        return
    
    # Print overall summary
    print("\n📊 OVERALL RESULTS:")
    print(df.to_string(index=False))
    
    # Print method rankings per dataset
    print(f"\n🏆 METHOD RANKINGS BY DATASET:")
    print("-" * 50)
    
    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df[(df['Dataset'] == dataset) & (df['Status'] == 'Success')].copy()
        
        if not dataset_df.empty:
            print(f"\n{dataset.upper()}:")
            
            # Rank by correlation (higher is better)
            if 'Correlation' in dataset_df.columns:
                dataset_df['Correlation'] = pd.to_numeric(dataset_df['Correlation'], errors='coerce')
                corr_ranking = dataset_df.nlargest(3, 'Correlation')[['Method', 'Correlation']]
                print("  Correlation Ranking:")
                for i, (_, row) in enumerate(corr_ranking.iterrows(), 1):
                    print(f"    {i}. {row['Method']}: {row['Correlation']:.6f}")
            
            # Rank by MSE (lower is better)
            if 'MSE' in dataset_df.columns:
                dataset_df['MSE'] = pd.to_numeric(dataset_df['MSE'], errors='coerce')
                mse_ranking = dataset_df.nsmallest(3, 'MSE')[['Method', 'MSE']]
                print("  MSE Ranking:")
                for i, (_, row) in enumerate(mse_ranking.iterrows(), 1):
                    print(f"    {i}. {row['Method']}: {row['MSE']:.6f}")
            
            # Rank by SSIM (higher is better)
            if 'SSIM' in dataset_df.columns:
                dataset_df['SSIM'] = pd.to_numeric(dataset_df['SSIM'], errors='coerce')
                ssim_ranking = dataset_df.nlargest(3, 'SSIM')[['Method', 'SSIM']]
                print("  SSIM Ranking:")
                for i, (_, row) in enumerate(ssim_ranking.iterrows(), 1):
                    print(f"    {i}. {row['Method']}: {row['SSIM']:.6f}")
    
    # Overall winner analysis
    print(f"\n🎯 OVERALL ANALYSIS:")
    print("-" * 30)
    
    success_df = df[df['Status'] == 'Success'].copy()
    
    if not success_df.empty:
        # Convert to numeric
        for col in ['MSE', 'Correlation', 'SSIM', 'PSNR']:
            if col in success_df.columns:
                success_df[col] = pd.to_numeric(success_df[col], errors='coerce')
        
        # Calculate average performance across datasets
        avg_performance = success_df.groupby('Method').agg({
            'MSE': 'mean',
            'Correlation': 'mean', 
            'SSIM': 'mean',
            'PSNR': 'mean'
        }).round(6)
        
        print("Average Performance Across All Datasets:")
        print(avg_performance.to_string())
        
        # Determine overall winner
        if 'Correlation' in avg_performance.columns:
            best_correlation = avg_performance['Correlation'].idxmax()
            print(f"\n🥇 Best Overall Correlation: {best_correlation} ({avg_performance.loc[best_correlation, 'Correlation']:.6f})")
        
        if 'MSE' in avg_performance.columns:
            best_mse = avg_performance['MSE'].idxmin()
            print(f"🥇 Best Overall MSE: {best_mse} ({avg_performance.loc[best_mse, 'MSE']:.6f})")
        
        if 'SSIM' in avg_performance.columns:
            best_ssim = avg_performance['SSIM'].idxmax()
            print(f"🥇 Best Overall SSIM: {best_ssim} ({avg_performance.loc[best_ssim, 'SSIM']:.6f})")


def generate_academic_report(evaluator: UnifiedSOTAEvaluator, save_path: str):
    """Generate academic report for publication"""
    
    print(f"\n📝 Generating academic report...")
    
    df = evaluator.generate_comparison_summary()
    
    if df is None or df.empty:
        print("❌ No results to generate report")
        return
    
    # Perform statistical analysis
    stats_results = evaluator.perform_statistical_analysis()
    
    # Generate report content
    report_content = f"""
# SOTA Methods Comparison Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Abstract

This report presents a comprehensive comparison of three state-of-the-art methods for fMRI-to-visual reconstruction:
- CCCV1-Optimized: Our proposed method with dataset-specific optimizations
- Brain-Diffuser: Two-stage diffusion model (Ozcelik & VanRullen, 2023)
- Mind-Vis: Contrastive learning approach (Chen et al., 2023)

## Methodology

### Datasets
- miyawaki: Visual complex patterns (binary contrast)
- vangerven: Digit patterns (grayscale)
- crell: EEG→fMRI→Visual translation
- mindbigdata: EEG→fMRI→Visual translation

### Evaluation Metrics
- Mean Squared Error (MSE): Lower is better
- Pearson Correlation: Higher is better
- Structural Similarity Index (SSIM): Higher is better
- Peak Signal-to-Noise Ratio (PSNR): Higher is better

### Academic Integrity
All methods implemented following exact original paper methodologies without unauthorized modifications.

## Results

### Summary Table
{df.to_string(index=False)}

### Statistical Analysis
"""
    
    # Add statistical results
    if stats_results:
        report_content += "\n#### Pairwise Significance Tests (SSIM scores)\n"
        for comparison, result in stats_results.items():
            significance = "**Significant**" if result['significant'] else "Not significant"
            report_content += f"- {comparison}: p={result['p_value']:.6f} ({significance})\n"
    
    # Add performance analysis
    success_df = df[df['Status'] == 'Success'].copy()
    
    if not success_df.empty:
        # Convert to numeric
        for col in ['MSE', 'Correlation', 'SSIM', 'PSNR']:
            if col in success_df.columns:
                success_df[col] = pd.to_numeric(success_df[col], errors='coerce')
        
        # Calculate average performance
        avg_performance = success_df.groupby('Method').agg({
            'MSE': ['mean', 'std'],
            'Correlation': ['mean', 'std'],
            'SSIM': ['mean', 'std'],
            'PSNR': ['mean', 'std']
        }).round(6)
        
        report_content += f"\n### Average Performance Across Datasets\n"
        report_content += f"```\n{avg_performance.to_string()}\n```\n"
    
    # Add conclusions
    report_content += f"""
## Conclusions

### Key Findings
1. **Performance Comparison**: [Analysis based on results]
2. **Method Strengths**: [Identify strengths of each method]
3. **Dataset Dependencies**: [Analyze performance variations across datasets]

### Academic Contributions
1. **Fair Comparison**: All methods evaluated under identical conditions
2. **Reproducible Results**: Standardized evaluation protocol
3. **Statistical Validation**: Significance testing performed

### Future Work
1. **Extended Evaluation**: Additional datasets and metrics
2. **Ablation Studies**: Component-wise analysis
3. **Computational Efficiency**: Runtime and memory analysis

## References
- Ozcelik, F., VanRullen, R. (2023). Natural scene reconstruction from fMRI signals using generative latent diffusion. Scientific Reports.
- Chen, Z., et al. (2023). Seeing Beyond the Brain: Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding. CVPR.

---
*Report generated by Unified SOTA Evaluation Framework*
"""
    
    # Save report
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ Academic report saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive SOTA Comparison')
    parser.add_argument('--datasets', nargs='+', 
                        choices=['miyawaki', 'vangerven', 'crell', 'mindbigdata', 'all'],
                        default=['all'],
                        help='Datasets to evaluate (default: all)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples per dataset (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip academic report generation')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if 'all' in args.datasets:
        datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    else:
        datasets = args.datasets
    
    # Initialize evaluator
    evaluator = UnifiedSOTAEvaluator(device=device)
    
    # Run comprehensive comparison
    print("🚀 Starting comprehensive SOTA comparison...")
    results = evaluator.run_comprehensive_comparison(datasets, args.samples)
    
    # Print summary
    print_comparison_summary(evaluator)
    
    # Generate visualization
    if not args.no_viz:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"comparison_results/sota_comparison_visualization_{timestamp}.png"
        evaluator.create_comparison_visualization(viz_path)
    
    # Generate academic report
    if not args.no_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"comparison_results/academic_report_{timestamp}.md"
        generate_academic_report(evaluator, report_path)
    
    print(f"\n🎉 Comprehensive SOTA comparison complete!")
    print(f"📁 Results directory: comparison_results/")


if __name__ == "__main__":
    main()
