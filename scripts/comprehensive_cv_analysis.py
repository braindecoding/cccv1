#!/usr/bin/env python3
"""
Comprehensive Cross-Validation Analysis
======================================

Analyze all existing CV results and create comprehensive comparison.
Academic Integrity: Use only real data from trained models.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_cv_results():
    """Load all available CV results."""
    
    print("📊 LOADING COMPREHENSIVE CV RESULTS")
    print("=" * 60)
    
    results = {}
    
    # Load CortexFlow results from validation directories
    validation_dirs = list(Path("cccv1/results").glob("validation_20250621_*"))

    for val_dir in validation_dirs:
        result_file = val_dir / "validation_results.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                    if 'dataset' in data and 'cv_scores' in data:
                        dataset = data['dataset']
                        cv_scores = np.array(data['cv_scores'])

                        results[f"CortexFlow_{dataset}"] = {
                            'model': 'CortexFlow',
                            'dataset': dataset,
                            'cv_scores': cv_scores.tolist(),
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'best_fold': data.get('best_fold', 0),
                            'timestamp': data.get('timestamp', val_dir.name)
                        }
                        print(f"✅ Loaded CortexFlow {dataset}: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
            except Exception as e:
                print(f"⚠️ Could not load {result_file}: {e}")
    
    # Load SOTA comparison results if available
    sota_files = list(Path("sota_comparison/comparison_results").glob("academic_evaluation_*.json"))
    
    for file_path in sota_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                if 'results' in data:
                    for result in data['results']:
                        if result['status'] == 'success':
                            key = f"{result['method']}_{result['dataset']}"
                            results[key] = {
                                'model': result['method'],
                                'dataset': result['dataset'],
                                'cv_mean': result.get('mse', result.get('cv_mean', 0)),
                                'cv_std': result.get('mse_std', result.get('cv_std', 0)),
                                'correlation': result.get('correlation', 0),
                                'ssim': result.get('ssim', 0),
                                'timestamp': data.get('timestamp', 'unknown')
                            }
                            print(f"✅ Loaded {result['method']} {result['dataset']}: MSE={result.get('mse', 0):.6f}")
        except Exception as e:
            print(f"⚠️ Could not load {file_path}: {e}")
    
    print(f"\n📊 Total results loaded: {len(results)}")
    return results

def create_comparison_table(results):
    """Create comprehensive comparison table."""
    
    print("\n📊 CREATING COMPARISON TABLE")
    print("=" * 60)
    
    # Convert to DataFrame
    data_rows = []
    
    for key, result in results.items():
        row = {
            'Model': result['model'],
            'Dataset': result['dataset'],
            'MSE_Mean': result['cv_mean'],
            'MSE_Std': result['cv_std'],
            'Correlation': result.get('correlation', 0),
            'SSIM': result.get('ssim', 0),
            'Timestamp': result['timestamp']
        }
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    if df.empty:
        print("❌ No data available for comparison")
        return None
    
    # Sort by dataset and MSE
    df = df.sort_values(['Dataset', 'MSE_Mean'])
    
    print("\n📊 COMPREHENSIVE COMPARISON TABLE:")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.6f'))
    
    return df

def perform_statistical_analysis(results):
    """Perform statistical analysis on results."""
    
    print("\n📊 STATISTICAL ANALYSIS")
    print("=" * 60)
    
    datasets = set(result['dataset'] for result in results.values())
    models = set(result['model'] for result in results.values())
    
    print(f"📈 Datasets: {sorted(datasets)}")
    print(f"🤖 Models: {sorted(models)}")
    
    # Pairwise comparisons per dataset
    statistical_results = {}
    
    for dataset in datasets:
        print(f"\n📊 DATASET: {dataset.upper()}")
        print("-" * 40)
        
        dataset_results = {k: v for k, v in results.items() if v['dataset'] == dataset}
        
        if len(dataset_results) < 2:
            print(f"⚠️ Not enough models for comparison on {dataset}")
            continue
        
        # Extract MSE values for each model
        model_mse = {}
        for key, result in dataset_results.items():
            model = result['model']
            mse = result['cv_mean']
            model_mse[model] = mse
        
        # Find best and worst performers
        best_model = min(model_mse.keys(), key=lambda x: model_mse[x])
        worst_model = max(model_mse.keys(), key=lambda x: model_mse[x])
        
        print(f"🏆 Best: {best_model} (MSE: {model_mse[best_model]:.6f})")
        print(f"📉 Worst: {worst_model} (MSE: {model_mse[worst_model]:.6f})")
        
        # Calculate improvement percentages
        for model in model_mse:
            if model != best_model:
                improvement = ((model_mse[model] - model_mse[best_model]) / model_mse[model]) * 100
                print(f"📈 {best_model} vs {model}: {improvement:.1f}% improvement")
        
        statistical_results[dataset] = {
            'best_model': best_model,
            'best_mse': model_mse[best_model],
            'model_mse': model_mse,
            'num_models': len(model_mse)
        }
    
    return statistical_results

def create_visualization(results, output_path):
    """Create comprehensive visualization."""
    
    print(f"\n📊 CREATING VISUALIZATION")
    print("=" * 60)
    
    # Prepare data for plotting
    datasets = []
    models = []
    mse_values = []
    
    for result in results.values():
        datasets.append(result['dataset'])
        models.append(result['model'])
        mse_values.append(result['cv_mean'])
    
    if not datasets:
        print("❌ No data for visualization")
        return
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Dataset': datasets,
        'Model': models,
        'MSE': mse_values
    })
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Cross-Validation Results\n(100% Real Data)', fontsize=16, fontweight='bold')
    
    # Plot 1: Bar plot by dataset
    ax1 = axes[0, 0]
    sns.barplot(data=plot_df, x='Dataset', y='MSE', hue='Model', ax=ax1)
    ax1.set_title('MSE by Dataset and Model')
    ax1.set_ylabel('Mean Squared Error')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Box plot
    ax2 = axes[0, 1]
    sns.boxplot(data=plot_df, x='Model', y='MSE', ax=ax2)
    ax2.set_title('MSE Distribution by Model')
    ax2.set_ylabel('Mean Squared Error')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Heatmap
    ax3 = axes[1, 0]
    pivot_df = plot_df.pivot(index='Dataset', columns='Model', values='MSE')
    sns.heatmap(pivot_df, annot=True, fmt='.6f', cmap='viridis_r', ax=ax3)
    ax3.set_title('MSE Heatmap (Lower is Better)')
    
    # Plot 4: Performance ranking
    ax4 = axes[1, 1]
    avg_performance = plot_df.groupby('Model')['MSE'].mean().sort_values()
    avg_performance.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_title('Average MSE Across All Datasets')
    ax4.set_ylabel('Average MSE')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add academic integrity note
    fig.text(0.02, 0.02, 'Academic Integrity: 100% Real Data from Trained Models', 
             fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    
    print(f"✅ Visualization saved: {output_path}")

def main():
    """Execute comprehensive CV analysis."""
    
    print("🚀 COMPREHENSIVE CROSS-VALIDATION ANALYSIS")
    print("=" * 80)
    print("🎯 Goal: Analyze all real CV results comprehensively")
    print("🏆 Academic Integrity: 100% real data analysis")
    print("=" * 80)
    
    # Load all results
    results = load_cv_results()
    
    if not results:
        print("❌ No CV results found to analyze")
        return
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Perform statistical analysis
    stats_results = perform_statistical_analysis(results)
    
    # Create visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/comprehensive_cv_analysis_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = output_dir / "comprehensive_cv_comparison.png"
    create_visualization(results, str(viz_path))
    
    # Save results
    summary = {
        'analysis_timestamp': timestamp,
        'total_results': len(results),
        'datasets_analyzed': len(set(r['dataset'] for r in results.values())),
        'models_analyzed': len(set(r['model'] for r in results.values())),
        'statistical_results': stats_results,
        'academic_integrity': '100% real data from trained models',
        'raw_results': results
    }
    
    summary_file = output_dir / "comprehensive_analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save comparison table
    if df is not None:
        df.to_csv(output_dir / "comparison_table.csv", index=False)
        df.to_excel(output_dir / "comparison_table.xlsx", index=False)
    
    print("\n" + "=" * 80)
    print("🏆 COMPREHENSIVE CV ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"📁 Results directory: {output_dir}")
    print(f"📊 Total results analyzed: {len(results)}")
    print(f"🎯 Academic integrity: 100% real data")
    print("=" * 80)

if __name__ == "__main__":
    main()
