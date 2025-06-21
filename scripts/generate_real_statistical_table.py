#!/usr/bin/env python3
"""
Generate Real Statistical Analysis Table
========================================

Creates real statistical significance testing based on actual CV results.
Academic Integrity: Real statistical analysis from actual data.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

def load_real_cv_results():
    """Load real CV results from the complete analysis."""
    
    results_file = Path("results/real_table2_data_20250620_181958/complete_cv_results.json")
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    
    n1, n2 = len(group1), len(group2)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                         (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return cohens_d

def interpret_effect_size(cohens_d):
    """Interpret Cohen's d effect size."""
    
    abs_d = abs(cohens_d)
    
    if abs_d < 0.2:
        return "Negligible Effect"
    elif abs_d < 0.5:
        return "Small Effect"
    elif abs_d < 0.8:
        return "Medium Effect"
    else:
        return "Large Effect"

def format_p_value(p_value):
    """Format p-value with appropriate significance stars."""
    
    if p_value < 0.001:
        return "< 0.001***"
    elif p_value < 0.01:
        return f"{p_value:.3f}**"
    elif p_value < 0.05:
        return f"{p_value:.3f}*"
    else:
        return f"{p_value:.3f}"

def perform_statistical_comparison(scores1, scores2, model1_name, model2_name, dataset):
    """Perform statistical comparison between two models."""
    
    # Perform paired t-test (since same CV folds)
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Calculate Cohen's d
    cohens_d = calculate_cohens_d(scores1, scores2)
    
    # Interpret effect size
    effect_interpretation = interpret_effect_size(cohens_d)
    
    # Format p-value
    p_formatted = format_p_value(p_value)
    
    result = {
        'comparison': f"{model1_name} vs {model2_name}",
        'dataset': dataset,
        't_statistic': round(t_stat, 2),
        'p_value': p_formatted,
        'p_value_raw': p_value,
        'cohens_d': round(cohens_d, 2),
        'effect_interpretation': effect_interpretation,
        'model1_mean': np.mean(scores1),
        'model1_std': np.std(scores1),
        'model2_mean': np.mean(scores2),
        'model2_std': np.std(scores2),
        'sample_size': len(scores1)
    }
    
    return result

def create_real_statistical_table():
    """Create real statistical analysis table."""
    
    print("📊 CREATING REAL STATISTICAL ANALYSIS TABLE")
    print("=" * 80)
    print("🏆 Academic Integrity: Real statistical analysis from actual CV data")
    
    # Load real CV results
    results = load_real_cv_results()
    
    if not results:
        print("❌ No CV results found")
        return None
    
    # Perform statistical comparisons
    comparisons = [
        ('CortexFlow', 'Mind-Vis'),
        ('CortexFlow', 'Brain-Diffuser'),
        ('Mind-Vis', 'Brain-Diffuser')
    ]
    
    statistical_results = []
    
    for dataset_name, dataset_results in results.items():
        print(f"\n📊 Processing {dataset_name}...")
        
        for model1, model2 in comparisons:
            if model1 in dataset_results and model2 in dataset_results:
                scores1 = dataset_results[model1]['cv_scores']
                scores2 = dataset_results[model2]['cv_scores']
                
                result = perform_statistical_comparison(
                    scores1, scores2, model1, model2, dataset_name
                )
                
                statistical_results.append(result)
                
                print(f"   {model1} vs {model2}: t={result['t_statistic']}, "
                      f"p={result['p_value']}, d={result['cohens_d']}")
    
    return statistical_results

def generate_statistical_table_markdown(statistical_results):
    """Generate statistical table in markdown format."""
    
    table_md = "# Real Table 3: Hasil Pengujian Signifikansi Statistik (100% Real Data)\n\n"
    table_md += "| Perbandingan | Dataset | t-statistic | p-value | Cohen's d | Interpretasi Effect Size |\n"
    table_md += "|--------------|---------|-------------|---------|-----------|-------------------------|\n"
    
    # Group by comparison
    comparisons = {}
    for result in statistical_results:
        comp = result['comparison']
        if comp not in comparisons:
            comparisons[comp] = []
        comparisons[comp].append(result)
    
    for comparison, results in comparisons.items():
        first = True
        for result in results:
            if first:
                table_md += f"| **{result['comparison']}** | {result['dataset'].title()} | "
                first = False
            else:
                table_md += f"| | {result['dataset'].title()} | "
            
            table_md += f"{result['t_statistic']} | {result['p_value']} | "
            table_md += f"{result['cohens_d']} | {result['effect_interpretation']} |\n"
    
    table_md += "\n**Catatan:**\n"
    table_md += "- Data berdasarkan hasil cross-validation real dari model yang dilatih\n"
    table_md += "- Uji t berpasangan (paired t-test) untuk membandingkan performa model\n"
    table_md += "- Cohen's d untuk mengukur effect size: < 0.2 (negligible), 0.2-0.5 (small), 0.5-0.8 (medium), > 0.8 (large)\n"
    table_md += "- Signifikansi: * p < 0.05, ** p < 0.01, *** p < 0.001\n"
    table_md += f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    table_md += "- Academic Integrity: 100% real statistical analysis\n"
    
    return table_md

def save_statistical_results(statistical_results, table_md):
    """Save statistical results."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/real_statistical_table_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    with open(output_dir / "statistical_results.json", 'w') as f:
        json.dump(statistical_results, f, indent=2, default=str)
    
    # Save table markdown
    with open(output_dir / "real_statistical_table.md", 'w') as f:
        f.write(table_md)
    
    print(f"💾 Statistical results saved to: {output_dir}")
    return output_dir

def main():
    """Generate real statistical analysis table."""
    
    print("📊 REAL STATISTICAL ANALYSIS TABLE GENERATION")
    print("=" * 80)
    print("🎯 Goal: Real statistical significance testing")
    print("🏆 Academic Integrity: No fabricated statistical data")
    
    # Create statistical analysis
    statistical_results = create_real_statistical_table()
    
    if statistical_results:
        # Generate table
        table_md = generate_statistical_table_markdown(statistical_results)
        
        # Save results
        output_dir = save_statistical_results(statistical_results, table_md)
        
        # Print table
        print(f"\n{table_md}")
        
        # Summary
        print(f"\n📋 SUMMARY:")
        significant_results = [r for r in statistical_results if r['p_value_raw'] < 0.05]
        large_effects = [r for r in statistical_results if abs(r['cohens_d']) > 0.8]
        
        print(f"   Total comparisons: {len(statistical_results)}")
        print(f"   Significant results: {len(significant_results)}")
        print(f"   Large effect sizes: {len(large_effects)}")
        
        print(f"\n✅ REAL STATISTICAL TABLE GENERATION COMPLETE!")
        print(f"📊 100% real statistical analysis")
        print(f"📁 Results saved to: {output_dir}")
        
    else:
        print(f"❌ No statistical results generated")

if __name__ == "__main__":
    main()
