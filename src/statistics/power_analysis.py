"""
Statistical Power Analysis for Neural Decoding Research
======================================================

Comprehensive statistical power analysis to justify sample sizes and 
validate the ability to detect meaningful effects.

Author: [Your Name]
Date: 2025-06-20
Version: 1.0
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import norm, t
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings


class PowerAnalysis:
    """
    Statistical power analysis for neural decoding experiments.
    
    This class provides comprehensive power analysis including:
    1. Sample size justification
    2. Minimum detectable effect size calculation
    3. Post-hoc power analysis
    4. Multiple comparisons considerations
    """
    
    def __init__(self):
        self.results = {}
        
    def calculate_power_ttest(self, n: int, effect_size: float, alpha: float = 0.05,
                             alternative: str = 'two-sided') -> float:
        """
        Calculate statistical power for t-test.
        
        Args:
            n: Sample size
            effect_size: Cohen's d effect size
            alpha: Significance level
            alternative: Type of test ('two-sided', 'greater', 'less')
            
        Returns:
            Statistical power (0-1)
        """
        
        if alternative == 'two-sided':
            critical_t = t.ppf(1 - alpha/2, df=n-1)
            # Non-central t-distribution
            ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
            power = 1 - t.cdf(critical_t, df=n-1, loc=ncp) + t.cdf(-critical_t, df=n-1, loc=ncp)
        else:
            critical_t = t.ppf(1 - alpha, df=n-1)
            ncp = effect_size * np.sqrt(n)
            if alternative == 'greater':
                power = 1 - t.cdf(critical_t, df=n-1, loc=ncp)
            else:  # 'less'
                power = t.cdf(-critical_t, df=n-1, loc=ncp)
        
        return power
    
    def minimum_detectable_effect(self, n: int, alpha: float = 0.05, 
                                 power: float = 0.8) -> float:
        """
        Calculate minimum detectable effect size.
        
        Args:
            n: Sample size
            alpha: Significance level
            power: Desired power
            
        Returns:
            Minimum detectable Cohen's d
        """
        
        # For two-sided t-test
        t_alpha = t.ppf(1 - alpha/2, df=n-1)
        t_beta = t.ppf(power, df=n-1)
        
        # Minimum detectable effect size
        min_effect = (t_alpha + t_beta) / np.sqrt(n)
        
        return min_effect
    
    def sample_size_calculation(self, effect_size: float, alpha: float = 0.05,
                               power: float = 0.8) -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected Cohen's d
            alpha: Significance level
            power: Desired power
            
        Returns:
            Required sample size
        """
        
        # For two-sided t-test
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))
    
    def analyze_dataset_power(self, dataset_name: str, n_train: int, n_test: int,
                             champion_score: float, observed_scores: List[float],
                             alpha: float = 0.05) -> Dict[str, Any]:
        """
        Comprehensive power analysis for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            n_train: Training sample size
            n_test: Test sample size  
            champion_score: Champion method score
            observed_scores: List of observed CV scores
            alpha: Significance level
            
        Returns:
            Dictionary with comprehensive power analysis results
        """
        
        print(f"📊 POWER ANALYSIS: {dataset_name.upper()}")
        print("=" * 50)
        
        # Calculate observed effect size
        observed_scores = np.array(observed_scores)
        mean_score = np.mean(observed_scores)
        std_score = np.std(observed_scores, ddof=1)
        
        # Cohen's d calculation
        if std_score > 0:
            cohens_d = (champion_score - mean_score) / std_score
        else:
            cohens_d = 0
            warnings.warn(f"Zero standard deviation for {dataset_name}")
        
        # Sample size for analysis (use CV fold size)
        n_cv = len(observed_scores)
        
        # Post-hoc power analysis
        observed_power = self.calculate_power_ttest(n_cv, abs(cohens_d), alpha)
        
        # Minimum detectable effect with current sample size
        min_detectable = self.minimum_detectable_effect(n_cv, alpha, 0.8)
        
        # Required sample size for different effect sizes
        small_effect_n = self.sample_size_calculation(0.2, alpha, 0.8)
        medium_effect_n = self.sample_size_calculation(0.5, alpha, 0.8)
        large_effect_n = self.sample_size_calculation(0.8, alpha, 0.8)
        
        # Power for different effect sizes with current n
        power_small = self.calculate_power_ttest(n_cv, 0.2, alpha)
        power_medium = self.calculate_power_ttest(n_cv, 0.5, alpha)
        power_large = self.calculate_power_ttest(n_cv, 0.8, alpha)
        
        # Determine adequacy
        adequate_power = observed_power >= 0.8
        adequate_for_medium = power_medium >= 0.8
        
        results = {
            'dataset_name': dataset_name,
            'sample_sizes': {
                'n_train': n_train,
                'n_test': n_test,
                'n_cv_folds': n_cv,
                'total_available': n_train + n_test
            },
            'effect_size_analysis': {
                'observed_cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
                'champion_score': champion_score,
                'observed_mean': mean_score,
                'observed_std': std_score
            },
            'power_analysis': {
                'observed_power': observed_power,
                'power_adequate': adequate_power,
                'min_detectable_effect': min_detectable,
                'power_for_effects': {
                    'small_effect_0.2': power_small,
                    'medium_effect_0.5': power_medium,
                    'large_effect_0.8': power_large
                }
            },
            'sample_size_requirements': {
                'for_small_effect': small_effect_n,
                'for_medium_effect': medium_effect_n,
                'for_large_effect': large_effect_n,
                'current_adequate_for_medium': adequate_for_medium
            },
            'recommendations': self._generate_recommendations(
                n_cv, observed_power, cohens_d, adequate_for_medium
            )
        }
        
        # Print summary
        self._print_power_summary(results)
        
        return results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _generate_recommendations(self, n: int, power: float, effect_size: float,
                                 adequate_for_medium: bool) -> List[str]:
        """Generate recommendations based on power analysis."""
        recommendations = []
        
        if power < 0.8:
            recommendations.append(f"Low statistical power ({power:.3f}). Consider increasing sample size or effect size interpretation.")
        
        if not adequate_for_medium:
            recommendations.append("Insufficient power to detect medium effects. Results should be interpreted cautiously.")
        
        if n < 30:
            recommendations.append("Small sample size may limit generalizability. Consider replication studies.")
        
        if abs(effect_size) < 0.2:
            recommendations.append("Very small effect size. Practical significance should be carefully considered.")
        
        if power >= 0.8 and abs(effect_size) >= 0.5:
            recommendations.append("Adequate power and meaningful effect size. Results are statistically robust.")
        
        return recommendations
    
    def _print_power_summary(self, results: Dict[str, Any]) -> None:
        """Print power analysis summary."""
        
        dataset = results['dataset_name']
        power = results['power_analysis']['observed_power']
        effect_size = results['effect_size_analysis']['observed_cohens_d']
        n_cv = results['sample_sizes']['n_cv_folds']
        
        print(f"📈 Sample Size: {n_cv} CV folds")
        print(f"📏 Effect Size (Cohen's d): {effect_size:.3f} ({results['effect_size_analysis']['effect_size_interpretation']})")
        print(f"⚡ Statistical Power: {power:.3f} ({'✅ Adequate' if power >= 0.8 else '⚠️ Low'})")
        print(f"🎯 Min Detectable Effect: {results['power_analysis']['min_detectable_effect']:.3f}")
        
        print(f"\n🔍 Power for Standard Effect Sizes:")
        power_effects = results['power_analysis']['power_for_effects']
        print(f"   Small (0.2): {power_effects['small_effect_0.2']:.3f}")
        print(f"   Medium (0.5): {power_effects['medium_effect_0.5']:.3f}")
        print(f"   Large (0.8): {power_effects['large_effect_0.8']:.3f}")
        
        print(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"   • {rec}")
    
    def comprehensive_power_analysis(self, datasets_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive power analysis across all datasets.
        
        Args:
            datasets_results: Dictionary with results for each dataset
            
        Returns:
            Comprehensive power analysis report
        """
        
        print("\n🔬 COMPREHENSIVE POWER ANALYSIS")
        print("=" * 60)
        
        all_results = {}
        summary_stats = {
            'adequate_power_count': 0,
            'total_datasets': 0,
            'mean_power': 0,
            'mean_effect_size': 0,
            'sample_size_range': [float('inf'), 0]
        }
        
        for dataset_name, data in datasets_results.items():
            # Extract relevant information
            champion_score = data.get('champion_score', 0)
            cv_scores = data.get('cv_scores', [])
            n_train = data.get('n_train', 0)
            n_test = data.get('n_test', 0)
            
            if len(cv_scores) > 0:
                # Perform power analysis
                results = self.analyze_dataset_power(
                    dataset_name, n_train, n_test, champion_score, cv_scores
                )
                all_results[dataset_name] = results
                
                # Update summary statistics
                power = results['power_analysis']['observed_power']
                effect_size = abs(results['effect_size_analysis']['observed_cohens_d'])
                n_cv = results['sample_sizes']['n_cv_folds']
                
                if power >= 0.8:
                    summary_stats['adequate_power_count'] += 1
                
                summary_stats['total_datasets'] += 1
                summary_stats['mean_power'] += power
                summary_stats['mean_effect_size'] += effect_size
                summary_stats['sample_size_range'][0] = min(summary_stats['sample_size_range'][0], n_cv)
                summary_stats['sample_size_range'][1] = max(summary_stats['sample_size_range'][1], n_cv)
        
        # Calculate means
        if summary_stats['total_datasets'] > 0:
            summary_stats['mean_power'] /= summary_stats['total_datasets']
            summary_stats['mean_effect_size'] /= summary_stats['total_datasets']
        
        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(summary_stats, all_results)
        
        comprehensive_report = {
            'individual_analyses': all_results,
            'summary_statistics': summary_stats,
            'overall_assessment': overall_assessment,
            'methodology_recommendations': self._generate_methodology_recommendations(all_results)
        }
        
        # Print comprehensive summary
        self._print_comprehensive_summary(comprehensive_report)
        
        return comprehensive_report
    
    def _generate_overall_assessment(self, summary_stats: Dict[str, Any], 
                                   all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment of statistical power."""
        
        adequate_ratio = summary_stats['adequate_power_count'] / summary_stats['total_datasets']
        mean_power = summary_stats['mean_power']
        mean_effect = summary_stats['mean_effect_size']
        
        assessment = {
            'power_adequacy': 'adequate' if adequate_ratio >= 0.75 else 'marginal' if adequate_ratio >= 0.5 else 'inadequate',
            'effect_size_magnitude': 'large' if mean_effect >= 0.8 else 'medium' if mean_effect >= 0.5 else 'small',
            'overall_conclusion': '',
            'statistical_rigor': 'high' if adequate_ratio >= 0.75 and mean_power >= 0.8 else 'moderate' if adequate_ratio >= 0.5 else 'low'
        }
        
        # Generate conclusion
        if assessment['power_adequacy'] == 'adequate' and assessment['effect_size_magnitude'] in ['medium', 'large']:
            assessment['overall_conclusion'] = "Study has adequate statistical power to detect meaningful effects."
        elif assessment['power_adequacy'] == 'marginal':
            assessment['overall_conclusion'] = "Study has marginal statistical power. Results should be interpreted cautiously."
        else:
            assessment['overall_conclusion'] = "Study has inadequate statistical power. Consider increasing sample sizes or replication."
        
        return assessment
    
    def _generate_methodology_recommendations(self, all_results: Dict[str, Any]) -> List[str]:
        """Generate methodology recommendations based on power analysis."""
        
        recommendations = []
        
        low_power_datasets = [name for name, results in all_results.items() 
                             if results['power_analysis']['observed_power'] < 0.8]
        
        if low_power_datasets:
            recommendations.append(f"Consider increasing sample sizes for: {', '.join(low_power_datasets)}")
        
        small_effect_datasets = [name for name, results in all_results.items()
                               if abs(results['effect_size_analysis']['observed_cohens_d']) < 0.2]
        
        if small_effect_datasets:
            recommendations.append(f"Small effect sizes observed for: {', '.join(small_effect_datasets)}. Consider practical significance.")
        
        recommendations.append("Report effect sizes alongside p-values for complete statistical reporting.")
        recommendations.append("Consider replication studies to validate findings, especially for datasets with marginal power.")
        
        return recommendations
    
    def _print_comprehensive_summary(self, report: Dict[str, Any]) -> None:
        """Print comprehensive power analysis summary."""
        
        summary = report['summary_statistics']
        assessment = report['overall_assessment']
        
        print(f"\n📊 OVERALL POWER ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"📈 Datasets with Adequate Power: {summary['adequate_power_count']}/{summary['total_datasets']}")
        print(f"⚡ Mean Statistical Power: {summary['mean_power']:.3f}")
        print(f"📏 Mean Effect Size: {summary['mean_effect_size']:.3f}")
        print(f"📊 Sample Size Range: {summary['sample_size_range'][0]}-{summary['sample_size_range'][1]} CV folds")
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   Power Adequacy: {assessment['power_adequacy'].upper()}")
        print(f"   Effect Size Magnitude: {assessment['effect_size_magnitude'].upper()}")
        print(f"   Statistical Rigor: {assessment['statistical_rigor'].upper()}")
        print(f"   Conclusion: {assessment['overall_conclusion']}")
        
        print(f"\n💡 METHODOLOGY RECOMMENDATIONS:")
        for rec in report['methodology_recommendations']:
            print(f"   • {rec}")


if __name__ == "__main__":
    # Example power analysis
    print("🧪 TESTING POWER ANALYSIS")
    print("=" * 40)
    
    analyzer = PowerAnalysis()
    
    # Example dataset results
    example_results = {
        'miyawaki': {
            'champion_score': 0.009845,
            'cv_scores': [0.005500, 0.006000, 0.005200, 0.005800, 0.005300],
            'n_train': 107,
            'n_test': 12
        }
    }
    
    # Run comprehensive analysis
    report = analyzer.comprehensive_power_analysis(example_results)
    
    print(f"\n✅ Power analysis test complete!")
    print(f"📊 Analysis completed for {len(report['individual_analyses'])} datasets")
