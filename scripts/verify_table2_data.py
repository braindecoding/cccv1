#!/usr/bin/env python3
"""
Verify Table 2 Data
===================

Compare Table 2 data with actual CV results from trained models.
Academic Integrity: Verify authenticity of published data.
"""

import json
from pathlib import Path
import pandas as pd

def load_real_cv_data():
    """Load real CV data from metadata files."""
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    models = {
        'CortexFlow': 'cv_best_metadata.json',
        'Mind-Vis': 'Mind-Vis-{}_cv_best_metadata.json', 
        'Brain-Diffuser': 'Lightweight-Brain-Diffuser-{}_cv_best_metadata.json'
    }
    
    real_data = {}
    
    for dataset in datasets:
        real_data[dataset] = {}
        
        # CortexFlow (CCCV1)
        cccv1_file = Path(f"models/{dataset}_cv_best_metadata.json")
        if cccv1_file.exists():
            with open(cccv1_file, 'r') as f:
                metadata = json.load(f)
                real_data[dataset]['CortexFlow'] = {
                    'best_score': metadata.get('best_score', 'N/A'),
                    'cv_mean': metadata.get('cv_mean', 'N/A'),
                    'cv_std': metadata.get('cv_std', 'N/A')
                }
        
        # Mind-Vis
        mindvis_file = Path(f"models/Mind-Vis-{dataset}_cv_best_metadata.json")
        if mindvis_file.exists():
            with open(mindvis_file, 'r') as f:
                metadata = json.load(f)
                real_data[dataset]['Mind-Vis'] = {
                    'best_score': metadata.get('best_score', 'N/A'),
                    'cv_mean': metadata.get('cv_mean', 'N/A'),
                    'cv_std': metadata.get('cv_std', 'N/A')
                }
        
        # Brain-Diffuser
        braindiff_file = Path(f"models/Lightweight-Brain-Diffuser-{dataset}_cv_best_metadata.json")
        if braindiff_file.exists():
            with open(braindiff_file, 'r') as f:
                metadata = json.load(f)
                # Brain-Diffuser metadata might have different structure
                if 'cv_mean' in metadata:
                    real_data[dataset]['Brain-Diffuser'] = {
                        'best_score': metadata.get('best_score', 'N/A'),
                        'cv_mean': metadata.get('cv_mean', 'N/A'),
                        'cv_std': metadata.get('cv_std', 'N/A')
                    }
                else:
                    # Use best_score as approximation
                    real_data[dataset]['Brain-Diffuser'] = {
                        'best_score': metadata.get('best_score', 'N/A'),
                        'cv_mean': metadata.get('best_score', 'N/A'),
                        'cv_std': 'N/A'
                    }
    
    return real_data

def table2_claimed_data():
    """Data claimed in Table 2."""
    
    return {
        'miyawaki': {
            'CortexFlow': {'cv_mean': 0.0037, 'cv_std': 0.0031},
            'Mind-Vis': {'cv_mean': 0.0306, 'cv_std': 0.0098},
            'Brain-Diffuser': {'cv_mean': 0.0645, 'cv_std': 0.0133}
        },
        'vangerven': {
            'CortexFlow': {'cv_mean': 0.0245, 'cv_std': 0.0035},
            'Mind-Vis': {'cv_mean': 0.0290, 'cv_std': 0.0018},
            'Brain-Diffuser': {'cv_mean': 0.0547, 'cv_std': 0.0044}
        },
        'crell': {
            'CortexFlow': {'cv_mean': 0.0324, 'cv_std': 0.0010},
            'Mind-Vis': {'cv_mean': 0.0330, 'cv_std': 0.0012},
            'Brain-Diffuser': {'cv_mean': 0.0421, 'cv_std': 0.0016}
        },
        'mindbigdata': {
            'CortexFlow': {'cv_mean': 0.0565, 'cv_std': 0.0013},
            'Mind-Vis': {'cv_mean': 0.0574, 'cv_std': 0.0012},
            'Brain-Diffuser': {'cv_mean': 0.0577, 'cv_std': 0.0011}
        }
    }

def compare_data(real_data, claimed_data):
    """Compare real vs claimed data."""
    
    print("🔍 COMPARING TABLE 2 DATA WITH REAL CV RESULTS")
    print("=" * 80)
    
    comparison_results = []
    
    for dataset in ['miyawaki', 'vangerven', 'crell', 'mindbigdata']:
        print(f"\n📊 DATASET: {dataset.upper()}")
        print("-" * 40)
        
        for model in ['CortexFlow', 'Mind-Vis', 'Brain-Diffuser']:
            real = real_data.get(dataset, {}).get(model, {})
            claimed = claimed_data.get(dataset, {}).get(model, {})
            
            if real and claimed:
                real_mean = real.get('cv_mean', 'N/A')
                real_std = real.get('cv_std', 'N/A')
                claimed_mean = claimed.get('cv_mean', 'N/A')
                claimed_std = claimed.get('cv_std', 'N/A')
                
                # Check if data matches
                if isinstance(real_mean, (int, float)) and isinstance(claimed_mean, (int, float)):
                    mean_diff = abs(real_mean - claimed_mean)
                    mean_match = mean_diff < 0.001  # Allow small tolerance
                    
                    if isinstance(real_std, (int, float)) and isinstance(claimed_std, (int, float)):
                        std_diff = abs(real_std - claimed_std)
                        std_match = std_diff < 0.001
                    else:
                        std_match = False
                        std_diff = 'N/A'
                    
                    status = "✅ MATCH" if (mean_match and std_match) else "❌ MISMATCH"
                    
                    print(f"   {model}:")
                    print(f"      Real:    {real_mean:.6f} ± {real_std}")
                    print(f"      Claimed: {claimed_mean:.4f} ± {claimed_std:.4f}")
                    print(f"      Status:  {status}")
                    
                    comparison_results.append({
                        'dataset': dataset,
                        'model': model,
                        'real_mean': real_mean,
                        'real_std': real_std,
                        'claimed_mean': claimed_mean,
                        'claimed_std': claimed_std,
                        'mean_diff': mean_diff,
                        'std_diff': std_diff,
                        'match': mean_match and std_match
                    })
                else:
                    print(f"   {model}: ❌ INCOMPLETE DATA")
                    print(f"      Real:    {real_mean} ± {real_std}")
                    print(f"      Claimed: {claimed_mean} ± {claimed_std}")
            else:
                print(f"   {model}: ❌ NO DATA AVAILABLE")
    
    return comparison_results

def generate_verification_report(comparison_results):
    """Generate verification report."""
    
    print(f"\n📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    total_comparisons = len(comparison_results)
    matches = sum(1 for r in comparison_results if r['match'])
    mismatches = total_comparisons - matches
    
    print(f"Total comparisons: {total_comparisons}")
    print(f"Exact matches: {matches}")
    print(f"Mismatches: {mismatches}")
    print(f"Match rate: {matches/total_comparisons*100:.1f}%")
    
    if mismatches > 0:
        print(f"\n❌ MISMATCHED DATA:")
        for result in comparison_results:
            if not result['match']:
                print(f"   {result['dataset']}-{result['model']}: "
                      f"Real={result['real_mean']:.6f}±{result['real_std']}, "
                      f"Claimed={result['claimed_mean']:.4f}±{result['claimed_std']:.4f}")
    
    # Overall assessment
    if matches == total_comparisons:
        print(f"\n✅ VERDICT: TABLE 2 DATA IS AUTHENTIC")
        print("All claimed values match real CV results.")
    elif matches > total_comparisons * 0.8:
        print(f"\n⚠️ VERDICT: TABLE 2 DATA IS MOSTLY AUTHENTIC")
        print("Most values match, but some discrepancies exist.")
    else:
        print(f"\n❌ VERDICT: TABLE 2 DATA IS QUESTIONABLE")
        print("Significant discrepancies with real CV results.")
    
    return matches, total_comparisons

def main():
    """Verify Table 2 data authenticity."""
    
    print("🔍 TABLE 2 DATA VERIFICATION")
    print("=" * 80)
    print("🎯 Comparing claimed CV results with actual trained model data")
    print("🏆 Academic Integrity Check")
    
    # Load real data
    print("\n📊 Loading real CV data from trained models...")
    real_data = load_real_cv_data()
    
    # Load claimed data
    print("📋 Loading claimed data from Table 2...")
    claimed_data = table2_claimed_data()
    
    # Compare data
    comparison_results = compare_data(real_data, claimed_data)
    
    # Generate report
    matches, total = generate_verification_report(comparison_results)
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    if matches == total:
        print("✅ Table 2 contains REAL DATA from actual CV results")
    elif matches > 0:
        print("⚠️ Table 2 contains MIXED DATA (some real, some not)")
    else:
        print("❌ Table 2 contains FABRICATED DATA")
    
    print(f"\n📁 Verification complete!")

if __name__ == "__main__":
    main()
