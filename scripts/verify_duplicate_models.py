#!/usr/bin/env python3
"""
Verify Duplicate Model Files
============================

Check if *_cccv1_best.pth files are duplicates of *_cv_best.pth files.
"""

import os
import hashlib
import json
from pathlib import Path
from datetime import datetime

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"❌ Error calculating hash for {file_path}: {e}")
        return None

def compare_model_files():
    """Compare cv_best and cccv1_best model files"""
    
    print("🔍 VERIFYING DUPLICATE MODEL FILES")
    print("=" * 60)
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    comparison_results = {}
    
    for dataset in datasets:
        print(f"\n📊 Dataset: {dataset}")
        
        cv_best_path = f"models/{dataset}_cv_best.pth"
        cccv1_best_path = f"models/{dataset}_cccv1_best.pth"
        
        dataset_result = {
            'cv_best_exists': os.path.exists(cv_best_path),
            'cccv1_best_exists': os.path.exists(cccv1_best_path),
            'cv_best_size': 0,
            'cccv1_best_size': 0,
            'cv_best_hash': None,
            'cccv1_best_hash': None,
            'are_identical': False,
            'size_difference': 0
        }
        
        # Check file existence and sizes
        if dataset_result['cv_best_exists']:
            dataset_result['cv_best_size'] = os.path.getsize(cv_best_path)
            print(f"   ✅ cv_best.pth: {dataset_result['cv_best_size']:,} bytes")
        else:
            print(f"   ❌ cv_best.pth: Not found")
        
        if dataset_result['cccv1_best_exists']:
            dataset_result['cccv1_best_size'] = os.path.getsize(cccv1_best_path)
            print(f"   ✅ cccv1_best.pth: {dataset_result['cccv1_best_size']:,} bytes")
        else:
            print(f"   ❌ cccv1_best.pth: Not found")
        
        # Calculate size difference
        if dataset_result['cv_best_exists'] and dataset_result['cccv1_best_exists']:
            dataset_result['size_difference'] = abs(dataset_result['cv_best_size'] - dataset_result['cccv1_best_size'])
            print(f"   📏 Size difference: {dataset_result['size_difference']:,} bytes")
            
            # If sizes are identical, calculate hashes
            if dataset_result['size_difference'] == 0:
                print(f"   🔐 Calculating hashes (identical sizes)...")
                
                dataset_result['cv_best_hash'] = calculate_file_hash(cv_best_path)
                dataset_result['cccv1_best_hash'] = calculate_file_hash(cccv1_best_path)
                
                if dataset_result['cv_best_hash'] and dataset_result['cccv1_best_hash']:
                    dataset_result['are_identical'] = (dataset_result['cv_best_hash'] == dataset_result['cccv1_best_hash'])
                    
                    if dataset_result['are_identical']:
                        print(f"   ✅ Files are IDENTICAL (same hash)")
                    else:
                        print(f"   ⚠️ Files are DIFFERENT (different hash)")
                        print(f"      cv_best hash:   {dataset_result['cv_best_hash']}")
                        print(f"      cccv1_best hash: {dataset_result['cccv1_best_hash']}")
            else:
                print(f"   ⚠️ Files have different sizes - not identical")
        
        comparison_results[dataset] = dataset_result
    
    return comparison_results

def check_script_usage():
    """Check which file format is used in scripts"""
    
    print(f"\n🔍 CHECKING SCRIPT USAGE")
    print("=" * 60)
    
    script_dir = Path("scripts")
    usage_stats = {
        'cv_best_references': [],
        'cccv1_best_references': [],
        'total_cv_best': 0,
        'total_cccv1_best': 0
    }
    
    if script_dir.exists():
        for script_file in script_dir.glob("*.py"):
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count references
                cv_best_count = content.count('_cv_best.pth')
                cccv1_best_count = content.count('_cccv1_best.pth')
                
                if cv_best_count > 0:
                    usage_stats['cv_best_references'].append({
                        'file': str(script_file),
                        'count': cv_best_count
                    })
                    usage_stats['total_cv_best'] += cv_best_count
                
                if cccv1_best_count > 0:
                    usage_stats['cccv1_best_references'].append({
                        'file': str(script_file),
                        'count': cccv1_best_count
                    })
                    usage_stats['total_cccv1_best'] += cccv1_best_count
                    
            except Exception as e:
                print(f"⚠️ Error reading {script_file}: {e}")
    
    print(f"📊 Usage Statistics:")
    print(f"   cv_best.pth references: {usage_stats['total_cv_best']} in {len(usage_stats['cv_best_references'])} files")
    print(f"   cccv1_best.pth references: {usage_stats['total_cccv1_best']} in {len(usage_stats['cccv1_best_references'])} files")
    
    if usage_stats['cv_best_references']:
        print(f"\n📝 Files using cv_best.pth:")
        for ref in usage_stats['cv_best_references']:
            print(f"   {ref['file']} ({ref['count']} references)")
    
    if usage_stats['cccv1_best_references']:
        print(f"\n📝 Files using cccv1_best.pth:")
        for ref in usage_stats['cccv1_best_references']:
            print(f"   {ref['file']} ({ref['count']} references)")
    
    return usage_stats

def generate_cleanup_recommendations(comparison_results, usage_stats):
    """Generate cleanup recommendations based on analysis"""
    
    print(f"\n📋 CLEANUP RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = {
        'safe_to_remove': [],
        'space_savings_mb': 0,
        'primary_format': None,
        'action_plan': []
    }
    
    # Determine primary format
    if usage_stats['total_cv_best'] > usage_stats['total_cccv1_best']:
        recommendations['primary_format'] = 'cv_best'
        secondary_format = 'cccv1_best'
    else:
        recommendations['primary_format'] = 'cccv1_best'
        secondary_format = 'cv_best'
    
    print(f"🎯 Primary format: {recommendations['primary_format']}.pth")
    print(f"🎯 Secondary format: {secondary_format}.pth")
    
    # Check for identical files
    identical_count = 0
    total_duplicate_size = 0
    
    for dataset, result in comparison_results.items():
        if result['are_identical']:
            identical_count += 1
            
            # Recommend removing the less-used format
            if recommendations['primary_format'] == 'cv_best':
                remove_file = f"models/{dataset}_cccv1_best.pth"
                size_mb = result['cccv1_best_size'] / (1024 * 1024)
            else:
                remove_file = f"models/{dataset}_cv_best.pth"
                size_mb = result['cv_best_size'] / (1024 * 1024)
            
            recommendations['safe_to_remove'].append({
                'file': remove_file,
                'size_mb': size_mb,
                'reason': 'identical_duplicate'
            })
            
            total_duplicate_size += size_mb
    
    recommendations['space_savings_mb'] = total_duplicate_size
    
    # Generate action plan
    if identical_count > 0:
        recommendations['action_plan'].append(f"✅ Found {identical_count} identical duplicate pairs")
        recommendations['action_plan'].append(f"💾 Can safely remove {total_duplicate_size:.1f} MB of duplicates")
        recommendations['action_plan'].append(f"🗑️ Remove {len(recommendations['safe_to_remove'])} duplicate files")
        
        if recommendations['primary_format'] == 'cv_best':
            recommendations['action_plan'].append("📝 Keep cv_best.pth format (more widely used)")
            recommendations['action_plan'].append("🗑️ Remove cccv1_best.pth duplicates")
        else:
            recommendations['action_plan'].append("📝 Keep cccv1_best.pth format (more widely used)")
            recommendations['action_plan'].append("🗑️ Remove cv_best.pth duplicates")
    else:
        recommendations['action_plan'].append("⚠️ No identical duplicates found")
        recommendations['action_plan'].append("🔍 Manual review needed for different files")
    
    # Print recommendations
    for action in recommendations['action_plan']:
        print(f"   {action}")
    
    if recommendations['safe_to_remove']:
        print(f"\n🗑️ SAFE TO REMOVE:")
        for item in recommendations['safe_to_remove']:
            print(f"   📄 {item['file']} ({item['size_mb']:.1f} MB) - {item['reason']}")
    
    return recommendations

def save_verification_report(comparison_results, usage_stats, recommendations):
    """Save detailed verification report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"results/model_verification_report_{timestamp}.json"
    
    report = {
        'verification_timestamp': timestamp,
        'comparison_results': comparison_results,
        'usage_statistics': usage_stats,
        'cleanup_recommendations': recommendations,
        'summary': {
            'total_datasets': len(comparison_results),
            'identical_pairs': sum(1 for r in comparison_results.values() if r['are_identical']),
            'potential_space_savings_mb': recommendations['space_savings_mb'],
            'primary_format': recommendations['primary_format']
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Verification report saved: {report_file}")
    return report_file

def main():
    """Execute model verification"""
    
    print("🔍 MODEL DUPLICATE VERIFICATION")
    print("=" * 80)
    print("🎯 Goal: Identify duplicate model files for cleanup")
    print("=" * 80)
    
    # Perform verification
    comparison_results = compare_model_files()
    usage_stats = check_script_usage()
    recommendations = generate_cleanup_recommendations(comparison_results, usage_stats)
    
    # Save report
    report_file = save_verification_report(comparison_results, usage_stats, recommendations)
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 MODEL VERIFICATION COMPLETED!")
    print("=" * 80)
    print(f"📊 Verification report: {report_file}")
    print(f"🔍 Identical pairs found: {sum(1 for r in comparison_results.values() if r['are_identical'])}/4")
    print(f"💾 Potential space savings: {recommendations['space_savings_mb']:.1f} MB")
    print(f"📝 Primary format: {recommendations['primary_format']}.pth")
    print("=" * 80)

if __name__ == "__main__":
    main()
