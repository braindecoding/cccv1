#!/usr/bin/env python3
"""
Analyze CortexFlow Model Files
=============================

Analyze all CortexFlow model files and variations to identify which ones are used and which can be cleaned up.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def analyze_model_files():
    """Analyze all model files in the repository"""
    
    print("🔍 ANALYZING CORTEXFLOW MODEL FILES")
    print("=" * 80)
    
    # Define model directories to check
    model_dirs = [
        Path("models"),
        Path("saved_models"),
        Path("src/models"),
        Path("sota_comparison"),
        Path("publication_packages")
    ]
    
    cortexflow_files = {}
    
    for model_dir in model_dirs:
        if model_dir.exists():
            print(f"\n📁 Checking directory: {model_dir}")
            
            # Find all CortexFlow related files
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    file_name = file_path.name.lower()
                    
                    # Check for CortexFlow related files
                    if any(keyword in file_name for keyword in ['cortex', 'cccv', 'clip_cnn']):
                        category = "unknown"
                        
                        # Categorize files
                        if file_name.endswith('.py'):
                            category = "source_code"
                        elif file_name.endswith('.pth'):
                            category = "pytorch_model"
                        elif file_name.endswith('.pkl'):
                            category = "pickle_model"
                        elif file_name.endswith('.json'):
                            category = "metadata"
                        elif file_name.endswith('.md'):
                            category = "documentation"
                        
                        if category not in cortexflow_files:
                            cortexflow_files[category] = []
                        
                        # Get file size
                        try:
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                        except:
                            size_mb = 0
                        
                        cortexflow_files[category].append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'size_mb': size_mb,
                            'directory': str(file_path.parent)
                        })
    
    return cortexflow_files

def analyze_model_usage():
    """Analyze which models are actually being used"""
    
    print("\n🔍 ANALYZING MODEL USAGE")
    print("=" * 80)
    
    # Check which models are referenced in scripts
    script_dir = Path("scripts")
    usage_analysis = {
        'cortexflow_clip_cnn_v1.py': [],
        'cccv1_models': [],
        'other_cortexflow': []
    }
    
    if script_dir.exists():
        for script_file in script_dir.glob("*.py"):
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for imports and references
                if 'cortexflow_clip_cnn_v1' in content:
                    usage_analysis['cortexflow_clip_cnn_v1.py'].append(str(script_file))
                
                if 'CortexFlowCLIPCNNV1' in content:
                    usage_analysis['cccv1_models'].append(str(script_file))
                
                if any(keyword in content.lower() for keyword in ['cortex', 'cccv']) and script_file.name not in [f.split('/')[-1] for f in usage_analysis['cortexflow_clip_cnn_v1.py']]:
                    usage_analysis['other_cortexflow'].append(str(script_file))
                    
            except Exception as e:
                print(f"⚠️ Error reading {script_file}: {e}")
    
    return usage_analysis

def check_model_checkpoints():
    """Check which model checkpoints exist and their status"""
    
    print("\n🔍 CHECKING MODEL CHECKPOINTS")
    print("=" * 80)
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    checkpoint_status = {}
    
    for dataset in datasets:
        checkpoint_status[dataset] = {}
        
        # Check different checkpoint formats
        checkpoint_patterns = [
            f"models/{dataset}_cv_best.pth",           # Current format
            f"models/{dataset}_cccv1_best.pth",       # Alternative format
            f"saved_models/cortexflow/{dataset}.pth",  # Saved models format
        ]
        
        for pattern in checkpoint_patterns:
            if Path(pattern).exists():
                try:
                    size_mb = Path(pattern).stat().st_size / (1024 * 1024)
                    checkpoint_status[dataset][pattern] = {
                        'exists': True,
                        'size_mb': size_mb,
                        'status': 'active' if 'cv_best' in pattern else 'backup'
                    }
                except:
                    checkpoint_status[dataset][pattern] = {
                        'exists': True,
                        'size_mb': 0,
                        'status': 'error'
                    }
            else:
                checkpoint_status[dataset][pattern] = {
                    'exists': False,
                    'size_mb': 0,
                    'status': 'missing'
                }
    
    return checkpoint_status

def identify_redundant_files(cortexflow_files, usage_analysis, checkpoint_status):
    """Identify files that can be safely removed"""
    
    print("\n🗑️ IDENTIFYING REDUNDANT FILES")
    print("=" * 80)
    
    redundant_files = []
    keep_files = []
    
    # Files to definitely keep
    essential_files = [
        'src/models/cortexflow_clip_cnn_v1.py',  # Main model definition
    ]
    
    # Active checkpoints to keep
    for dataset in checkpoint_status:
        for checkpoint_path, info in checkpoint_status[dataset].items():
            if info['exists'] and info['status'] == 'active':
                essential_files.append(checkpoint_path)
                # Also keep metadata
                metadata_path = checkpoint_path.replace('.pth', '_metadata.json')
                if Path(metadata_path).exists():
                    essential_files.append(metadata_path)
    
    # Analyze each category
    for category, files in cortexflow_files.items():
        for file_info in files:
            file_path = file_info['path']
            
            # Check if file is essential
            is_essential = any(essential in file_path for essential in essential_files)
            
            # Check if file is in publication packages (can be removed)
            is_publication_backup = 'publication_packages' in file_path
            
            # Check if file is duplicate checkpoint
            is_duplicate_checkpoint = (
                file_path.endswith('.pth') and 
                'cccv1_best' in file_path and 
                file_path.replace('cccv1_best', 'cv_best') in [f['path'] for f in files]
            )
            
            if is_essential:
                keep_files.append(file_info)
            elif is_publication_backup or is_duplicate_checkpoint:
                redundant_files.append({
                    **file_info,
                    'reason': 'publication_backup' if is_publication_backup else 'duplicate_checkpoint'
                })
            else:
                # Need manual review
                keep_files.append({
                    **file_info,
                    'review_needed': True
                })
    
    return redundant_files, keep_files

def create_cleanup_recommendations(redundant_files, keep_files):
    """Create cleanup recommendations"""
    
    print("\n📋 CLEANUP RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = {
        'safe_to_remove': [],
        'review_needed': [],
        'keep_essential': [],
        'total_space_saved_mb': 0
    }
    
    # Safe to remove
    for file_info in redundant_files:
        recommendations['safe_to_remove'].append(file_info)
        recommendations['total_space_saved_mb'] += file_info['size_mb']
    
    # Files to keep
    for file_info in keep_files:
        if file_info.get('review_needed'):
            recommendations['review_needed'].append(file_info)
        else:
            recommendations['keep_essential'].append(file_info)
    
    return recommendations

def print_analysis_results(cortexflow_files, usage_analysis, checkpoint_status, recommendations):
    """Print comprehensive analysis results"""
    
    print("\n📊 ANALYSIS RESULTS")
    print("=" * 80)
    
    # File summary
    total_files = sum(len(files) for files in cortexflow_files.values())
    total_size = sum(file_info['size_mb'] for files in cortexflow_files.values() for file_info in files)
    
    print(f"📁 Total CortexFlow files found: {total_files}")
    print(f"💾 Total size: {total_size:.1f} MB")
    
    # By category
    print(f"\n📋 Files by category:")
    for category, files in cortexflow_files.items():
        category_size = sum(f['size_mb'] for f in files)
        print(f"   {category}: {len(files)} files ({category_size:.1f} MB)")
    
    # Usage analysis
    print(f"\n🔍 Usage analysis:")
    print(f"   Scripts using cortexflow_clip_cnn_v1.py: {len(usage_analysis['cortexflow_clip_cnn_v1.py'])}")
    print(f"   Scripts using CortexFlowCLIPCNNV1: {len(usage_analysis['cccv1_models'])}")
    print(f"   Other CortexFlow references: {len(usage_analysis['other_cortexflow'])}")
    
    # Checkpoint status
    print(f"\n💾 Checkpoint status:")
    for dataset, checkpoints in checkpoint_status.items():
        active_checkpoints = sum(1 for info in checkpoints.values() if info['status'] == 'active')
        print(f"   {dataset}: {active_checkpoints} active checkpoints")
    
    # Cleanup recommendations
    print(f"\n🗑️ Cleanup recommendations:")
    print(f"   Safe to remove: {len(recommendations['safe_to_remove'])} files ({recommendations['total_space_saved_mb']:.1f} MB)")
    print(f"   Need review: {len(recommendations['review_needed'])} files")
    print(f"   Keep essential: {len(recommendations['keep_essential'])} files")
    
    # Detailed recommendations
    if recommendations['safe_to_remove']:
        print(f"\n🗑️ SAFE TO REMOVE:")
        for file_info in recommendations['safe_to_remove']:
            print(f"   📄 {file_info['path']} ({file_info['size_mb']:.1f} MB) - {file_info['reason']}")
    
    if recommendations['review_needed']:
        print(f"\n⚠️ NEED MANUAL REVIEW:")
        for file_info in recommendations['review_needed']:
            print(f"   📄 {file_info['path']} ({file_info['size_mb']:.1f} MB)")

def save_analysis_report(cortexflow_files, usage_analysis, checkpoint_status, recommendations):
    """Save detailed analysis report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"results/cortexflow_model_analysis_{timestamp}.json"
    
    report = {
        'analysis_timestamp': timestamp,
        'summary': {
            'total_files': sum(len(files) for files in cortexflow_files.values()),
            'total_size_mb': sum(file_info['size_mb'] for files in cortexflow_files.values() for file_info in files),
            'categories': {cat: len(files) for cat, files in cortexflow_files.items()}
        },
        'files_by_category': cortexflow_files,
        'usage_analysis': usage_analysis,
        'checkpoint_status': checkpoint_status,
        'cleanup_recommendations': recommendations
    }
    
    os.makedirs("results", exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Analysis report saved: {report_file}")
    return report_file

def main():
    """Execute CortexFlow model analysis"""
    
    print("🔍 CORTEXFLOW MODEL ANALYSIS")
    print("=" * 80)
    print("🎯 Goal: Identify all CortexFlow models and recommend cleanup")
    print("=" * 80)
    
    # Perform analysis
    cortexflow_files = analyze_model_files()
    usage_analysis = analyze_model_usage()
    checkpoint_status = check_model_checkpoints()
    redundant_files, keep_files = identify_redundant_files(cortexflow_files, usage_analysis, checkpoint_status)
    recommendations = create_cleanup_recommendations(redundant_files, keep_files)
    
    # Print results
    print_analysis_results(cortexflow_files, usage_analysis, checkpoint_status, recommendations)
    
    # Save report
    report_file = save_analysis_report(cortexflow_files, usage_analysis, checkpoint_status, recommendations)
    
    print("\n" + "=" * 80)
    print("🏆 CORTEXFLOW MODEL ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"📊 Analysis report: {report_file}")
    print(f"💾 Potential space savings: {recommendations['total_space_saved_mb']:.1f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()
