#!/usr/bin/env python3
"""
Model Checkpoint Verification
============================

Verify that all trained models are properly saved and can be loaded without retraining.
"""

import os
import sys
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def verify_cortexflow_checkpoints():
    """Verify CortexFlow model checkpoints."""
    
    print("🔍 VERIFYING CORTEXFLOW CHECKPOINTS")
    print("=" * 60)
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    cortexflow_status = {}
    
    for dataset in datasets:
        print(f"\n📊 Dataset: {dataset}")
        
        # Check for different checkpoint formats
        checkpoint_files = [
            f"models/{dataset}_cv_best.pth",
            f"models/{dataset}_cccv1_best.pth",
            f"saved_models/{dataset}/best_model.pth"
        ]
        
        metadata_files = [
            f"models/{dataset}_cv_best_metadata.json",
            f"results/cv_results/{dataset}_cv_results.json"
        ]
        
        found_checkpoint = None
        found_metadata = None
        
        # Check checkpoints
        for checkpoint_file in checkpoint_files:
            if os.path.exists(checkpoint_file):
                found_checkpoint = checkpoint_file
                print(f"   ✅ Checkpoint: {checkpoint_file}")
                break
        
        # Check metadata
        for metadata_file in metadata_files:
            if os.path.exists(metadata_file):
                found_metadata = metadata_file
                print(f"   ✅ Metadata: {metadata_file}")
                break
        
        # Test loading
        can_load = False
        if found_checkpoint:
            try:
                checkpoint = torch.load(found_checkpoint, map_location='cpu')
                can_load = True
                print(f"   ✅ Loading: Success")
            except Exception as e:
                print(f"   ❌ Loading: Failed - {e}")
        
        cortexflow_status[dataset] = {
            'checkpoint_file': found_checkpoint,
            'metadata_file': found_metadata,
            'can_load': can_load,
            'status': 'ready' if (found_checkpoint and can_load) else 'missing'
        }
        
        if not found_checkpoint:
            print(f"   ❌ No checkpoint found")
    
    return cortexflow_status

def verify_brain_diffuser_checkpoints():
    """Verify Brain-Diffuser model checkpoints."""
    
    print("\n🔍 VERIFYING BRAIN-DIFFUSER CHECKPOINTS")
    print("=" * 60)
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    brain_diffuser_status = {}
    
    for dataset in datasets:
        print(f"\n📊 Dataset: {dataset}")
        
        # Check for different checkpoint formats
        checkpoint_files = [
            f"models/{dataset}_brain_diffuser_simplified.pkl",
            f"models/{dataset}_lightweight_brain_diffuser_best.pth",
            f"models/Lightweight-Brain-Diffuser-{dataset}_cv_best.pth"
        ]
        
        found_checkpoint = None
        
        # Check checkpoints
        for checkpoint_file in checkpoint_files:
            if os.path.exists(checkpoint_file):
                found_checkpoint = checkpoint_file
                print(f"   ✅ Checkpoint: {checkpoint_file}")
                break
        
        # Test loading
        can_load = False
        if found_checkpoint:
            try:
                if found_checkpoint.endswith('.pkl'):
                    with open(found_checkpoint, 'rb') as f:
                        model = pickle.load(f)
                else:
                    model = torch.load(found_checkpoint, map_location='cpu')
                can_load = True
                print(f"   ✅ Loading: Success")
            except Exception as e:
                print(f"   ❌ Loading: Failed - {e}")
        
        brain_diffuser_status[dataset] = {
            'checkpoint_file': found_checkpoint,
            'can_load': can_load,
            'status': 'ready' if (found_checkpoint and can_load) else 'missing'
        }
        
        if not found_checkpoint:
            print(f"   ❌ No checkpoint found")
    
    return brain_diffuser_status

def verify_mind_vis_checkpoints():
    """Verify Mind-Vis model checkpoints."""
    
    print("\n🔍 VERIFYING MIND-VIS CHECKPOINTS")
    print("=" * 60)
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    mind_vis_status = {}
    
    for dataset in datasets:
        print(f"\n📊 Dataset: {dataset}")
        
        # Check for different checkpoint formats
        checkpoint_files = [
            f"sota_comparison/mind_vis/models/{dataset}_mind_vis_best.pth",
            f"models/{dataset}_mind_vis_best.pth",
            f"models/Mind-Vis-{dataset}_cv_best.pth"
        ]
        
        found_checkpoint = None
        
        # Check checkpoints
        for checkpoint_file in checkpoint_files:
            if os.path.exists(checkpoint_file):
                found_checkpoint = checkpoint_file
                print(f"   ✅ Checkpoint: {checkpoint_file}")
                break
        
        # Test loading
        can_load = False
        if found_checkpoint:
            try:
                checkpoint = torch.load(found_checkpoint, map_location='cpu')
                can_load = True
                print(f"   ✅ Loading: Success")
            except Exception as e:
                print(f"   ❌ Loading: Failed - {e}")
        
        mind_vis_status[dataset] = {
            'checkpoint_file': found_checkpoint,
            'can_load': can_load,
            'status': 'ready' if (found_checkpoint and can_load) else 'missing'
        }
        
        if not found_checkpoint:
            print(f"   ❌ No checkpoint found")
    
    return mind_vis_status

def verify_cv_results():
    """Verify CV results are saved."""
    
    print("\n🔍 VERIFYING CV RESULTS")
    print("=" * 60)
    
    cv_results_files = [
        'results/brain_diffuser_cv_results.json',
        'results/mind_vis_cv_results.json'
    ]
    
    cv_status = {}
    
    for results_file in cv_results_files:
        method = results_file.split('/')[-1].replace('_cv_results.json', '').replace('_', '-').title()
        
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                datasets_with_results = len([d for d in data.values() if d.get('status') == 'success'])
                print(f"   ✅ {method}: {datasets_with_results}/4 datasets")
                
                cv_status[method] = {
                    'file': results_file,
                    'datasets_completed': datasets_with_results,
                    'status': 'complete' if datasets_with_results == 4 else 'partial'
                }
                
            except Exception as e:
                print(f"   ❌ {method}: Error loading - {e}")
                cv_status[method] = {'status': 'error', 'error': str(e)}
        else:
            print(f"   ❌ {method}: File not found")
            cv_status[method] = {'status': 'missing'}
    
    return cv_status

def create_checkpoint_summary():
    """Create comprehensive checkpoint summary."""
    
    print("\n📋 CREATING CHECKPOINT SUMMARY")
    print("=" * 60)
    
    # Verify all checkpoints
    cortexflow_status = verify_cortexflow_checkpoints()
    brain_diffuser_status = verify_brain_diffuser_checkpoints()
    mind_vis_status = verify_mind_vis_checkpoints()
    cv_status = verify_cv_results()
    
    # Create summary
    summary = {
        'verification_timestamp': datetime.now().isoformat(),
        'checkpoint_status': {
            'CortexFlow': cortexflow_status,
            'Brain-Diffuser': brain_diffuser_status,
            'Mind-Vis': mind_vis_status
        },
        'cv_results_status': cv_status,
        'overall_readiness': {}
    }
    
    # Calculate overall readiness
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    
    for dataset in datasets:
        cortexflow_ready = cortexflow_status.get(dataset, {}).get('status') == 'ready'
        brain_diffuser_ready = brain_diffuser_status.get(dataset, {}).get('status') == 'ready'
        mind_vis_ready = mind_vis_status.get(dataset, {}).get('status') == 'ready'
        
        summary['overall_readiness'][dataset] = {
            'CortexFlow': cortexflow_ready,
            'Brain-Diffuser': brain_diffuser_ready,
            'Mind-Vis': mind_vis_ready,
            'all_ready': cortexflow_ready and brain_diffuser_ready and mind_vis_ready
        }
    
    # Save summary
    os.makedirs('results/checkpoint_verification', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"results/checkpoint_verification/checkpoint_summary_{timestamp}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Checkpoint summary saved: {summary_file}")
    
    return summary

def print_final_status(summary):
    """Print final checkpoint status."""
    
    print("\n" + "=" * 80)
    print("🏆 CHECKPOINT VERIFICATION SUMMARY")
    print("=" * 80)
    
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    
    # Overall status
    total_ready = 0
    total_possible = len(datasets) * 3  # 3 methods × 4 datasets
    
    for dataset in datasets:
        readiness = summary['overall_readiness'][dataset]
        ready_count = sum(readiness[method] for method in ['CortexFlow', 'Brain-Diffuser', 'Mind-Vis'])
        total_ready += ready_count
        
        status_icon = "✅" if readiness['all_ready'] else "⚠️"
        print(f"{status_icon} {dataset.upper()}: {ready_count}/3 models ready")
        
        for method in ['CortexFlow', 'Brain-Diffuser', 'Mind-Vis']:
            method_icon = "✅" if readiness[method] else "❌"
            print(f"   {method_icon} {method}")
    
    print(f"\n📊 OVERALL STATUS: {total_ready}/{total_possible} models ready ({total_ready/total_possible*100:.1f}%)")
    
    if total_ready == total_possible:
        print("🎉 ALL MODELS READY - NO RETRAINING NEEDED!")
        print("✅ Fair comparison can be run immediately")
    else:
        print("⚠️ Some models missing - may need retraining")
        missing = total_possible - total_ready
        print(f"❌ {missing} models need to be trained")
    
    print("=" * 80)

def main():
    """Execute checkpoint verification."""
    
    print("🔍 MODEL CHECKPOINT VERIFICATION")
    print("=" * 80)
    print("🎯 Goal: Verify all models are saved and loadable")
    print("🏆 Benefit: No need to retrain from scratch")
    print("=" * 80)
    
    # Create comprehensive summary
    summary = create_checkpoint_summary()
    
    # Print final status
    print_final_status(summary)
    
    return summary

if __name__ == "__main__":
    main()
