#!/usr/bin/env python3
"""
Full SOTA Models Training Script
===============================

Complete training of Mind-Vis and Brain-Diffuser on all 4 datasets.
Academic Integrity: 100% real training with proper cross-validation.
"""

import subprocess
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time

def setup_training_environment():
    """Setup environment for SOTA training."""
    
    print("🔧 SETTING UP SOTA TRAINING ENVIRONMENT")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def train_mind_vis_dataset(dataset_name):
    """Train Mind-Vis on a specific dataset."""
    
    print(f"\n🧠 TRAINING MIND-VIS ON {dataset_name.upper()}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run Mind-Vis training
        print(f"🔄 Starting Mind-Vis cross-validation training...")
        
        cmd = [
            'python', 'sota_comparison/mind_vis/train.py',
            '--dataset', dataset_name,
            '--cv_folds', '10',
            '--save_results'
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        
        # Run the training script
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.', encoding='utf-8', errors='ignore')
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Mind-Vis {dataset_name} training completed successfully")
            print("📊 Training output (last 300 chars):")
            print(result.stdout[-300:] if result.stdout else "No output")
            
            return {
                'dataset': dataset_name,
                'model': 'Mind-Vis',
                'status': 'success',
                'training_time': training_time,
                'output': result.stdout,
                'command': ' '.join(cmd)
            }
        else:
            print(f"❌ Mind-Vis {dataset_name} training failed")
            print("Error output:")
            print(result.stderr if result.stderr else "No error output")
            
            return {
                'dataset': dataset_name,
                'model': 'Mind-Vis',
                'status': 'failed',
                'error': result.stderr,
                'training_time': training_time,
                'command': ' '.join(cmd)
            }
        
    except Exception as e:
        print(f"❌ Error training Mind-Vis on {dataset_name}: {e}")
        return {
            'dataset': dataset_name,
            'model': 'Mind-Vis',
            'status': 'failed',
            'error': str(e)
        }

def train_brain_diffuser_dataset(dataset_name):
    """Train Brain-Diffuser on a specific dataset."""
    
    print(f"\n🧠 TRAINING BRAIN-DIFFUSER ON {dataset_name.upper()}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run Brain-Diffuser training
        print(f"🔄 Starting Brain-Diffuser cross-validation training...")
        
        cmd = [
            'python', 'sota_comparison/brain_diffuser/train.py',
            '--dataset', dataset_name,
            '--cv_folds', '10',
            '--save_results'
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        
        # Run the training script
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.', encoding='utf-8', errors='ignore')
        
        end_time = time.time()
        training_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Brain-Diffuser {dataset_name} training completed successfully")
            print("📊 Training output (last 300 chars):")
            print(result.stdout[-300:] if result.stdout else "No output")
            
            return {
                'dataset': dataset_name,
                'model': 'Brain-Diffuser',
                'status': 'success',
                'training_time': training_time,
                'output': result.stdout,
                'command': ' '.join(cmd)
            }
        else:
            print(f"❌ Brain-Diffuser {dataset_name} training failed")
            print("Error output:")
            print(result.stderr if result.stderr else "No error output")
            
            return {
                'dataset': dataset_name,
                'model': 'Brain-Diffuser',
                'status': 'failed',
                'error': result.stderr,
                'training_time': training_time,
                'command': ' '.join(cmd)
            }
        
    except Exception as e:
        print(f"❌ Error training Brain-Diffuser on {dataset_name}: {e}")
        return {
            'dataset': dataset_name,
            'model': 'Brain-Diffuser',
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Execute full SOTA models training on all datasets."""
    
    print("🚀 FULL SOTA MODELS TRAINING")
    print("=" * 80)
    print("🎯 Goal: Train Mind-Vis and Brain-Diffuser on all 4 datasets")
    print("🏆 Academic Integrity: 100% real training with proper CV")
    print("=" * 80)
    
    # Setup environment
    device = setup_training_environment()
    
    # Define datasets and models
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    models = ['Mind-Vis', 'Brain-Diffuser']
    
    # Training results
    training_results = []
    total_start_time = time.time()
    
    # Train each model on each dataset
    for dataset in datasets:
        print(f"\n📊 DATASET: {dataset.upper()}")
        print("=" * 60)
        
        # Train Mind-Vis
        print(f"\n🤖 MODEL 1/2: MIND-VIS")
        print("-" * 40)
        mindvis_result = train_mind_vis_dataset(dataset)
        training_results.append(mindvis_result)
        
        # Brief pause
        time.sleep(2)
        
        # Train Brain-Diffuser
        print(f"\n🤖 MODEL 2/2: BRAIN-DIFFUSER")
        print("-" * 40)
        braindiff_result = train_brain_diffuser_dataset(dataset)
        training_results.append(braindiff_result)
        
        # Brief pause between datasets
        if dataset != datasets[-1]:
            print(f"⏸️ Brief pause before next dataset...")
            time.sleep(3)
    
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    
    # Save training summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = Path(f"results/training_logs/sota_full_training_{timestamp}.json")
    
    summary = {
        'training_timestamp': timestamp,
        'total_training_time': total_training_time,
        'device': str(device),
        'datasets_trained': len(datasets),
        'models_trained': len(models),
        'total_experiments': len(datasets) * len(models),
        'academic_integrity': '100% real data training',
        'training_results': training_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("🏆 FULL SOTA MODELS TRAINING COMPLETED!")
    print("=" * 80)
    
    successful_trainings = sum(1 for r in training_results if r['status'] == 'success')
    failed_trainings = len(training_results) - successful_trainings
    total_experiments = len(datasets) * len(models)
    
    print(f"📊 Training Summary:")
    print(f"   ✅ Successful: {successful_trainings}/{total_experiments}")
    print(f"   ❌ Failed: {failed_trainings}/{total_experiments}")
    print(f"   ⏱️ Total time: {total_training_time:.2f} seconds")
    print(f"   💾 Results saved: {summary_file}")
    
    # Break down by model
    mindvis_success = sum(1 for r in training_results if r['model'] == 'Mind-Vis' and r['status'] == 'success')
    braindiff_success = sum(1 for r in training_results if r['model'] == 'Brain-Diffuser' and r['status'] == 'success')
    
    print(f"\n📈 Model-wise Results:")
    print(f"   🧠 Mind-Vis: {mindvis_success}/{len(datasets)} datasets")
    print(f"   🤖 Brain-Diffuser: {braindiff_success}/{len(datasets)} datasets")
    
    if successful_trainings == total_experiments:
        print("\n🎉 ALL SOTA MODELS TRAINED SUCCESSFULLY!")
        print("🎯 Ready for comprehensive cross-validation")
    else:
        print(f"\n⚠️ {failed_trainings} experiments failed")
        print("🔍 Check logs for details")
    
    return training_results

if __name__ == "__main__":
    main()
