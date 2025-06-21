#!/usr/bin/env python3
"""
Verify Computational Efficiency Data
====================================

Verify Table 4 data by measuring actual computational characteristics.
Academic Integrity: Real measurement vs claimed data.
"""

import torch
import time
import psutil
import json
from pathlib import Path
import numpy as np
import sys
from datetime import datetime

# Add paths for imports
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / 'src'))
sys.path.append(str(parent_dir / 'sota_comparison'))

# Import models
try:
    from src.models.cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized
    from sota_comparison.mind_vis.src.mind_vis_manual import MindVisModel
    from sota_comparison.brain_diffuser.src.lightweight_brain_diffuser import LightweightBrainDiffuser
except ImportError as e:
    print(f"❌ Import error: {e}")

def count_parameters(model):
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())

def measure_memory_usage(model, input_tensor, device='cuda'):
    """Measure GPU memory usage during inference."""
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warm up
        with torch.no_grad():
            _ = model(input_tensor)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory
        with torch.no_grad():
            _ = model(input_tensor)
        
        memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        return memory_used
    else:
        # For CPU, measure process memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**3)
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        memory_after = process.memory_info().rss / (1024**3)
        return memory_after - memory_before

def measure_inference_time(model, input_tensor, num_runs=100):
    """Measure inference time with multiple runs."""
    
    model.eval()
    times = []
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Measure inference time
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_tensor)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return np.mean(times), np.std(times)

def load_training_time_from_metadata(model_type, dataset='miyawaki'):
    """Load training time from metadata if available."""
    
    if model_type == 'CortexFlow':
        metadata_file = Path(f"models/{dataset}_cv_best_metadata.json")
    elif model_type == 'Mind-Vis':
        metadata_file = Path(f"models/Mind-Vis-{dataset}_cv_best_metadata.json")
    elif model_type == 'Brain-Diffuser':
        metadata_file = Path(f"models/Lightweight-Brain-Diffuser-{dataset}_cv_best_metadata.json")
    else:
        return None
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Look for training time information
            training_time = metadata.get('training_time', None)
            if training_time:
                return training_time
            
            # Estimate from timestamp if available
            if 'timestamp' in metadata:
                # This is just a placeholder - real training time would need to be logged
                return "Not measured during training"
            
        except Exception as e:
            print(f"❌ Error reading metadata: {e}")
    
    return None

def analyze_model_efficiency(model_type, dataset='miyawaki', device='cuda'):
    """Analyze computational efficiency of a model."""
    
    print(f"\n🔍 ANALYZING {model_type.upper()} EFFICIENCY")
    print("-" * 50)
    
    # Load model
    if model_type == 'CortexFlow':
        metadata_file = Path(f"models/{dataset}_cv_best_metadata.json")
        model_file = Path(f"models/{dataset}_cv_best.pth")
    elif model_type == 'Mind-Vis':
        metadata_file = Path(f"models/Mind-Vis-{dataset}_cv_best_metadata.json")
        model_file = Path(f"models/Mind-Vis-{dataset}_cv_best.pth")
    elif model_type == 'Brain-Diffuser':
        metadata_file = Path(f"models/Lightweight-Brain-Diffuser-{dataset}_cv_best_metadata.json")
        model_file = Path(f"models/Lightweight-Brain-Diffuser-{dataset}_cv_best.pth")
    
    if not metadata_file.exists() or not model_file.exists():
        print(f"❌ Model files not found for {model_type}")
        return None
    
    try:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        input_dim = metadata.get('input_dim', 1000)
        
        # Create model
        if model_type == 'CortexFlow':
            model = CortexFlowCLIPCNNV1Optimized(input_dim, device, dataset).to(device)
        elif model_type == 'Mind-Vis':
            model = MindVisModel(input_dim, device, image_size=28).to(device)
        elif model_type == 'Brain-Diffuser':
            model = LightweightBrainDiffuser(input_dim=input_dim, device=device, image_size=28).to(device)
        
        # Load state dict
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Create test input
        test_input = torch.randn(1, input_dim).to(device)
        
        # Measure characteristics
        print("📊 Measuring computational characteristics...")
        
        # 1. Parameter count
        param_count = count_parameters(model)
        param_count_m = param_count / 1e6
        
        # 2. Memory usage
        memory_usage = measure_memory_usage(model, test_input, device)
        
        # 3. Inference time
        inf_mean, inf_std = measure_inference_time(model, test_input)
        
        # 4. Training time (from metadata or estimate)
        training_time = load_training_time_from_metadata(model_type, dataset)
        
        results = {
            'model_type': model_type,
            'parameter_count': param_count,
            'parameter_count_m': param_count_m,
            'memory_usage_gb': memory_usage,
            'inference_time_ms_mean': inf_mean,
            'inference_time_ms_std': inf_std,
            'training_time': training_time,
            'dataset': dataset,
            'device': device
        }
        
        print(f"✅ Results:")
        print(f"   Parameters: {param_count_m:.1f}M ({param_count:,})")
        print(f"   Memory Usage: {memory_usage:.3f} GB")
        print(f"   Inference Time: {inf_mean:.2f} ± {inf_std:.2f} ms")
        print(f"   Training Time: {training_time}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing {model_type}: {e}")
        return None

def compare_with_claimed_data():
    """Compare measured data with claimed Table 4 data."""
    
    claimed_data = {
        'CortexFlow': {
            'parameter_count_m': 156,
            'memory_usage_gb': 0.37,
            'inference_time_ms_mean': 1.22,
            'inference_time_ms_std': 0.07,
            'training_time_hours': 78.6
        },
        'Mind-Vis': {
            'parameter_count_m': 318,
            'memory_usage_gb': 1.23,
            'inference_time_ms_mean': 4.39,
            'inference_time_ms_std': 0.18,
            'training_time_hours': 384.3
        },
        'Brain-Diffuser': {
            'parameter_count_m': 158,
            'memory_usage_gb': 0.62,
            'inference_time_ms_mean': 2.17,
            'inference_time_ms_std': 0.01,
            'training_time_hours': 238.3
        }
    }
    
    return claimed_data

def main():
    """Verify computational efficiency data."""
    
    print("🔍 COMPUTATIONAL EFFICIENCY DATA VERIFICATION")
    print("=" * 80)
    print("🎯 Comparing claimed Table 4 data with actual measurements")
    print("🏆 Academic Integrity Check")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎯 Using device: {device}")
    
    models = ['CortexFlow', 'Mind-Vis', 'Brain-Diffuser']
    measured_results = {}
    
    # Measure actual data
    for model_type in models:
        result = analyze_model_efficiency(model_type, 'miyawaki', device)
        if result:
            measured_results[model_type] = result
    
    # Load claimed data
    claimed_data = compare_with_claimed_data()
    
    # Compare results
    print(f"\n📊 COMPARISON: MEASURED vs CLAIMED")
    print("=" * 80)
    
    comparison_results = []
    
    for model_type in models:
        if model_type in measured_results:
            measured = measured_results[model_type]
            claimed = claimed_data[model_type]
            
            print(f"\n🔍 {model_type.upper()}:")
            
            # Parameter count
            measured_params = measured['parameter_count_m']
            claimed_params = claimed['parameter_count_m']
            param_match = abs(measured_params - claimed_params) < 10  # Allow 10M tolerance
            
            print(f"   Parameters: Measured={measured_params:.1f}M, Claimed={claimed_params}M, Match={param_match}")
            
            # Memory usage
            measured_memory = measured['memory_usage_gb']
            claimed_memory = claimed['memory_usage_gb']
            memory_match = abs(measured_memory - claimed_memory) < 0.5  # Allow 0.5GB tolerance
            
            print(f"   Memory: Measured={measured_memory:.3f}GB, Claimed={claimed_memory}GB, Match={memory_match}")
            
            # Inference time
            measured_inf = measured['inference_time_ms_mean']
            claimed_inf = claimed['inference_time_ms_mean']
            inf_match = abs(measured_inf - claimed_inf) < 2.0  # Allow 2ms tolerance
            
            print(f"   Inference: Measured={measured_inf:.2f}ms, Claimed={claimed_inf}ms, Match={inf_match}")
            
            # Training time (harder to verify)
            print(f"   Training Time: Measured={measured['training_time']}, Claimed={claimed['training_time_hours']}h")
            
            comparison_results.append({
                'model': model_type,
                'param_match': param_match,
                'memory_match': memory_match,
                'inference_match': inf_match,
                'measured': measured,
                'claimed': claimed
            })
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    total_comparisons = len(comparison_results) * 3  # 3 metrics per model
    matches = sum(r['param_match'] + r['memory_match'] + r['inference_match'] for r in comparison_results)
    
    print(f"   Total metric comparisons: {total_comparisons}")
    print(f"   Matches: {matches}")
    print(f"   Match rate: {matches/total_comparisons*100:.1f}%")
    
    if matches == total_comparisons:
        print("✅ VERDICT: TABLE 4 DATA IS AUTHENTIC")
    elif matches > total_comparisons * 0.7:
        print("⚠️ VERDICT: TABLE 4 DATA IS MOSTLY AUTHENTIC")
    else:
        print("❌ VERDICT: TABLE 4 DATA IS QUESTIONABLE")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/efficiency_verification_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "efficiency_verification.json", 'w') as f:
        json.dump({
            'measured_results': measured_results,
            'claimed_data': claimed_data,
            'comparison_results': comparison_results,
            'match_rate': matches/total_comparisons,
            'timestamp': timestamp
        }, f, indent=2, default=str)
    
    print(f"💾 Verification results saved to: {output_dir}")

if __name__ == "__main__":
    main()
