"""
Green Neural Decoding Analysis
=============================

Comprehensive analysis of computational efficiency and environmental impact
of CCCV1 vs SOTA methods for sustainable neural decoding research.
"""

import torch
import time
import psutil
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
import sys
sys.path.append('src')
sys.path.append('sota_comparison')
from models.cortexflow_clip_cnn_v1 import create_cccv1_model
from mind_vis_implementation import MindVis
from lightweight_brain_diffuser import LightweightBrainDiffuser

class GreenNeuralDecodingAnalyzer:
    """Analyze computational efficiency and environmental impact"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
    def count_parameters(self, model):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def measure_inference_time(self, model, input_tensor, num_runs=100):
        """Measure inference time with GPU synchronization"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Synchronize GPU
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        return avg_time
    
    def measure_memory_usage(self, model, input_tensor):
        """Measure GPU memory usage"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                _ = model(input_tensor)
            
            memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            return memory_used
        else:
            return 0
    
    def estimate_training_time(self, model, input_tensor, epochs=100):
        """Estimate training time for one epoch"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Dummy target
        if hasattr(model, 'forward'):
            with torch.no_grad():
                dummy_output = model(input_tensor)
                if isinstance(dummy_output, tuple):
                    target_shape = dummy_output[0].shape
                else:
                    target_shape = dummy_output.shape
        
        target = torch.randn(target_shape, device=self.device)
        
        # Measure one training step
        start_time = time.time()
        
        optimizer.zero_grad()
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        step_time = time.time() - start_time
        estimated_epoch_time = step_time * 100  # Assume 100 steps per epoch
        estimated_total_time = estimated_epoch_time * epochs
        
        return step_time, estimated_epoch_time, estimated_total_time
    
    def calculate_carbon_footprint(self, training_time_hours, inference_time_ms, num_inferences=1000):
        """Estimate carbon footprint based on computational requirements"""
        
        # GPU power consumption estimates (Watts)
        gpu_power = {
            'RTX_3060': 170,  # Current GPU
            'RTX_4090': 450,  # High-end GPU
            'A100': 400      # Data center GPU
        }
        
        # Carbon intensity (kg CO2/kWh) - global average
        carbon_intensity = 0.5
        
        # Training carbon footprint
        training_power_kwh = (gpu_power['RTX_3060'] / 1000) * training_time_hours
        training_carbon = training_power_kwh * carbon_intensity
        
        # Inference carbon footprint (per 1000 inferences)
        inference_power_kwh = (gpu_power['RTX_3060'] / 1000) * (inference_time_ms * num_inferences / 3600000)
        inference_carbon = inference_power_kwh * carbon_intensity
        
        return {
            'training_carbon_kg': training_carbon,
            'inference_carbon_kg': inference_carbon,
            'total_carbon_kg': training_carbon + inference_carbon
        }
    
    def analyze_model(self, model_name, model, input_dim):
        """Comprehensive analysis of a single model"""
        print(f"\n🔍 Analyzing {model_name}...")
        
        # Create input tensor
        batch_size = 8
        input_tensor = torch.randn(batch_size, input_dim, device=self.device)
        
        # Parameter count
        total_params, trainable_params = self.count_parameters(model)
        
        # Inference time
        inference_time = self.measure_inference_time(model, input_tensor)
        
        # Memory usage
        memory_usage = self.measure_memory_usage(model, input_tensor)
        
        # Training time estimation
        step_time, epoch_time, total_time = self.estimate_training_time(model, input_tensor)
        
        # Carbon footprint
        training_hours = total_time / 3600
        carbon_footprint = self.calculate_carbon_footprint(training_hours, inference_time * 1000)
        
        # Model size (MB)
        model_size = sum(p.numel() * 4 for p in model.parameters()) / (1024**2)  # 4 bytes per float32
        
        results = {
            'model_name': model_name,
            'parameters': {
                'total': total_params,
                'trainable': trainable_params,
                'size_mb': model_size
            },
            'performance': {
                'inference_time_ms': inference_time * 1000,
                'memory_usage_mb': memory_usage,
                'training_step_time_s': step_time,
                'estimated_epoch_time_min': epoch_time / 60,
                'estimated_total_training_hours': training_hours
            },
            'environmental': carbon_footprint,
            'efficiency_metrics': {
                'params_per_mb': total_params / model_size,
                'inference_fps': 1 / inference_time,
                'carbon_per_param': carbon_footprint['total_carbon_kg'] / total_params * 1e6,  # mg CO2 per param
                'energy_efficiency': 1 / (inference_time * memory_usage)  # Custom efficiency metric
            }
        }
        
        self.results[model_name] = results
        return results
    
    def run_comprehensive_analysis(self):
        """Run analysis on all models"""
        print("🌱 GREEN NEURAL DECODING ANALYSIS")
        print("=" * 60)
        
        # Test different input dimensions
        test_configs = [
            {'name': 'miyawaki', 'input_dim': 967},
            {'name': 'vangerven', 'input_dim': 3092}
        ]
        
        for config in test_configs:
            print(f"\n📊 Dataset: {config['name'].upper()} (input_dim: {config['input_dim']})")
            print("-" * 40)
            
            dataset_results = {}
            
            # CCCV1-Optimized
            try:
                cccv1_model = create_cccv1_model(config['input_dim'], self.device, config['name'], optimized=True)
                cccv1_results = self.analyze_model(f"CCCV1-{config['name']}", cccv1_model, config['input_dim'])
                dataset_results['CCCV1'] = cccv1_results
            except Exception as e:
                print(f"❌ CCCV1 analysis failed: {e}")
            
            # Mind-Vis
            try:
                mind_vis_model = MindVis(config['input_dim'], device=self.device)
                mind_vis_results = self.analyze_model(f"Mind-Vis-{config['name']}", mind_vis_model, config['input_dim'])
                dataset_results['Mind-Vis'] = mind_vis_results
            except Exception as e:
                print(f"❌ Mind-Vis analysis failed: {e}")
            
            # Lightweight Brain-Diffuser
            try:
                brain_diff_model = LightweightBrainDiffuser(config['input_dim'], device=self.device)
                brain_diff_results = self.analyze_model(f"Brain-Diffuser-{config['name']}", brain_diff_model, config['input_dim'])
                dataset_results['Brain-Diffuser'] = brain_diff_results
            except Exception as e:
                print(f"❌ Brain-Diffuser analysis failed: {e}")
            
            self.results[config['name']] = dataset_results
    
    def create_green_comparison_visualization(self):
        """Create comprehensive green computing visualization"""
        
        # Prepare data for visualization
        methods = []
        datasets = []
        carbon_footprints = []
        inference_times = []
        memory_usage = []
        parameters = []
        
        for dataset_name, dataset_results in self.results.items():
            if isinstance(dataset_results, dict) and 'CCCV1' in str(dataset_results):
                for method_name, method_results in dataset_results.items():
                    methods.append(method_name)
                    datasets.append(dataset_name)
                    carbon_footprints.append(method_results['environmental']['total_carbon_kg'])
                    inference_times.append(method_results['performance']['inference_time_ms'])
                    memory_usage.append(method_results['performance']['memory_usage_mb'])
                    parameters.append(method_results['parameters']['total'] / 1e6)  # Millions
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Green Neural Decoding: Computational Efficiency Analysis\n' +
                    'Sustainable AI for Neural Decoding Applications', 
                    fontsize=16, fontweight='bold')
        
        # 1. Carbon Footprint Comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(methods)), carbon_footprints, 
                       color=['green', 'orange', 'red'][:len(methods)], alpha=0.7)
        ax1.set_title('Carbon Footprint (kg CO₂)', fontweight='bold')
        ax1.set_ylabel('Carbon Emissions (kg CO₂)')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add values on bars
        for bar, value in zip(bars1, carbon_footprints):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Inference Time Comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(methods)), inference_times,
                       color=['green', 'orange', 'red'][:len(methods)], alpha=0.7)
        ax2.set_title('Inference Time (ms)', fontweight='bold')
        ax2.set_ylabel('Time (milliseconds)')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        
        # 3. Memory Usage Comparison
        ax3 = axes[0, 2]
        bars3 = ax3.bar(range(len(methods)), memory_usage,
                       color=['green', 'orange', 'red'][:len(methods)], alpha=0.7)
        ax3.set_title('Memory Usage (MB)', fontweight='bold')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=45, ha='right')
        
        # 4. Parameter Count Comparison
        ax4 = axes[1, 0]
        bars4 = ax4.bar(range(len(methods)), parameters,
                       color=['green', 'orange', 'red'][:len(methods)], alpha=0.7)
        ax4.set_title('Model Parameters (Millions)', fontweight='bold')
        ax4.set_ylabel('Parameters (M)')
        ax4.set_xticks(range(len(methods)))
        ax4.set_xticklabels(methods, rotation=45, ha='right')
        
        # 5. Efficiency Scatter Plot
        ax5 = axes[1, 1]
        scatter = ax5.scatter(carbon_footprints, inference_times, 
                            s=[p*20 for p in parameters], 
                            c=['green', 'orange', 'red'][:len(methods)], 
                            alpha=0.7)
        ax5.set_xlabel('Carbon Footprint (kg CO₂)')
        ax5.set_ylabel('Inference Time (ms)')
        ax5.set_title('Efficiency Trade-off\n(Size = Parameters)', fontweight='bold')
        
        # Add method labels
        for i, method in enumerate(methods):
            ax5.annotate(method, (carbon_footprints[i], inference_times[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 6. Green Computing Metrics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate green metrics
        if len(carbon_footprints) > 0:
            best_carbon_idx = np.argmin(carbon_footprints)
            best_speed_idx = np.argmin(inference_times)
            best_memory_idx = np.argmin(memory_usage)
            
            green_text = f"""🌱 GREEN COMPUTING METRICS
            
🏆 Lowest Carbon Footprint:
   {methods[best_carbon_idx]}
   {carbon_footprints[best_carbon_idx]:.3f} kg CO₂
   
⚡ Fastest Inference:
   {methods[best_speed_idx]}
   {inference_times[best_speed_idx]:.2f} ms
   
💾 Lowest Memory Usage:
   {methods[best_memory_idx]}
   {memory_usage[best_memory_idx]:.1f} MB
   
🌍 Environmental Impact:
   • Training: {sum(carbon_footprints):.3f} kg CO₂ total
   • Equivalent to {sum(carbon_footprints)*2.2:.1f} lbs CO₂
   • Tree offset: {sum(carbon_footprints)*40:.1f} trees/year
            """
            
            ax6.text(0.05, 0.95, green_text, transform=ax6.transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        
        # Save visualization
        output_dir = Path("results/green_neural_decoding")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save PNG
        png_file = output_dir / f"green_neural_decoding_analysis_{timestamp}.png"
        plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Green analysis saved: {png_file}")
        
        # Save SVG
        svg_file = output_dir / f"green_neural_decoding_analysis_{timestamp}.svg"
        plt.savefig(svg_file, format='svg', bbox_inches='tight', facecolor='white')
        print(f"🎨 SVG version saved: {svg_file}")
        
        plt.show()
        
        return png_file, svg_file
    
    def save_results(self):
        """Save comprehensive results to JSON"""
        output_dir = Path("results/green_neural_decoding")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"green_analysis_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📊 Results saved: {results_file}")
        return results_file

def main():
    """Run comprehensive green neural decoding analysis"""
    
    print("🌱 STARTING GREEN NEURAL DECODING ANALYSIS")
    print("=" * 60)
    print("🎯 Analyzing computational efficiency and environmental impact")
    print("🌍 Focus: Sustainable AI for Neural Decoding")
    
    # Initialize analyzer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = GreenNeuralDecodingAnalyzer(device)
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()
    
    # Create visualizations
    png_file, svg_file = analyzer.create_green_comparison_visualization()
    
    # Save results
    results_file = analyzer.save_results()
    
    print(f"\n🎉 GREEN NEURAL DECODING ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"📊 Results: {results_file}")
    print(f"🎨 Visualization: {png_file}")
    print(f"🎨 SVG Version: {svg_file}")
    print(f"\n🌱 Ready for Green AI publication!")

if __name__ == "__main__":
    main()
