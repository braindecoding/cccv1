#!/usr/bin/env python3
"""
Visualize with Correct CortexFlow Model
======================================

Use the actual CortexFlowCLIPCNNV1Optimized model that was used in fair comparison.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import load_dataset_gpu_optimized
from models.cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized

class CorrectCortexFlowVisualizer:
    """Visualizer using the correct CortexFlow model"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
        
    def load_correct_cortexflow(self, dataset_name):
        """Load CortexFlow using the exact same method as fair comparison"""
        
        print(f"🔧 Loading CORRECT CortexFlow for {dataset_name}")
        
        try:
            # Load dataset to get input dimension
            _, _, _, _, input_dim = load_dataset_gpu_optimized(dataset_name, device=self.device)
            
            # Create model using the EXACT same method as fair comparison
            model = CortexFlowCLIPCNNV1Optimized(input_dim=input_dim, device=self.device, dataset_name=dataset_name)
            
            print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Load checkpoint
            model_path = f"models/{dataset_name}_cv_best.pth"
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.eval()
                
                print(f"✅ CortexFlow loaded correctly for {dataset_name}")
                return model
            else:
                print(f"❌ Checkpoint not found: {model_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading CortexFlow: {e}")
            return None
    
    def test_cortexflow_accuracy(self, dataset_name):
        """Test if CortexFlow gives results matching fair comparison"""
        
        print(f"\n🧪 TESTING CORTEXFLOW ACCURACY FOR {dataset_name.upper()}")
        print("-" * 60)
        
        # Load model
        model = self.load_correct_cortexflow(dataset_name)
        if model is None:
            return None
        
        # Load test data
        _, _, X_test, y_test, _ = load_dataset_gpu_optimized(dataset_name, device=self.device)
        
        # Test on small sample
        X_sample = X_test[:10]
        y_sample = y_test[:10]
        
        # Get prediction
        with torch.no_grad():
            if hasattr(model, 'forward') and len(model.forward.__code__.co_varnames) > 1:
                # Model returns tuple (visual_output, embedding)
                pred, _ = model(X_sample)
            else:
                pred = model(X_sample)
        
        # Calculate MSE
        mse = torch.mean((y_sample - pred) ** 2).item()
        
        # Compare with fair comparison results
        fair_comparison_mse = {
            'miyawaki': 0.005500,
            'vangerven': 0.044505,
            'crell': 0.032525,
            'mindbigdata': 0.057019
        }
        
        expected_mse = fair_comparison_mse.get(dataset_name, 0.0)
        
        print(f"📊 Test Results:")
        print(f"   Sample MSE: {mse:.6f}")
        print(f"   Fair comparison MSE: {expected_mse:.6f}")
        print(f"   Difference: {abs(mse - expected_mse):.6f}")
        
        # Check if results are close (within reasonable range for different samples)
        if abs(mse - expected_mse) < expected_mse * 2:  # Within 2x range
            print(f"✅ Results are consistent with fair comparison!")
            return model, mse
        else:
            print(f"⚠️ Results differ significantly from fair comparison")
            return model, mse
    
    def generate_correct_reconstructions(self, dataset_name, num_samples=6):
        """Generate reconstructions using correct CortexFlow"""
        
        print(f"\n🎨 GENERATING CORRECT RECONSTRUCTIONS FOR {dataset_name.upper()}")
        print("-" * 60)
        
        # Load dataset
        _, _, X_test, y_test, _ = load_dataset_gpu_optimized(dataset_name, device=self.device)
        
        # Select samples
        if num_samples > X_test.shape[0]:
            num_samples = X_test.shape[0]
        
        indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
        X_samples = X_test[indices]
        y_samples = y_test[indices]
        
        reconstructions = {
            'original': y_samples.cpu().numpy(),
            'fmri_input': X_samples.cpu().numpy()
        }
        
        # CortexFlow reconstruction (CORRECT)
        model = self.load_correct_cortexflow(dataset_name)
        if model is not None:
            try:
                with torch.no_grad():
                    # Handle different return types
                    output = model(X_samples)
                    if isinstance(output, tuple):
                        cortexflow_pred, _ = output  # (visual_output, embedding)
                    else:
                        cortexflow_pred = output
                
                reconstructions['CortexFlow'] = cortexflow_pred.cpu().numpy()
                print(f"✅ CortexFlow reconstruction: {cortexflow_pred.shape}")
                
                # Calculate MSE for verification
                mse = torch.mean((y_samples - cortexflow_pred) ** 2).item()
                print(f"📊 CortexFlow MSE: {mse:.6f}")
                
            except Exception as e:
                print(f"❌ CortexFlow reconstruction failed: {e}")
        
        return reconstructions
    
    def calculate_correct_metrics(self, dataset_name):
        """Calculate metrics using correct CortexFlow"""
        
        print(f"\n📊 CALCULATING CORRECT METRICS FOR {dataset_name.upper()}")
        print("-" * 60)
        
        # Load dataset
        _, _, X_test, y_test, _ = load_dataset_gpu_optimized(dataset_name, device=self.device)
        
        # Use more samples for better metrics
        num_samples = min(20, X_test.shape[0])
        indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
        X_samples = X_test[indices]
        y_samples = y_test[indices]
        
        original = y_samples.cpu().numpy()
        
        # Load correct CortexFlow
        model = self.load_correct_cortexflow(dataset_name)
        if model is None:
            return None
        
        # Get predictions
        with torch.no_grad():
            output = model(X_samples)
            if isinstance(output, tuple):
                pred_tensor, _ = output
            else:
                pred_tensor = output
        
        pred = pred_tensor.cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((original - pred) ** 2)
        correlation = np.corrcoef(original.flatten(), pred.flatten())[0, 1]
        
        # SSIM (simplified)
        def ssim_simple(img1, img2):
            mu1, mu2 = np.mean(img1), np.mean(img2)
            sigma1, sigma2 = np.std(img1), np.std(img2)
            sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
            
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
            return ssim
        
        ssim_scores = []
        for i in range(original.shape[0]):
            ssim_score = ssim_simple(original[i, 0], pred[i, 0])
            ssim_scores.append(ssim_score)
        
        ssim_mean = np.mean(ssim_scores)
        
        metrics = {
            'mse': float(mse),
            'correlation': float(correlation),
            'ssim': float(ssim_mean),
            'num_samples': int(num_samples),
            'status': 'success_correct'
        }
        
        print(f"✅ CORRECT CortexFlow metrics:")
        print(f"   MSE: {mse:.6f}")
        print(f"   Correlation: {correlation:.6f}")
        print(f"   SSIM: {ssim_mean:.6f}")
        
        # Compare with fair comparison
        fair_mse = {
            'miyawaki': 0.005500,
            'vangerven': 0.044505,
            'crell': 0.032525,
            'mindbigdata': 0.057019
        }.get(dataset_name, 0.0)
        
        print(f"   Fair comparison MSE: {fair_mse:.6f}")
        print(f"   Difference: {abs(mse - fair_mse):.6f}")
        
        return metrics

def main():
    """Execute correct CortexFlow testing"""
    
    print("🔧 CORRECT CORTEXFLOW VISUALIZATION TEST")
    print("=" * 80)
    print("🎯 Goal: Use the EXACT same CortexFlow model as fair comparison")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualizer = CorrectCortexFlowVisualizer(device=device)
    
    all_metrics = {}
    
    for dataset in visualizer.datasets:
        print(f"\n{'='*60}")
        print(f"TESTING DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        try:
            # Test accuracy first
            model, test_mse = visualizer.test_cortexflow_accuracy(dataset)
            
            if model is not None:
                # Calculate comprehensive metrics
                metrics = visualizer.calculate_correct_metrics(dataset)
                if metrics:
                    all_metrics[dataset] = metrics
            
        except Exception as e:
            print(f"❌ Error testing {dataset}: {e}")
    
    # Save results
    if all_metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/correct_cortexflow_metrics_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\n💾 Correct metrics saved: {output_file}")
        
        # Print summary
        print(f"\n📊 CORRECT CORTEXFLOW METRICS SUMMARY:")
        print("=" * 60)
        
        for dataset, metrics in all_metrics.items():
            print(f"\n🧠 {dataset.upper()}:")
            print(f"   ✅ MSE: {metrics['mse']:.6f}")
            print(f"   ✅ Correlation: {metrics['correlation']:.6f}")
            print(f"   ✅ SSIM: {metrics['ssim']:.6f}")
    
    print("\n" + "=" * 80)
    print("🏆 CORRECT CORTEXFLOW TESTING COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
