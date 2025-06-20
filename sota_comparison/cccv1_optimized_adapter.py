"""
CCCV1 Optimized Adapter for SOTA Comparison
==========================================

This script adapts the existing optimized CCCV1 models for comparison
with Brain-Diffuser and Mind-Vis, maintaining all optimizations.
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
import argparse

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized
from data.loader import load_dataset_gpu_optimized


class CCCV1OptimizedAdapter:
    """Adapter for existing optimized CCCV1 models"""
    
    def __init__(self, dataset_name, device='cuda'):
        self.dataset_name = dataset_name
        self.device = device
        self.model = None
        self.metadata = None
        
        # Load existing trained model
        self.load_trained_model()
    
    def load_trained_model(self):
        """Load existing trained CCCV1 model"""
        
        # Try CV model first (best for comparison)
        cv_model_path = f"models/{self.dataset_name}_cv_best.pth"
        cv_metadata_path = f"models/{self.dataset_name}_cv_best_metadata.json"
        
        if os.path.exists(cv_model_path) and os.path.exists(cv_metadata_path):
            # Load metadata
            with open(cv_metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Create model with same configuration
            input_dim = self.metadata['input_dim']
            self.model = CortexFlowCLIPCNNV1Optimized(
                input_dim=input_dim,
                device=self.device,
                dataset_name=self.dataset_name
            )
            
            # Load trained weights
            self.model.load_state_dict(torch.load(cv_model_path, map_location=self.device))
            self.model.eval()
            
            print(f"✅ Loaded CV model: {cv_model_path}")
            print(f"   Best fold: {self.metadata['best_fold']}")
            print(f"   Best score: {self.metadata['best_score']:.6f}")
            
        else:
            # Fallback to regular CCCV1 model
            cccv1_model_path = f"models/{self.dataset_name}_cccv1_best.pth"
            
            if os.path.exists(cccv1_model_path):
                # Load dataset to get input_dim
                X_train, _, _, _, input_dim = load_dataset_gpu_optimized(self.dataset_name, device=self.device)
                # input_dim already obtained from load_dataset_gpu_optimized
                
                self.model = CortexFlowCLIPCNNV1Optimized(
                    input_dim=input_dim,
                    device=self.device,
                    dataset_name=self.dataset_name
                )
                
                self.model.load_state_dict(torch.load(cccv1_model_path, map_location=self.device))
                self.model.eval()
                
                print(f"✅ Loaded CCCV1 model: {cccv1_model_path}")
                
                # Create metadata
                self.metadata = {
                    'dataset_name': self.dataset_name,
                    'input_dim': input_dim,
                    'model_architecture': 'CortexFlowCLIPCNNV1Optimized',
                    'source': 'cccv1_best'
                }
            else:
                raise FileNotFoundError(f"No trained model found for {self.dataset_name}")
    
    def evaluate(self, num_samples=6):
        """Evaluate the optimized model"""
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Load test data
        _, _, X_test, y_test, _ = load_dataset_gpu_optimized(self.dataset_name, device=self.device)
        
        print(f"📊 Evaluating {self.dataset_name} with optimized CCCV1...")
        print(f"   Test samples: {X_test.shape[0]}")
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            predictions, embeddings = self.model(X_test)
        
        # Calculate metrics
        mse = torch.nn.functional.mse_loss(predictions, y_test).item()
        
        # Correlation
        pred_flat = predictions.cpu().numpy().flatten()
        true_flat = y_test.cpu().numpy().flatten()
        correlation = np.corrcoef(pred_flat, true_flat)[0, 1]
        
        metrics = {
            'mse': mse,
            'correlation': correlation,
            'num_samples': X_test.shape[0]
        }
        
        print(f"✅ Results:")
        print(f"   MSE: {mse:.6f}")
        print(f"   Correlation: {correlation:.6f}")
        
        # Generate sample reconstructions
        if num_samples > 0:
            indices = np.random.choice(X_test.shape[0], min(num_samples, X_test.shape[0]), replace=False)
            sample_x = X_test[indices]
            sample_y = y_test[indices]
            
            with torch.no_grad():
                sample_pred, _ = self.model(sample_x)
            
            # Save samples for comparison
            samples = {
                'original': sample_y.cpu().numpy(),
                'reconstructed': sample_pred.cpu().numpy(),
                'indices': indices.tolist()
            }
            
            return metrics, samples
        
        return metrics, None
    
    def get_model_info(self):
        """Get model information for comparison"""
        
        if self.metadata is None:
            return {}
        
        info = {
            'method_name': 'CCCV1-Optimized',
            'dataset': self.dataset_name,
            'architecture': self.metadata.get('model_architecture', 'CortexFlowCLIPCNNV1Optimized'),
            'input_dim': self.metadata.get('input_dim'),
            'optimizations': {
                'dataset_specific': True,
                'clip_guidance': True,
                'progressive_dropout': True,
                'semantic_enhancement': True
            }
        }
        
        if 'config' in self.metadata:
            info['config'] = self.metadata['config']
        
        return info


def run_cccv1_comparison(datasets=['miyawaki', 'vangerven', 'crell', 'mindbigdata'], 
                        device='cuda', num_samples=6):
    """Run CCCV1 optimized evaluation for comparison"""
    
    print("🎯 CCCV1 OPTIMIZED EVALUATION FOR SOTA COMPARISON")
    print("=" * 60)
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*50}")
        
        try:
            # Create adapter
            adapter = CCCV1OptimizedAdapter(dataset, device)
            
            # Evaluate
            metrics, samples = adapter.evaluate(num_samples)
            
            # Get model info
            model_info = adapter.get_model_info()
            
            # Store results
            results[dataset] = {
                'metrics': metrics,
                'model_info': model_info,
                'samples': samples,
                'status': 'success'
            }
            
            print(f"✅ {dataset}: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
            
        except Exception as e:
            print(f"❌ {dataset}: {str(e)}")
            results[dataset] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"sota_comparison/comparison_results/cccv1_optimized_results_{timestamp}.json"
    
    os.makedirs("sota_comparison/comparison_results", exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    for dataset, result in results.items():
        if result['status'] == 'success' and result['samples'] is not None:
            result['samples']['original'] = result['samples']['original'].tolist()
            result['samples']['reconstructed'] = result['samples']['reconstructed'].tolist()
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved: {results_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("CCCV1 OPTIMIZED SUMMARY")
    print(f"{'='*60}")
    
    for dataset, result in results.items():
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"{dataset:12}: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
        else:
            print(f"{dataset:12}: FAILED")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CCCV1 Optimized Evaluation for SOTA Comparison')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['miyawaki', 'vangerven', 'crell', 'mindbigdata', 'all'],
                        help='Dataset to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of sample reconstructions')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'all':
        datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    else:
        datasets = [args.dataset]
    
    run_cccv1_comparison(datasets, device, args.samples)


if __name__ == "__main__":
    main()
