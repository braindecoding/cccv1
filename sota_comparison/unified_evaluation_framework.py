"""
Unified SOTA Evaluation Framework
=================================

Comprehensive comparison framework for CCCV1, Brain-Diffuser, and Mind-Vis.
Academic Integrity: Fair comparison with standardized metrics and protocols.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all SOTA methods
from cccv1_optimized_adapter import CCCV1OptimizedAdapter
from brain_diffuser.src.brain_diffuser import BrainDiffuser
from mind_vis.src.mind_vis_model import MindVis
from data.loader import load_dataset_gpu_optimized


class UnifiedSOTAEvaluator:
    """
    Unified evaluation framework for SOTA comparison
    
    Provides fair and comprehensive comparison between:
    - CCCV1 Optimized
    - Brain-Diffuser  
    - Mind-Vis
    """
    
    def __init__(self, device='cuda', results_dir='comparison_results'):
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Available methods
        self.methods = {
            'CCCV1-Optimized': 'cccv1_optimized',
            'Brain-Diffuser': 'brain_diffuser',
            'Mind-Vis': 'mind_vis'
        }
        
        # Standard datasets
        self.datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
        
        # Evaluation results storage
        self.results = {}
        
        print(f"🎯 Unified SOTA Evaluator initialized")
        print(f"   Device: {device}")
        print(f"   Methods: {list(self.methods.keys())}")
        print(f"   Datasets: {self.datasets}")
        print(f"   Results dir: {results_dir}")
    
    def calculate_comprehensive_metrics(self, y_true: torch.Tensor, 
                                      y_pred: torch.Tensor) -> Dict[str, float]:
        """Calculate standardized evaluation metrics"""
        
        # Ensure same device and shape
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        
        # Flatten for correlation calculation
        y_true_flat = y_true.cpu().numpy().flatten()
        y_pred_flat = y_pred.cpu().numpy().flatten()
        
        # MSE
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        
        # Correlation
        correlation, p_value = pearsonr(y_true_flat, y_pred_flat)
        if np.isnan(correlation):
            correlation = 0.0
            p_value = 1.0
        
        # SSIM (simplified but consistent)
        def calculate_ssim_batch(imgs_true, imgs_pred):
            """Calculate SSIM for batch of images"""
            ssim_scores = []
            
            for i in range(imgs_true.shape[0]):
                img1 = imgs_true[i].cpu().numpy().squeeze()
                img2 = imgs_pred[i].cpu().numpy().squeeze()
                
                # SSIM calculation
                mu1 = np.mean(img1)
                mu2 = np.mean(img2)
                sigma1 = np.var(img1)
                sigma2 = np.var(img2)
                sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
                
                c1 = 0.01 ** 2
                c2 = 0.03 ** 2
                
                ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
                
                ssim_scores.append(ssim)
            
            return ssim_scores
        
        ssim_scores = calculate_ssim_batch(y_true, y_pred)
        ssim_mean = np.mean(ssim_scores)
        ssim_std = np.std(ssim_scores)
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse_tensor = torch.nn.functional.mse_loss(y_pred, y_true)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_tensor)).item()
        
        return {
            'mse': float(mse),
            'correlation': float(correlation),
            'correlation_p_value': float(p_value),
            'ssim_mean': float(ssim_mean),
            'ssim_std': float(ssim_std),
            'ssim_scores': [float(x) for x in ssim_scores],
            'psnr': float(psnr)
        }
    
    def evaluate_cccv1_optimized(self, dataset_name: str, 
                                num_samples: Optional[int] = None) -> Dict:
        """Evaluate CCCV1 Optimized method"""
        
        print(f"📊 Evaluating CCCV1-Optimized on {dataset_name}...")
        
        try:
            # Create adapter
            adapter = CCCV1OptimizedAdapter(dataset_name, self.device)
            
            # Load test data
            _, _, X_test, y_test, _ = load_dataset_gpu_optimized(dataset_name, device=self.device)
            
            if num_samples and num_samples < X_test.shape[0]:
                indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
                X_test = X_test[indices]
                y_test = y_test[indices]
            
            # Generate predictions
            with torch.no_grad():
                predictions, _ = adapter.model(X_test)
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y_test, predictions)
            
            result = {
                'method': 'CCCV1-Optimized',
                'dataset': dataset_name,
                'num_samples': X_test.shape[0],
                'metrics': metrics,
                'model_info': adapter.get_model_info(),
                'status': 'success'
            }
            
            print(f"✅ CCCV1-Optimized: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
            return result
            
        except Exception as e:
            print(f"❌ CCCV1-Optimized failed: {str(e)}")
            return {
                'method': 'CCCV1-Optimized',
                'dataset': dataset_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_brain_diffuser(self, dataset_name: str, 
                               num_samples: Optional[int] = None) -> Dict:
        """Evaluate Brain-Diffuser method"""
        
        print(f"📊 Evaluating Brain-Diffuser on {dataset_name}...")
        
        try:
            # Initialize Brain-Diffuser
            brain_diffuser = BrainDiffuser(device=self.device)
            
            # Setup models (using mock models for now)
            if not brain_diffuser.setup_models():
                raise Exception("Failed to setup Brain-Diffuser models")
            
            # Load trained models (if available)
            if not brain_diffuser.load_trained_models(dataset_name):
                print(f"⚠️  No trained Brain-Diffuser model for {dataset_name}, using mock evaluation")
                # For now, return mock results
                return {
                    'method': 'Brain-Diffuser',
                    'dataset': dataset_name,
                    'status': 'mock',
                    'metrics': {
                        'mse': 0.05,  # Mock values
                        'correlation': 0.75,
                        'ssim_mean': 0.80,
                        'ssim_std': 0.10,
                        'psnr': 15.0
                    },
                    'note': 'Mock evaluation - model not trained'
                }
            
            # Evaluate (if model is trained)
            result = brain_diffuser.evaluate(dataset_name, num_samples or 6)
            result['status'] = 'success'
            
            print(f"✅ Brain-Diffuser: MSE={result['mse']:.6f}, Corr={result['correlation']:.6f}")
            return result
            
        except Exception as e:
            print(f"❌ Brain-Diffuser failed: {str(e)}")
            return {
                'method': 'Brain-Diffuser',
                'dataset': dataset_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def evaluate_mind_vis(self, dataset_name: str, 
                         num_samples: Optional[int] = None) -> Dict:
        """Evaluate Mind-Vis method"""
        
        print(f"📊 Evaluating Mind-Vis on {dataset_name}...")
        
        try:
            # Load test data
            _, _, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device=self.device)
            
            if num_samples and num_samples < X_test.shape[0]:
                indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
                X_test = X_test[indices]
                y_test = y_test[indices]
            
            # Load trained model
            model_path = f"mind_vis/models/{dataset_name}_mind_vis_best.pth"
            
            if not os.path.exists(model_path):
                print(f"⚠️  No trained Mind-Vis model for {dataset_name}, using mock evaluation")
                return {
                    'method': 'Mind-Vis',
                    'dataset': dataset_name,
                    'status': 'mock',
                    'metrics': {
                        'mse': 0.03,  # Mock values
                        'correlation': 0.85,
                        'ssim_mean': 0.85,
                        'ssim_std': 0.08,
                        'psnr': 18.0
                    },
                    'note': 'Mock evaluation - model not trained'
                }
            
            # Create and load model
            model = MindVis(
                input_dim=input_dim,
                device=self.device,
                image_size=y_test.shape[-1],
                channels=y_test.shape[1]
            )
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            
            # Generate predictions
            with torch.no_grad():
                _, _, predictions = model(X_test)
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y_test, predictions)
            
            result = {
                'method': 'Mind-Vis',
                'dataset': dataset_name,
                'num_samples': X_test.shape[0],
                'metrics': metrics,
                'model_info': model.get_model_info(),
                'status': 'success'
            }
            
            print(f"✅ Mind-Vis: MSE={metrics['mse']:.6f}, Corr={metrics['correlation']:.6f}")
            return result
            
        except Exception as e:
            print(f"❌ Mind-Vis failed: {str(e)}")
            return {
                'method': 'Mind-Vis',
                'dataset': dataset_name,
                'status': 'failed',
                'error': str(e)
            }


    def run_comprehensive_comparison(self, datasets: Optional[List[str]] = None,
                                    num_samples: Optional[int] = None) -> Dict:
        """Run comprehensive comparison across all methods and datasets"""

        if datasets is None:
            datasets = self.datasets

        print(f"🎯 COMPREHENSIVE SOTA COMPARISON")
        print(f"Datasets: {datasets}")
        print(f"Methods: {list(self.methods.keys())}")
        print(f"Samples per dataset: {num_samples or 'All'}")
        print("=" * 80)

        comparison_results = {}

        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"DATASET: {dataset.upper()}")
            print(f"{'='*60}")

            dataset_results = {}

            # Evaluate CCCV1-Optimized
            dataset_results['CCCV1-Optimized'] = self.evaluate_cccv1_optimized(dataset, num_samples)

            # Evaluate Brain-Diffuser
            dataset_results['Brain-Diffuser'] = self.evaluate_brain_diffuser(dataset, num_samples)

            # Evaluate Mind-Vis
            dataset_results['Mind-Vis'] = self.evaluate_mind_vis(dataset, num_samples)

            comparison_results[dataset] = dataset_results

        # Store results
        self.results = comparison_results

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"comprehensive_comparison_{timestamp}.json")

        with open(results_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)

        print(f"\n💾 Results saved: {results_path}")

        return comparison_results

    def generate_comparison_summary(self) -> pd.DataFrame:
        """Generate summary table of comparison results"""

        if not self.results:
            print("❌ No results available. Run comparison first.")
            return None

        summary_data = []

        for dataset, methods in self.results.items():
            for method, result in methods.items():
                if result['status'] == 'success':
                    metrics = result['metrics']
                    summary_data.append({
                        'Dataset': dataset,
                        'Method': method,
                        'MSE': metrics['mse'],
                        'Correlation': metrics['correlation'],
                        'SSIM': metrics['ssim_mean'],
                        'PSNR': metrics['psnr'],
                        'Samples': result.get('num_samples', 'N/A'),
                        'Status': 'Success'
                    })
                else:
                    summary_data.append({
                        'Dataset': dataset,
                        'Method': method,
                        'MSE': 'N/A',
                        'Correlation': 'N/A',
                        'SSIM': 'N/A',
                        'PSNR': 'N/A',
                        'Samples': 'N/A',
                        'Status': result['status'].title()
                    })

        df = pd.DataFrame(summary_data)
        return df

    def create_comparison_visualization(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive comparison visualization"""

        if not self.results:
            print("❌ No results available. Run comparison first.")
            return

        # Prepare data for visualization
        df = self.generate_comparison_summary()

        # Filter only successful results
        df_success = df[df['Status'] == 'Success'].copy()

        if df_success.empty:
            print("❌ No successful results to visualize.")
            return

        # Convert metrics to numeric
        for col in ['MSE', 'Correlation', 'SSIM', 'PSNR']:
            df_success[col] = pd.to_numeric(df_success[col], errors='coerce')

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # MSE comparison
        sns.barplot(data=df_success, x='Dataset', y='MSE', hue='Method', ax=axes[0,0])
        axes[0,0].set_title('Mean Squared Error (Lower is Better)')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Correlation comparison
        sns.barplot(data=df_success, x='Dataset', y='Correlation', hue='Method', ax=axes[0,1])
        axes[0,1].set_title('Correlation (Higher is Better)')
        axes[0,1].tick_params(axis='x', rotation=45)

        # SSIM comparison
        sns.barplot(data=df_success, x='Dataset', y='SSIM', hue='Method', ax=axes[1,0])
        axes[1,0].set_title('SSIM (Higher is Better)')
        axes[1,0].tick_params(axis='x', rotation=45)

        # PSNR comparison
        sns.barplot(data=df_success, x='Dataset', y='PSNR', hue='Method', ax=axes[1,1])
        axes[1,1].set_title('PSNR (Higher is Better)')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.suptitle('SOTA Methods Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved: {save_path}")

        plt.show()

    def perform_statistical_analysis(self) -> Dict:
        """Perform statistical significance testing"""

        if not self.results:
            print("❌ No results available. Run comparison first.")
            return {}

        print("📊 Performing statistical analysis...")

        # Collect data for statistical tests
        method_metrics = {}

        for dataset, methods in self.results.items():
            for method, result in methods.items():
                if result['status'] == 'success' and 'ssim_scores' in result['metrics']:
                    if method not in method_metrics:
                        method_metrics[method] = {'ssim_scores': [], 'datasets': []}

                    method_metrics[method]['ssim_scores'].extend(result['metrics']['ssim_scores'])
                    method_metrics[method]['datasets'].extend([dataset] * len(result['metrics']['ssim_scores']))

        # Perform pairwise t-tests
        statistical_results = {}
        methods = list(method_metrics.keys())

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                if len(method_metrics[method1]['ssim_scores']) > 0 and len(method_metrics[method2]['ssim_scores']) > 0:
                    t_stat, p_value = stats.ttest_ind(
                        method_metrics[method1]['ssim_scores'],
                        method_metrics[method2]['ssim_scores']
                    )

                    statistical_results[f"{method1}_vs_{method2}"] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }

        print("✅ Statistical analysis complete")
        return statistical_results


# Export main class
__all__ = ['UnifiedSOTAEvaluator']
