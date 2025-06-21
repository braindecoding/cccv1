#!/usr/bin/env python3
"""
Create Final SOTA Visualization
===============================

Create comprehensive SOTA comparison visualization with correct CortexFlow results.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import load_dataset_gpu_optimized
from models.cortexflow_clip_cnn_v1 import CortexFlowCLIPCNNV1Optimized

class FinalSOTAVisualizer:
    """Create final SOTA comparison visualization"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
        
    def load_all_models(self, dataset_name):
        """Load all three models for comparison"""
        
        models = {}
        
        # 1. Load CortexFlow (correct version)
        try:
            _, _, _, _, input_dim = load_dataset_gpu_optimized(dataset_name, device=self.device)
            cortexflow_model = CortexFlowCLIPCNNV1Optimized(input_dim=input_dim, device=self.device, dataset_name=dataset_name)
            
            model_path = f"models/{dataset_name}_cv_best.pth"
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                cortexflow_model.load_state_dict(state_dict)
                cortexflow_model.eval()
                models['CortexFlow'] = cortexflow_model
                print(f"✅ CortexFlow loaded for {dataset_name}")
        except Exception as e:
            print(f"❌ CortexFlow loading failed: {e}")
        
        # 2. Load Brain-Diffuser
        try:
            bd_path = f"models/{dataset_name}_brain_diffuser_simplified.pkl"
            if os.path.exists(bd_path):
                with open(bd_path, 'rb') as f:
                    bd_model = pickle.load(f)
                models['Brain-Diffuser'] = bd_model
                print(f"✅ Brain-Diffuser loaded for {dataset_name}")
        except Exception as e:
            print(f"❌ Brain-Diffuser loading failed: {e}")
        
        # 3. Load Mind-Vis
        try:
            mv_path = f"sota_comparison/mind_vis/models/{dataset_name}_mind_vis_best.pth"
            if os.path.exists(mv_path):
                from scripts.train_mind_vis_fixed import SimplifiedMindVis
                _, _, _, _, input_dim = load_dataset_gpu_optimized(dataset_name, device=self.device)
                output_dim = 28 * 28
                
                mv_model = SimplifiedMindVis(input_dim, output_dim, self.device)
                mv_model.load_state_dict(torch.load(mv_path, map_location=self.device))
                mv_model.eval()
                models['Mind-Vis'] = mv_model
                print(f"✅ Mind-Vis loaded for {dataset_name}")
        except Exception as e:
            print(f"❌ Mind-Vis loading failed: {e}")
        
        return models
    
    def generate_reconstructions(self, dataset_name, models, num_samples=6):
        """Generate reconstructions from all models"""
        
        print(f"\n🎨 GENERATING RECONSTRUCTIONS FOR {dataset_name.upper()}")
        print("-" * 60)
        
        # Load dataset
        _, _, X_test, y_test, _ = load_dataset_gpu_optimized(dataset_name, device=self.device)
        
        # Select samples
        if num_samples > X_test.shape[0]:
            num_samples = X_test.shape[0]
        
        # Use fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
        X_samples = X_test[indices]
        y_samples = y_test[indices]
        
        reconstructions = {
            'original': y_samples.cpu().numpy(),
            'fmri_input': X_samples.cpu().numpy()
        }
        
        # CortexFlow reconstruction
        if 'CortexFlow' in models:
            try:
                with torch.no_grad():
                    output = models['CortexFlow'](X_samples)
                    if isinstance(output, tuple):
                        cortexflow_pred, _ = output
                    else:
                        cortexflow_pred = output
                
                reconstructions['CortexFlow'] = cortexflow_pred.cpu().numpy()
                mse = torch.mean((y_samples - cortexflow_pred) ** 2).item()
                print(f"✅ CortexFlow: MSE {mse:.6f}")
            except Exception as e:
                print(f"❌ CortexFlow reconstruction failed: {e}")
        
        # Brain-Diffuser reconstruction
        if 'Brain-Diffuser' in models:
            try:
                X_np = X_samples.cpu().numpy()
                bd_pred_flat = models['Brain-Diffuser'].predict(X_np)
                bd_pred = bd_pred_flat.reshape(num_samples, 1, 28, 28)
                reconstructions['Brain-Diffuser'] = bd_pred
                
                mse = np.mean((y_samples.cpu().numpy() - bd_pred) ** 2)
                print(f"✅ Brain-Diffuser: MSE {mse:.6f}")
            except Exception as e:
                print(f"❌ Brain-Diffuser reconstruction failed: {e}")
        
        # Mind-Vis reconstruction
        if 'Mind-Vis' in models:
            try:
                with torch.no_grad():
                    mv_pred_flat = models['Mind-Vis'](X_samples)
                    mv_pred = mv_pred_flat.view(num_samples, 1, 28, 28)
                
                reconstructions['Mind-Vis'] = mv_pred.cpu().numpy()
                mse = torch.mean((y_samples - mv_pred) ** 2).item()
                print(f"✅ Mind-Vis: MSE {mse:.6f}")
            except Exception as e:
                print(f"❌ Mind-Vis reconstruction failed: {e}")
        
        return reconstructions
    
    def create_sota_comparison_visualization(self, all_reconstructions, save_dir):
        """Create comprehensive SOTA comparison visualization"""
        
        print(f"\n📊 CREATING SOTA COMPARISON VISUALIZATION")
        print("-" * 60)
        
        # Create figure with subplots for each dataset
        fig, axes = plt.subplots(len(self.datasets), 4, figsize=(16, 4 * len(self.datasets)))
        
        methods = ['CortexFlow', 'Brain-Diffuser', 'Mind-Vis']
        
        for i, dataset in enumerate(self.datasets):
            if dataset in all_reconstructions:
                reconstructions = all_reconstructions[dataset]
                
                # Show one sample from each dataset
                sample_idx = 0
                
                # Original
                original_img = reconstructions['original'][sample_idx, 0]
                axes[i, 0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
                axes[i, 0].set_title('Original' if i == 0 else '', fontsize=12, fontweight='bold')
                axes[i, 0].axis('off')
                
                # Methods
                for j, method in enumerate(methods):
                    if method in reconstructions:
                        recon_img = reconstructions[method][sample_idx, 0]
                        axes[i, j + 1].imshow(recon_img, cmap='gray', vmin=0, vmax=1)
                        axes[i, j + 1].set_title(method if i == 0 else '', fontsize=12, fontweight='bold')
                        axes[i, j + 1].axis('off')
                        
                        # Calculate MSE
                        mse = np.mean((original_img - recon_img) ** 2)
                        axes[i, j + 1].text(0.02, 0.98, f'MSE: {mse:.4f}', 
                                           transform=axes[i, j + 1].transAxes,
                                           verticalalignment='top', fontsize=8,
                                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Dataset label
                axes[i, 0].text(-0.15, 0.5, dataset.upper(), 
                               transform=axes[i, 0].transAxes,
                               rotation=90, verticalalignment='center', 
                               fontsize=12, fontweight='bold')
        
        plt.suptitle('SOTA Comparison: Reconstruction Results Across All Datasets', 
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save visualization
        save_path = save_dir / "sota_reconstruction_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        save_path_svg = save_dir / "sota_reconstruction_comparison.svg"
        plt.savefig(save_path_svg, bbox_inches='tight')
        
        plt.close()
        
        print(f"💾 SOTA comparison saved: {save_path}")
        print(f"💾 SOTA comparison saved: {save_path_svg}")
        
        return save_path
    
    def create_performance_summary_table(self, save_dir):
        """Create performance summary table"""
        
        print(f"\n📊 CREATING PERFORMANCE SUMMARY TABLE")
        print("-" * 60)
        
        # Load correct CortexFlow metrics
        cortexflow_metrics_file = "results/correct_cortexflow_metrics_20250621_082448.json"
        with open(cortexflow_metrics_file, 'r') as f:
            cortexflow_metrics = json.load(f)
        
        # Load other metrics (from previous analysis)
        brain_diffuser_metrics = {
            'miyawaki': {'mse': 0.000002, 'correlation': 0.999996, 'ssim': 0.999994},
            'vangerven': {'mse': 0.000000, 'correlation': 0.999999, 'ssim': 0.999999},
            'crell': {'mse': 0.028426, 'correlation': 0.603957, 'ssim': 0.536842},
            'mindbigdata': {'mse': 0.044114, 'correlation': 0.662465, 'ssim': 0.580379}
        }
        
        mind_vis_metrics = {
            'miyawaki': {'mse': 0.004152, 'correlation': 0.989936, 'ssim': 0.985541},
            'vangerven': {'mse': 0.038078, 'correlation': 0.778187, 'ssim': 0.723215},
            'crell': {'mse': 0.032111, 'correlation': 0.533125, 'ssim': 0.463699},
            'mindbigdata': {'mse': 0.054151, 'correlation': 0.556072, 'ssim': 0.467810}
        }
        
        # Create performance table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table
        table_data = []
        headers = ['Dataset', 'Metric', 'CortexFlow', 'Brain-Diffuser', 'Mind-Vis', 'Winner']
        
        for dataset in self.datasets:
            # MSE row
            cf_mse = cortexflow_metrics[dataset]['mse']
            bd_mse = brain_diffuser_metrics[dataset]['mse']
            mv_mse = mind_vis_metrics[dataset]['mse']
            
            mse_winner = 'Brain-Diffuser' if bd_mse <= min(cf_mse, mv_mse) else ('CortexFlow' if cf_mse <= mv_mse else 'Mind-Vis')
            
            table_data.append([
                dataset.upper(), 'MSE',
                f'{cf_mse:.6f}', f'{bd_mse:.6f}', f'{mv_mse:.6f}',
                f'🏆 {mse_winner}'
            ])
            
            # Correlation row
            cf_corr = cortexflow_metrics[dataset]['correlation']
            bd_corr = brain_diffuser_metrics[dataset]['correlation']
            mv_corr = mind_vis_metrics[dataset]['correlation']
            
            corr_winner = 'Brain-Diffuser' if bd_corr >= max(cf_corr, mv_corr) else ('CortexFlow' if cf_corr >= mv_corr else 'Mind-Vis')
            
            table_data.append([
                '', 'Correlation',
                f'{cf_corr:.6f}', f'{bd_corr:.6f}', f'{mv_corr:.6f}',
                f'🏆 {corr_winner}'
            ])
            
            # SSIM row
            cf_ssim = cortexflow_metrics[dataset]['ssim']
            bd_ssim = brain_diffuser_metrics[dataset]['ssim']
            mv_ssim = mind_vis_metrics[dataset]['ssim']
            
            ssim_winner = 'Brain-Diffuser' if bd_ssim >= max(cf_ssim, mv_ssim) else ('CortexFlow' if cf_ssim >= mv_ssim else 'Mind-Vis')
            
            table_data.append([
                '', 'SSIM',
                f'{cf_ssim:.6f}', f'{bd_ssim:.6f}', f'{mv_ssim:.6f}',
                f'🏆 {ssim_winner}'
            ])
            
            # Add separator
            if dataset != self.datasets[-1]:
                table_data.append(['', '', '', '', '', ''])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.12, 0.15, 0.15, 0.15, 0.18])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('SOTA Comparison: Performance Summary Table', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save table
        table_path = save_dir / "sota_performance_table.png"
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        
        table_path_svg = save_dir / "sota_performance_table.svg"
        plt.savefig(table_path_svg, bbox_inches='tight')
        
        plt.close()
        
        print(f"💾 Performance table saved: {table_path}")
        print(f"💾 Performance table saved: {table_path_svg}")
        
        return table_path

def main():
    """Execute final SOTA visualization"""
    
    print("🎨 FINAL SOTA COMPARISON VISUALIZATION")
    print("=" * 80)
    print("🎯 Goal: Create comprehensive SOTA comparison with correct results")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualizer = FinalSOTAVisualizer(device=device)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"results/final_sota_visualization_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reconstructions for all datasets
    all_reconstructions = {}
    
    for dataset in visualizer.datasets:
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        try:
            models = visualizer.load_all_models(dataset)
            if models:
                reconstructions = visualizer.generate_reconstructions(dataset, models, num_samples=6)
                all_reconstructions[dataset] = reconstructions
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
    
    # Create visualizations
    if all_reconstructions:
        # Create SOTA comparison
        visualizer.create_sota_comparison_visualization(all_reconstructions, save_dir)
        
        # Create performance table
        visualizer.create_performance_summary_table(save_dir)
    
    print("\n" + "=" * 80)
    print("🏆 FINAL SOTA VISUALIZATION COMPLETED!")
    print("=" * 80)
    print(f"📁 Results directory: {save_dir}")
    print(f"🎨 SOTA comparison: sota_reconstruction_comparison.png/.svg")
    print(f"📊 Performance table: sota_performance_table.png/.svg")
    print("=" * 80)

if __name__ == "__main__":
    main()
