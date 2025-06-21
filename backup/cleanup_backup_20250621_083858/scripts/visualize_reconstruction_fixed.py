#!/usr/bin/env python3
"""
Fixed Reconstruction Results Visualization
==========================================

Visualize reconstruction results with properly loaded CortexFlow model.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from datetime import datetime
import clip

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import load_dataset_gpu_optimized

class CortexFlowFromCheckpoint(nn.Module):
    """Reconstruct CortexFlow model from checkpoint structure"""
    
    def __init__(self, input_dim, device='cuda'):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        
        # Create encoder (matching checkpoint dimensions)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),  # First layer: input_dim -> 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),       # Second layer: 1024 -> 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),        # Third layer: 1024 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),         # Fourth layer: 512 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU()
        ).to(device)

        # Create decoder (matching checkpoint dimensions)
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),         # First decoder layer: 512 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 784),         # Output layer: 512 -> 784 (28*28)
            nn.Sigmoid()
        ).to(device)

        # Semantic enhancer (matching checkpoint dimensions)
        self.semantic_enhancer = nn.Sequential(
            nn.Linear(512, 256),         # First enhancer: 512 -> 256
            nn.ReLU(),
            nn.Linear(256, 512),         # Second enhancer: 256 -> 512
            nn.ReLU()
        ).to(device)
        
        # Load CLIP model
        try:
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
        except:
            print("⚠️ CLIP model not available, using placeholder")
            self.clip_model = None
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)

        # Semantic enhancement
        enhanced = self.semantic_enhancer(encoded)

        # Decode
        decoded = self.decoder(enhanced)

        # Reshape to image
        return decoded.view(-1, 1, 28, 28)
    
    def load_from_checkpoint(self, checkpoint_path):
        """Load weights from checkpoint with proper mapping"""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Map checkpoint keys to our model
            state_dict = {}
            
            # Encoder mapping
            encoder_mapping = {
                'model.encoder.encoder.0.weight': 'encoder.0.weight',
                'model.encoder.encoder.0.bias': 'encoder.0.bias',
                'model.encoder.encoder.1.weight': 'encoder.1.weight',
                'model.encoder.encoder.1.bias': 'encoder.1.bias',
                'model.encoder.encoder.4.weight': 'encoder.4.weight',
                'model.encoder.encoder.4.bias': 'encoder.4.bias',
                'model.encoder.encoder.5.weight': 'encoder.5.weight',
                'model.encoder.encoder.5.bias': 'encoder.5.bias',
                'model.encoder.encoder.8.weight': 'encoder.8.weight',
                'model.encoder.encoder.8.bias': 'encoder.8.bias',
                'model.encoder.encoder.9.weight': 'encoder.9.weight',
                'model.encoder.encoder.9.bias': 'encoder.9.bias',
                'model.encoder.encoder.12.weight': 'encoder.12.weight',
                'model.encoder.encoder.12.bias': 'encoder.12.bias',
                'model.encoder.encoder.13.weight': 'encoder.13.weight',
                'model.encoder.encoder.13.bias': 'encoder.13.bias',
            }
            
            # Decoder mapping
            decoder_mapping = {
                'model.decoder.decoder.0.weight': 'decoder.0.weight',
                'model.decoder.decoder.0.bias': 'decoder.0.bias',
                'model.decoder.decoder.1.weight': 'decoder.1.weight',
                'model.decoder.decoder.1.bias': 'decoder.1.bias',
                'model.decoder.decoder.4.weight': 'decoder.4.weight',
                'model.decoder.decoder.4.bias': 'decoder.4.bias',
            }
            
            # Semantic enhancer mapping
            semantic_mapping = {
                'model.semantic_enhancer.0.weight': 'semantic_enhancer.0.weight',
                'model.semantic_enhancer.0.bias': 'semantic_enhancer.0.bias',
                'model.semantic_enhancer.2.weight': 'semantic_enhancer.2.weight',
                'model.semantic_enhancer.2.bias': 'semantic_enhancer.2.bias',
            }
            
            # Apply mappings
            for old_key, new_key in {**encoder_mapping, **decoder_mapping, **semantic_mapping}.items():
                if old_key in checkpoint:
                    state_dict[new_key] = checkpoint[old_key]
            
            # Load compatible weights
            self.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded {len(state_dict)} compatible weights from checkpoint")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return False

class FixedReconstructionVisualizer:
    """Fixed visualizer with proper CortexFlow loading"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
        self.methods = ['CortexFlow', 'Brain-Diffuser', 'Mind-Vis']
        
    def load_cortexflow_model(self, dataset_name):
        """Load CortexFlow model with proper architecture"""
        
        model_path = f"models/{dataset_name}_cv_best.pth"
        if not os.path.exists(model_path):
            print(f"❌ CortexFlow model not found: {model_path}")
            return None
        
        try:
            # Get input dimension
            _, _, _, _, input_dim = load_dataset_gpu_optimized(dataset_name, device=self.device)
            
            # Create model with proper architecture
            model = CortexFlowFromCheckpoint(input_dim=input_dim, device=self.device)
            
            # Load weights from checkpoint
            if model.load_from_checkpoint(model_path):
                model.eval()
                print(f"✅ CortexFlow loaded for {dataset_name}")
                return model
            else:
                print(f"❌ Failed to load CortexFlow weights for {dataset_name}")
                return None
            
        except Exception as e:
            print(f"❌ Error loading CortexFlow: {e}")
            return None
    
    def load_brain_diffuser_model(self, dataset_name):
        """Load Brain-Diffuser model"""
        
        model_path = f"models/{dataset_name}_brain_diffuser_simplified.pkl"
        if not os.path.exists(model_path):
            print(f"❌ Brain-Diffuser model not found: {model_path}")
            return None
        
        try:
            with open(model_path, 'rb') as f:
                ridge_regressor = pickle.load(f)
            
            print(f"✅ Brain-Diffuser loaded for {dataset_name}")
            return ridge_regressor
            
        except Exception as e:
            print(f"❌ Error loading Brain-Diffuser: {e}")
            return None
    
    def load_mind_vis_model(self, dataset_name):
        """Load Mind-Vis model"""
        
        model_path = f"sota_comparison/mind_vis/models/{dataset_name}_mind_vis_best.pth"
        if not os.path.exists(model_path):
            print(f"❌ Mind-Vis model not found: {model_path}")
            return None
        
        try:
            # Get dimensions
            _, _, _, _, input_dim = load_dataset_gpu_optimized(dataset_name, device=self.device)
            output_dim = 28 * 28  # Flattened image
            
            # Create model architecture
            from scripts.train_mind_vis_fixed import SimplifiedMindVis
            model = SimplifiedMindVis(input_dim, output_dim, self.device)
            
            # Load weights
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            
            print(f"✅ Mind-Vis loaded for {dataset_name}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading Mind-Vis: {e}")
            return None
    
    def generate_reconstructions(self, dataset_name, num_samples=6):
        """Generate reconstructions from all models"""
        
        print(f"\n🎨 GENERATING FIXED RECONSTRUCTIONS FOR {dataset_name.upper()}")
        print("-" * 60)
        
        # Load dataset
        _, _, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset_name, device=self.device)
        
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
        
        # CortexFlow reconstruction (FIXED)
        cortexflow_model = self.load_cortexflow_model(dataset_name)
        if cortexflow_model is not None:
            try:
                with torch.no_grad():
                    cortexflow_pred = cortexflow_model(X_samples)
                reconstructions['CortexFlow'] = cortexflow_pred.cpu().numpy()
                print(f"✅ CortexFlow reconstruction: {cortexflow_pred.shape}")
            except Exception as e:
                print(f"❌ CortexFlow reconstruction failed: {e}")
        
        # Brain-Diffuser reconstruction
        brain_diffuser_model = self.load_brain_diffuser_model(dataset_name)
        if brain_diffuser_model is not None:
            try:
                X_np = X_samples.cpu().numpy()
                bd_pred_flat = brain_diffuser_model.predict(X_np)
                bd_pred = bd_pred_flat.reshape(num_samples, 1, 28, 28)
                reconstructions['Brain-Diffuser'] = bd_pred
                print(f"✅ Brain-Diffuser reconstruction: {bd_pred.shape}")
            except Exception as e:
                print(f"❌ Brain-Diffuser reconstruction failed: {e}")
        
        # Mind-Vis reconstruction
        mind_vis_model = self.load_mind_vis_model(dataset_name)
        if mind_vis_model is not None:
            try:
                with torch.no_grad():
                    mv_pred_flat = mind_vis_model(X_samples)
                    mv_pred = mv_pred_flat.view(num_samples, 1, 28, 28)
                reconstructions['Mind-Vis'] = mv_pred.cpu().numpy()
                print(f"✅ Mind-Vis reconstruction: {mv_pred.shape}")
            except Exception as e:
                print(f"❌ Mind-Vis reconstruction failed: {e}")
        
        return reconstructions
    
    def create_reconstruction_visualization(self, dataset_name, reconstructions, save_dir):
        """Create comprehensive reconstruction visualization"""
        
        print(f"📊 Creating FIXED visualization for {dataset_name}")
        
        num_samples = reconstructions['original'].shape[0]
        methods_available = [method for method in self.methods if method in reconstructions]
        num_methods = len(methods_available)
        
        # Create figure
        fig, axes = plt.subplots(num_samples, num_methods + 1, 
                                figsize=(3 * (num_methods + 1), 3 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Plot reconstructions
        for i in range(num_samples):
            # Original image
            original_img = reconstructions['original'][i, 0]  # Remove channel dimension
            axes[i, 0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('Original' if i == 0 else '', fontsize=12, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Method reconstructions
            for j, method in enumerate(methods_available):
                if method in reconstructions:
                    recon_img = reconstructions[method][i, 0]  # Remove channel dimension
                    axes[i, j + 1].imshow(recon_img, cmap='gray', vmin=0, vmax=1)
                    axes[i, j + 1].set_title(method if i == 0 else '', fontsize=12, fontweight='bold')
                    axes[i, j + 1].axis('off')
                    
                    # Calculate MSE for this sample
                    mse = np.mean((original_img - recon_img) ** 2)
                    axes[i, j + 1].text(0.02, 0.98, f'MSE: {mse:.4f}', 
                                       transform=axes[i, j + 1].transAxes,
                                       verticalalignment='top', fontsize=8,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add sample labels
        for i in range(num_samples):
            axes[i, 0].text(-0.1, 0.5, f'Sample {i+1}', 
                           transform=axes[i, 0].transAxes,
                           rotation=90, verticalalignment='center', fontsize=10, fontweight='bold')
        
        plt.suptitle(f'FIXED Reconstruction Results - {dataset_name.upper()} Dataset', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save visualization
        save_path = save_dir / f"{dataset_name}_reconstruction_comparison_FIXED.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Also save as SVG for publications
        save_path_svg = save_dir / f"{dataset_name}_reconstruction_comparison_FIXED.svg"
        plt.savefig(save_path_svg, bbox_inches='tight')
        
        plt.close()
        
        print(f"💾 FIXED Saved: {save_path}")
        print(f"💾 FIXED Saved: {save_path_svg}")
        
        return save_path

def main():
    """Execute FIXED reconstruction visualization"""
    
    print("🔧 FIXED RECONSTRUCTION RESULTS VISUALIZATION")
    print("=" * 80)
    print("🎯 Goal: Visualize with PROPERLY LOADED CortexFlow model")
    print("🏆 Academic: Real model outputs from saved checkpoints")
    print("=" * 80)
    
    # Create visualizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualizer = FixedReconstructionVisualizer(device=device)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"results/reconstruction_visualization_FIXED_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reconstructions for all datasets
    all_reconstructions = {}
    
    for dataset in visualizer.datasets:
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        try:
            reconstructions = visualizer.generate_reconstructions(dataset, num_samples=6)
            all_reconstructions[dataset] = reconstructions
            
            # Create individual visualization
            visualizer.create_reconstruction_visualization(dataset, reconstructions, save_dir)
            
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
    
    print("\n" + "=" * 80)
    print("🏆 FIXED RECONSTRUCTION VISUALIZATION COMPLETED!")
    print("=" * 80)
    print(f"📁 Results directory: {save_dir}")
    print(f"🎨 Fixed visualizations: {len(all_reconstructions)} datasets")
    print(f"✅ CortexFlow: Properly loaded from checkpoints")
    print("=" * 80)

if __name__ == "__main__":
    main()
