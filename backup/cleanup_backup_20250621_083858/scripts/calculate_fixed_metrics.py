#!/usr/bin/env python3
"""
Calculate Fixed Reconstruction Metrics
=====================================

Calculate accurate reconstruction metrics with properly loaded CortexFlow model.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
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

def calculate_fixed_metrics():
    """Calculate metrics with properly loaded models"""
    
    print("📊 CALCULATING FIXED RECONSTRUCTION METRICS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    datasets = ['miyawaki', 'vangerven', 'crell', 'mindbigdata']
    
    all_metrics = {}
    
    for dataset in datasets:
        print(f"\n📊 DATASET: {dataset.upper()}")
        print("-" * 40)
        
        try:
            # Load dataset
            _, _, X_test, y_test, input_dim = load_dataset_gpu_optimized(dataset, device=device)
            
            # Select samples for evaluation
            num_samples = min(20, X_test.shape[0])  # More samples for better metrics
            indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
            X_samples = X_test[indices]
            y_samples = y_test[indices]
            
            original = y_samples.cpu().numpy()
            dataset_metrics = {}
            
            # 1. CortexFlow (FIXED)
            print("   🔧 Loading CortexFlow (FIXED)...")
            model_path = f"models/{dataset}_cv_best.pth"
            if os.path.exists(model_path):
                try:
                    cortexflow_model = CortexFlowFromCheckpoint(input_dim=input_dim, device=device)
                    if cortexflow_model.load_from_checkpoint(model_path):
                        cortexflow_model.eval()
                        
                        with torch.no_grad():
                            cortexflow_pred = cortexflow_model(X_samples)
                        
                        pred = cortexflow_pred.cpu().numpy()
                        
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
                        
                        dataset_metrics['CortexFlow'] = {
                            'mse': float(mse),
                            'correlation': float(correlation),
                            'ssim': float(ssim_mean),
                            'num_samples': int(num_samples),
                            'status': 'success_fixed'
                        }
                        
                        print(f"   ✅ CortexFlow (FIXED): MSE={mse:.6f}, Corr={correlation:.6f}, SSIM={ssim_mean:.6f}")
                    else:
                        print(f"   ❌ CortexFlow: Failed to load weights")
                except Exception as e:
                    print(f"   ❌ CortexFlow error: {e}")
            
            # 2. Brain-Diffuser
            print("   🧠 Loading Brain-Diffuser...")
            bd_path = f"models/{dataset}_brain_diffuser_simplified.pkl"
            if os.path.exists(bd_path):
                try:
                    import pickle
                    with open(bd_path, 'rb') as f:
                        bd_model = pickle.load(f)
                    
                    X_np = X_samples.cpu().numpy()
                    bd_pred_flat = bd_model.predict(X_np)
                    bd_pred = bd_pred_flat.reshape(num_samples, 1, 28, 28)
                    
                    # Calculate metrics
                    mse = np.mean((original - bd_pred) ** 2)
                    correlation = np.corrcoef(original.flatten(), bd_pred.flatten())[0, 1]
                    
                    ssim_scores = []
                    for i in range(original.shape[0]):
                        ssim_score = ssim_simple(original[i, 0], bd_pred[i, 0])
                        ssim_scores.append(ssim_score)
                    
                    ssim_mean = np.mean(ssim_scores)
                    
                    dataset_metrics['Brain-Diffuser'] = {
                        'mse': float(mse),
                        'correlation': float(correlation),
                        'ssim': float(ssim_mean),
                        'num_samples': int(num_samples),
                        'status': 'success'
                    }
                    
                    print(f"   ✅ Brain-Diffuser: MSE={mse:.6f}, Corr={correlation:.6f}, SSIM={ssim_mean:.6f}")
                    
                except Exception as e:
                    print(f"   ❌ Brain-Diffuser error: {e}")
            
            # 3. Mind-Vis
            print("   👁️ Loading Mind-Vis...")
            mv_path = f"sota_comparison/mind_vis/models/{dataset}_mind_vis_best.pth"
            if os.path.exists(mv_path):
                try:
                    from scripts.train_mind_vis_fixed import SimplifiedMindVis
                    output_dim = 28 * 28
                    
                    mv_model = SimplifiedMindVis(input_dim, output_dim, device)
                    mv_model.load_state_dict(torch.load(mv_path, map_location=device))
                    mv_model.eval()
                    
                    with torch.no_grad():
                        mv_pred_flat = mv_model(X_samples)
                        mv_pred = mv_pred_flat.view(num_samples, 1, 28, 28)
                    
                    pred = mv_pred.cpu().numpy()
                    
                    # Calculate metrics
                    mse = np.mean((original - pred) ** 2)
                    correlation = np.corrcoef(original.flatten(), pred.flatten())[0, 1]
                    
                    ssim_scores = []
                    for i in range(original.shape[0]):
                        ssim_score = ssim_simple(original[i, 0], pred[i, 0])
                        ssim_scores.append(ssim_score)
                    
                    ssim_mean = np.mean(ssim_scores)
                    
                    dataset_metrics['Mind-Vis'] = {
                        'mse': float(mse),
                        'correlation': float(correlation),
                        'ssim': float(ssim_mean),
                        'num_samples': int(num_samples),
                        'status': 'success'
                    }
                    
                    print(f"   ✅ Mind-Vis: MSE={mse:.6f}, Corr={correlation:.6f}, SSIM={ssim_mean:.6f}")
                    
                except Exception as e:
                    print(f"   ❌ Mind-Vis error: {e}")
            
            all_metrics[dataset] = dataset_metrics
            
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
    
    # Save fixed metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/reconstruction_metrics_FIXED_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n💾 Fixed metrics saved: {output_file}")
    
    # Print summary
    print(f"\n📊 FIXED METRICS SUMMARY:")
    print("=" * 60)
    
    for dataset, metrics in all_metrics.items():
        print(f"\n🧠 {dataset.upper()}:")
        for method, data in metrics.items():
            if data['status'] == 'success' or data['status'] == 'success_fixed':
                status_icon = "🔧" if data['status'] == 'success_fixed' else "✅"
                print(f"   {status_icon} {method}: MSE={data['mse']:.6f}, Corr={data['correlation']:.6f}, SSIM={data['ssim']:.6f}")
    
    return all_metrics

if __name__ == "__main__":
    calculate_fixed_metrics()
