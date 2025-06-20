"""
CCCV1 Baseline Model for Fair SOTA Comparison
============================================

This is a simplified baseline version of CCCV1 without optimizations,
designed for fair comparison with Brain-Diffuser and Mind-Vis.

Academic Integrity: No optimizations beyond basic neural network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCCV1BaselineEncoder(nn.Module):
    """Simple encoder without optimizations"""
    
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super(CCCV1BaselineEncoder, self).__init__()
        
        # Simple 3-layer encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class CCCV1BaselineDecoder(nn.Module):
    """Simple decoder without optimizations"""
    
    def __init__(self, hidden_dim=512, output_dim=784, dropout=0.1):
        super(CCCV1BaselineDecoder, self).__init__()
        
        # Simple 2-layer decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Output in [0,1] range
        )
    
    def forward(self, x):
        output = self.decoder(x)
        return output.view(-1, 1, 28, 28)  # Reshape to image format


class CCCV1Baseline(nn.Module):
    """
    CCCV1 Baseline Model for Fair Comparison
    
    This is a simplified version without:
    - Dataset-specific optimizations
    - Progressive dropout
    - CLIP guidance
    - Semantic enhancement
    - Advanced normalization
    """
    
    def __init__(self, input_dim, device='cuda'):
        super(CCCV1Baseline, self).__init__()
        self.name = "CCCV1-Baseline"
        self.device = device
        
        # Standard hyperparameters (no dataset-specific tuning)
        self.hidden_dim = 512
        self.dropout = 0.1
        
        # Simple encoder-decoder architecture
        self.encoder = CCCV1BaselineEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(device)
        
        self.decoder = CCCV1BaselineDecoder(
            hidden_dim=self.hidden_dim,
            output_dim=784,  # 28x28 images
            dropout=self.dropout
        ).to(device)
    
    def forward(self, x):
        """Simple forward pass without optimizations"""
        # Encode fMRI to hidden representation
        hidden = self.encoder(x)
        
        # Decode to image
        output = self.decoder(hidden)
        
        return output, hidden  # Return both for compatibility
    
    def get_parameter_count(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_baseline_config():
    """
    Standard configuration for all datasets
    No dataset-specific optimizations
    """
    return {
        'architecture': {
            'hidden_dim': 512,
            'dropout': 0.1,
        },
        'training': {
            'lr': 0.001,           # Standard learning rate
            'batch_size': 16,      # Standard batch size
            'weight_decay': 1e-4,  # Standard weight decay
            'epochs': 100,         # Standard epochs
            'patience': 15,        # Standard patience
            'optimizer': 'Adam',
            'betas': [0.9, 0.999], # Standard Adam betas
            'scheduler_factor': 0.5,
            'gradient_clip': 1.0   # Standard gradient clipping
        }
    }


def create_baseline_model(input_dim, device='cuda'):
    """Factory function to create baseline model"""
    return CCCV1Baseline(input_dim, device)


# Export main classes
__all__ = [
    'CCCV1Baseline',
    'get_baseline_config',
    'create_baseline_model'
]
