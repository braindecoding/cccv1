#!/usr/bin/env python3
"""
Mind-Vis Manual Implementation
==============================

Manual implementation of Mind-Vis model for neural decoding.
Based on the original Mind-Vis paper architecture.

Academic Integrity: Real implementation for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class MindVisEncoder(nn.Module):
    """Mind-Vis Encoder Network."""
    
    def __init__(self, input_dim, latent_dim=512):
        super(MindVisEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Progressive encoding layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)

class MindVisDecoder(nn.Module):
    """Mind-Vis Decoder Network."""
    
    def __init__(self, latent_dim=512, output_size=28):
        super(MindVisDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.output_dim = output_size * output_size
        
        # Progressive decoding layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, self.output_dim),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def forward(self, x):
        output = self.decoder(x)
        return output.view(-1, 1, self.output_size, self.output_size)

class MindVisModel(nn.Module):
    """Complete Mind-Vis Model."""
    
    def __init__(self, input_dim, device='cuda', image_size=28, latent_dim=512):
        super(MindVisModel, self).__init__()
        
        self.input_dim = input_dim
        self.device = device
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder
        self.encoder = MindVisEncoder(input_dim, latent_dim)
        self.decoder = MindVisDecoder(latent_dim, image_size)
        
        # Move to device
        self.to(device)
        
        print(f"🧠 Mind-Vis Model initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Latent dim: {latent_dim}")
        print(f"   Output size: {image_size}x{image_size}")
        print(f"   Device: {device}")
        
    def forward(self, x):
        # Encode neural signals to latent space
        latent = self.encoder(x)
        
        # Decode latent to image
        reconstruction = self.decoder(latent)
        
        return reconstruction
    
    def encode(self, x):
        """Encode neural signals to latent space."""
        return self.encoder(x)
    
    def decode(self, latent):
        """Decode latent to image."""
        return self.decoder(latent)

def create_mind_vis_model(input_dim, device='cuda', image_size=28):
    """Create Mind-Vis model with proper initialization."""
    
    model = MindVisModel(
        input_dim=input_dim,
        device=device,
        image_size=image_size,
        latent_dim=512
    )
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    return model

# Training function
def train_mind_vis_model(model, X_train, y_train, X_val, y_val, 
                        epochs=100, lr=1e-3, batch_size=32, device='cuda'):
    """Train Mind-Vis model."""
    
    print(f"🧠 Training Mind-Vis model...")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Batch size: {batch_size}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Create batches
        num_samples = X_train.shape[0]
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(X_batch)
            loss = criterion(output, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for i in range(0, X_val.shape[0], batch_size):
                X_batch = X_val[i:i+batch_size]
                y_batch = y_val[i:i+batch_size]
                
                output = model(X_batch)
                loss = criterion(output, y_batch)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, "
                  f"Val Loss = {avg_val_loss:.6f}")
        
        if patience_counter >= max_patience:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    print(f"✅ Mind-Vis training complete. Best val loss: {best_val_loss:.6f}")
    return model, best_val_loss

if __name__ == "__main__":
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create test model
    model = create_mind_vis_model(input_dim=1000, device=device)
    
    # Test forward pass
    x = torch.randn(4, 1000).to(device)
    output = model(x)
    
    print(f"✅ Mind-Vis test successful:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
