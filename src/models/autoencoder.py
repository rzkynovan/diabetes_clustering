"""
Autoencoder for Deep Embedded Clustering (DEC)
Implements symmetric encoder-decoder architecture with bottleneck latent space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder with bottleneck architecture
    
    Architecture:
        Encoder: input_dim → 128 → 64 → 32 → latent_dim
        Decoder: latent_dim → 32 → 64 → 128 → input_dim
    """
    
    def __init__(self, input_dim=69, latent_dim=16, hidden_dims=[128, 64, 32]):
        """
        Args:
            input_dim (int): Number of input features (default: 69)
            latent_dim (int): Dimension of latent space (default: 16)
            hidden_dims (list): Hidden layer dimensions (default: [128, 64, 32])
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # ============================================================
        # Encoder
        # ============================================================
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck layer (no activation)
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ============================================================
        # Decoder (symmetric to encoder)
        # ============================================================
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode input to latent representation
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Latent representation [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation to reconstruction
        
        Args:
            z (torch.Tensor): Latent tensor [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Reconstructed output [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass: encode then decode
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            tuple: (reconstruction, latent_representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_latent_features(self, x):
        """
        Extract latent features (for clustering)
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            numpy.ndarray: Latent features [batch_size, latent_dim]
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
        return z.cpu().numpy()


class AutoencoderTrainer:
    """
    Trainer class for autoencoder with early stopping and logging
    """
    
    def __init__(self, model, device='cpu', learning_rate=1e-3, weight_decay=1e-5):
        """
        Args:
            model (nn.Module): Autoencoder model
            device (str): 'cpu' or 'cuda'
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            # 🔧 FIX: Extract tensor from tuple
            batch_x = batch[0].to(self.device)  # batch is a tuple (tensor,)
            
            # Forward pass
            x_recon, z = self.model(batch_x)
            loss = self.criterion(x_recon, batch_x)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss

    def validate(self, val_loader):
        """
        Validate on validation set
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 🔧 FIX: Extract tensor from tuple
                batch_x = batch[0].to(self.device)  # batch is a tuple (tensor,)
                
                x_recon, z = self.model(batch_x)
                loss = self.criterion(x_recon, batch_x)
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss

    def fit(self, train_loader, val_loader, epochs=100, patience=10, verbose=True):
        """
        Train autoencoder with early stopping
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Maximum number of epochs
            patience (int): Early stopping patience
            verbose (bool): Print progress
            
        Returns:
            dict: Training history
        """
        print(f"Training autoencoder on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*70)
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best validation loss: {self.best_val_loss:.6f}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        print("="*70)
        print(f"✅ Training completed!")
        print(f"   Final train loss: {self.train_losses[-1]:.6f}")
        print(f"   Best val loss: {self.best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'n_epochs': len(self.train_losses)
        }
