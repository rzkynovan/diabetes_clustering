"""
Deep Embedded Clustering (DEC) Implementation
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple

class ClusteringLayer(nn.Module):
    """
    Clustering layer using Student's t-distribution as kernel.
    """
    def __init__(self, n_clusters: int, hidden: int = 16, alpha: float = 1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        
        # Initialize cluster centers
        self.cluster_centers = nn.Parameter(
            torch.zeros(n_clusters, hidden, dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignment using Student's t-distribution.
        
        Args:
            x: Input tensor of shape (batch_size, hidden)
        
        Returns:
            q: Soft cluster assignment of shape (batch_size, n_clusters)
        """
        # Compute squared distances
        # x: (batch_size, hidden)
        # cluster_centers: (n_clusters, hidden)
        # distances: (batch_size, n_clusters)
        distances = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        
        # Student's t-distribution kernel
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        
        # Normalize
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q
    
    def set_cluster_centers(self, centers: np.ndarray):
        """Initialize cluster centers from K-Means."""
        self.cluster_centers.data = torch.tensor(centers, dtype=torch.float32)


class DEC(nn.Module):
    """
    Deep Embedded Clustering (DEC) model.
    
    Args:
        encoder: Encoder network (can be Sequential or custom class)
        n_clusters: Number of clusters
        latent_dim: Dimensionality of latent space (default: 16)
        alpha: Degrees of freedom for Student's t-distribution (default: 1.0)
    """
    def __init__(self, encoder, n_clusters: int, latent_dim: int = 16, alpha: float = 1.0):
        super(DEC, self).__init__()
        self.encoder = encoder
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        
        # Finetune encoder during training
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        # Clustering layer
        self.clustering_layer = ClusteringLayer(
            n_clusters=n_clusters,
            hidden=latent_dim,  # Use explicit parameter
            alpha=alpha
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            z: Latent representation (batch_size, latent_dim)
            q: Soft cluster assignment (batch_size, n_clusters)
        """
        z = self.encoder(x)
        q = self.clustering_layer(z)
        return z, q
    
    def initialize_cluster_centers(self, X: np.ndarray, n_init: int = 20):
        """
        Initialize cluster centers using K-Means++.
        
        Args:
            X: Latent features of shape (n_samples, latent_dim)
            n_init: Number of K-Means initializations
        
        Returns:
            labels: Initial cluster assignments
        """
        print(f"\n🔄 Initializing cluster centers with K-Means...")
        print(f"   Latent features: {X.shape}")
        
        # Run K-Means with multiple initializations
        best_kmeans = None
        best_inertia = float('inf')
        
        for i in range(5):  # Try 5 different seeds
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=n_init,
                max_iter=500,
                random_state=42 + i,
                verbose=0
            )
            kmeans.fit(X)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
        
        # Check if K-Means found valid clusters
        labels = best_kmeans.labels_
        n_unique = len(np.unique(labels))
        
        if n_unique < self.n_clusters:
            raise ValueError(
                f"K-Means found only {n_unique} clusters instead of {self.n_clusters}!"
            )
        
        # Set cluster centers
        self.clustering_layer.set_cluster_centers(best_kmeans.cluster_centers_)
        
        print(f"   ✅ Cluster centers initialized!")
        print(f"   Initial inertia: {best_inertia:.2f}")
        print(f"   Unique clusters: {n_unique}")
        
        # Print cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"   Initial cluster sizes: {dict(zip(unique, counts))}")
        
        return labels


class DECTrainer:
    """
    Trainer for Deep Embedded Clustering.
    """
    def __init__(
        self,
        model: DEC,
        device: str = 'cpu',
        learning_rate: float = 1e-5,
        update_interval: int = 560,
        tol: float = 1e-4,
        max_epochs: int = 300,
        cluster_reg_weight: float = 0.05
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.tol = tol
        self.max_epochs = max_epochs
        self.cluster_reg_weight = cluster_reg_weight
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        
        # LR scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=learning_rate/10
        )
    
    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution P from soft assignment Q.
        """
        # Compute weight per cluster
        weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        
        # Normalize
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        
        return p
    
    def cluster_balance_loss(self, q: torch.Tensor) -> torch.Tensor:
        """
        Regularization term to prevent cluster collapse.
        """
        # Cluster frequencies
        cluster_freq = torch.mean(q, dim=0)
        
        # Uniform distribution
        uniform_dist = torch.ones_like(cluster_freq) / self.model.n_clusters
        
        # KL divergence
        balance_loss = torch.sum(cluster_freq * torch.log(cluster_freq / uniform_dist + 1e-10))
        
        return balance_loss
    
    def fit(self, train_loader, val_loader=None):
        """
        Train DEC model.
        """
        print(f"\n{'='*70}")
        print(f"🚀 Training DEC with k={self.model.n_clusters} clusters")
        print(f"{'='*70}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Update Interval: {self.update_interval} batches")
        print(f"Max Epochs: {self.max_epochs}")
        print(f"Tolerance: {self.tol}")
        print(f"Cluster Regularization Weight: {self.cluster_reg_weight}")
        
        self.model.train()
        
        # Initialize target distribution
        target_p = None
        
        # Training history
        history = {
            'epoch': [],
            'loss': [],
            'cluster_change': [],
            'cluster_sizes': []
        }
        
        batch_idx = 0
        prev_labels = None
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_x in train_loader:
                batch_x = batch_x[0].to(self.device) if isinstance(batch_x, (tuple, list)) else batch_x.to(self.device)
                
                # Forward pass
                z, q = self.model(batch_x)
                
                # Update target distribution every `update_interval` batches
                if batch_idx % self.update_interval == 0:
                    self.model.eval()
                    q_full = []
                    with torch.no_grad():
                        for batch in train_loader:
                            batch = batch[0].to(self.device) if isinstance(batch, (tuple, list)) else batch.to(self.device)
                            _, q_batch = self.model(batch)
                            q_full.append(q_batch.cpu())
                    q_full = torch.cat(q_full, dim=0)
                    target_p = self.target_distribution(q_full).to(self.device)
                    
                    # Check cluster assignments
                    current_labels = torch.argmax(q_full, dim=1).numpy()
                    n_unique = len(np.unique(current_labels))
                    
                    if n_unique == 1:
                        print(f"   ⚠️  WARNING: Only 1 cluster at batch {batch_idx}!")
                    
                    # Compute cluster change
                    if prev_labels is not None:
                        delta = np.sum(current_labels != prev_labels) / len(current_labels)
                        history['cluster_change'].append(delta)
                        
                        if delta < self.tol:
                            print(f"\n   ✅ Converged at epoch {epoch+1}, batch {batch_idx}")
                            print(f"      Cluster change: {delta:.6f} < tolerance {self.tol}")
                            break
                    
                    prev_labels = current_labels
                    
                    # Cluster sizes
                    unique, counts = np.unique(current_labels, return_counts=True)
                    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
                    history['cluster_sizes'].append(cluster_sizes)
                    
                    self.model.train()
                
                # Compute loss
                batch_size = batch_x.size(0)
                batch_start = (batch_idx % len(train_loader)) * train_loader.batch_size
                batch_end = batch_start + batch_size
                
                if target_p is not None and batch_end <= len(target_p):
                    p_batch = target_p[batch_start:batch_end]
                    
                    # KL divergence loss
                    kl_loss = torch.mean(
                        torch.sum(p_batch * torch.log(p_batch / (q + 1e-10) + 1e-10), dim=1)
                    )
                    
                    # Cluster balance regularization
                    balance_loss = self.cluster_balance_loss(q)
                    
                    # Total loss
                    loss = kl_loss + self.cluster_reg_weight * balance_loss
                    
                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                batch_idx += 1
            
            # Epoch summary
            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                history['epoch'].append(epoch + 1)
                history['loss'].append(avg_loss)
                
                print(f"   Epoch {epoch+1}/{self.max_epochs} - Loss: {avg_loss:.6f} - LR: {self.scheduler.get_last_lr()[0]:.2e}")
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"   ⏹️  Early stopping at epoch {epoch+1}")
                    break
            
            # Step scheduler
            self.scheduler.step()
            
            # Check convergence
            if len(history['cluster_change']) > 0 and history['cluster_change'][-1] < self.tol:
                break
        
        print(f"\n{'='*70}")
        print(f"✅ Training Complete!")
        print(f"   Final Loss: {history['loss'][-1]:.6f}")
        print(f"   Total Epochs: {len(history['epoch'])}")
        if len(history['cluster_change']) > 0:
            print(f"   Final Cluster Change: {history['cluster_change'][-1]:.6f}")
        print(f"{'='*70}\n")
        
        return history
    
    def predict(self, data_loader) -> np.ndarray:
        """
        Predict cluster labels.
        """
        self.model.eval()
        
        labels = []
        with torch.no_grad():
            for batch_x in data_loader:
                batch_x = batch_x[0].to(self.device) if isinstance(batch_x, (tuple, list)) else batch_x.to(self.device)
                _, q = self.model(batch_x)
                batch_labels = torch.argmax(q, dim=1).cpu().numpy()
                labels.append(batch_labels)
        
        return np.concatenate(labels)
