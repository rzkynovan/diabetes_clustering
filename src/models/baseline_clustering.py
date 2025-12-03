"""
Baseline Clustering Models
Traditional clustering algorithms for benchmarking
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    silhouette_samples
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class BaselineClustering:
    """
    Wrapper class for traditional clustering algorithms
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.metrics_ = {}
        
    def fit_kmeans(self, X: np.ndarray, n_clusters: int, **kwargs) -> 'BaselineClustering':
        """Fit K-Means clustering"""
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300,
            **kwargs
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_dbscan(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 5, 
                   **kwargs) -> 'BaselineClustering':
        """Fit DBSCAN clustering"""
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1,
            **kwargs
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_hierarchical(self, X: np.ndarray, n_clusters: int, 
                        linkage: str = 'ward', **kwargs) -> 'BaselineClustering':
        """Fit Agglomerative Hierarchical clustering"""
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            **kwargs
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_gmm(self, X: np.ndarray, n_components: int, 
                covariance_type: str = 'full', **kwargs) -> 'BaselineClustering':
        """Fit Gaussian Mixture Model"""
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=self.random_state,
            n_init=10,
            max_iter=200,
            **kwargs
        )
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        return self
    
    def evaluate(self, X: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering performance with multiple metrics
        
        Returns:
            Dictionary of metric scores
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted first!")
        
        # Filter out noise points (label = -1 for DBSCAN)
        mask = self.labels_ != -1
        X_filtered = X[mask]
        labels_filtered = self.labels_[mask]
        
        n_clusters = len(np.unique(labels_filtered))
        n_noise = np.sum(self.labels_ == -1)
        
        self.metrics_ = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_ratio': n_noise / len(self.labels_)
        }
        
        # Only compute metrics if we have valid clusters
        if n_clusters >= 2 and len(labels_filtered) > n_clusters:
            try:
                self.metrics_['silhouette_score'] = silhouette_score(
                    X_filtered, labels_filtered, metric='euclidean'
                )
            except:
                self.metrics_['silhouette_score'] = -1.0
            
            try:
                self.metrics_['davies_bouldin_score'] = davies_bouldin_score(
                    X_filtered, labels_filtered
                )
            except:
                self.metrics_['davies_bouldin_score'] = np.inf
            
            try:
                self.metrics_['calinski_harabasz_score'] = calinski_harabasz_score(
                    X_filtered, labels_filtered
                )
            except:
                self.metrics_['calinski_harabasz_score'] = 0.0
        else:
            self.metrics_['silhouette_score'] = -1.0
            self.metrics_['davies_bouldin_score'] = np.inf
            self.metrics_['calinski_harabasz_score'] = 0.0
        
        # Add model-specific metrics
        if hasattr(self.model, 'inertia_'):
            self.metrics_['inertia'] = self.model.inertia_
        
        if hasattr(self.model, 'bic'):
            self.metrics_['bic'] = self.model.bic(X)
            self.metrics_['aic'] = self.model.aic(X)
        
        return self.metrics_
    
    def get_cluster_sizes(self) -> pd.Series:
        """Get cluster size distribution"""
        if self.labels_ is None:
            raise ValueError("Model must be fitted first!")
        return pd.Series(self.labels_).value_counts().sort_index()


class ClusteringExperiment:
    """
    Run and compare multiple clustering algorithms
    """
    
    def __init__(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None,
                 y_train: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                 random_state: int = 42):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.random_state = random_state
        self.results = {}
        
    def run_kmeans_sweep(self, k_range: range) -> pd.DataFrame:
        """
        Run K-Means with different k values
        
        Args:
            k_range: Range of k values to try
            
        Returns:
            DataFrame with results for each k
        """
        results = []
        
        for k in k_range:
            print(f"\n  Testing K-Means with k={k}...")
            
            clusterer = BaselineClustering(random_state=self.random_state)
            clusterer.fit_kmeans(self.X_train, n_clusters=k)
            metrics = clusterer.evaluate(self.X_train)
            
            result = {
                'k': k,
                'algorithm': 'K-Means',
                **metrics
            }
            results.append(result)
            
            # Store model
            self.results[f'kmeans_k{k}'] = {
                'model': clusterer,
                'metrics': metrics,
                'labels': clusterer.labels_
            }
        
        return pd.DataFrame(results)
    
    def run_dbscan_sweep(self, eps_range: List[float], 
                         min_samples_range: List[int]) -> pd.DataFrame:
        """
        Run DBSCAN with different parameter combinations
        
        Args:
            eps_range: List of epsilon values
            min_samples_range: List of min_samples values
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                print(f"\n  Testing DBSCAN with eps={eps}, min_samples={min_samples}...")
                
                clusterer = BaselineClustering(random_state=self.random_state)
                clusterer.fit_dbscan(self.X_train, eps=eps, min_samples=min_samples)
                metrics = clusterer.evaluate(self.X_train)
                
                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'algorithm': 'DBSCAN',
                    **metrics
                }
                results.append(result)
                
                # Store model
                key = f'dbscan_eps{eps}_ms{min_samples}'
                self.results[key] = {
                    'model': clusterer,
                    'metrics': metrics,
                    'labels': clusterer.labels_
                }
        
        return pd.DataFrame(results)
    
    def run_hierarchical_sweep(self, k_range: range, 
                               linkage_types: List[str] = ['ward', 'complete', 'average']) -> pd.DataFrame:
        """
        Run Hierarchical clustering with different parameters
        
        Args:
            k_range: Range of k values
            linkage_types: List of linkage methods
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for linkage in linkage_types:
            for k in k_range:
                print(f"\n  Testing Hierarchical with k={k}, linkage={linkage}...")
                
                clusterer = BaselineClustering(random_state=self.random_state)
                clusterer.fit_hierarchical(self.X_train, n_clusters=k, linkage=linkage)
                metrics = clusterer.evaluate(self.X_train)
                
                result = {
                    'k': k,
                    'linkage': linkage,
                    'algorithm': 'Hierarchical',
                    **metrics
                }
                results.append(result)
                
                # Store model
                key = f'hierarchical_k{k}_{linkage}'
                self.results[key] = {
                    'model': clusterer,
                    'metrics': metrics,
                    'labels': clusterer.labels_
                }
        
        return pd.DataFrame(results)
    
    def run_gmm_sweep(self, k_range: range,
                      covariance_types: List[str] = ['full', 'tied', 'diag']) -> pd.DataFrame:
        """
        Run Gaussian Mixture Model with different parameters
        
        Args:
            k_range: Range of component numbers
            covariance_types: List of covariance types
            
        Returns:
            DataFrame with results
        """
        results = []
        
        for cov_type in covariance_types:
            for k in k_range:
                print(f"\n  Testing GMM with k={k}, covariance={cov_type}...")
                
                clusterer = BaselineClustering(random_state=self.random_state)
                clusterer.fit_gmm(self.X_train, n_components=k, covariance_type=cov_type)
                metrics = clusterer.evaluate(self.X_train)
                
                result = {
                    'k': k,
                    'covariance_type': cov_type,
                    'algorithm': 'GMM',
                    **metrics
                }
                results.append(result)
                
                # Store model
                key = f'gmm_k{k}_{cov_type}'
                self.results[key] = {
                    'model': clusterer,
                    'metrics': metrics,
                    'labels': clusterer.labels_
                }
        
        return pd.DataFrame(results)
    
    def get_best_models(self, metric: str = 'silhouette_score', 
                       top_n: int = 5) -> pd.DataFrame:
        """
        Get top N models based on a specific metric
        
        Args:
            metric: Metric to sort by
            top_n: Number of top models to return
            
        Returns:
            DataFrame with top models
        """
        results_list = []
        
        for key, value in self.results.items():
            result = {
                'model_name': key,
                **value['metrics']
            }
            results_list.append(result)
        
        df = pd.DataFrame(results_list)
        
        # Sort based on metric (lower is better for davies_bouldin)
        if metric == 'davies_bouldin_score':
            df_sorted = df.sort_values(metric, ascending=True)
        else:
            df_sorted = df.sort_values(metric, ascending=False)
        
        return df_sorted.head(top_n)


def plot_elbow_curve(results_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 5)):
    """
    Plot elbow curve for K-Means results
    
    Args:
        results_df: DataFrame with K-Means results
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    k_values = results_df['k'].values
    
    # Inertia (SSE)
    if 'inertia' in results_df.columns:
        axes[0].plot(k_values, results_df['inertia'], marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontweight='bold')
        axes[0].set_ylabel('Inertia (SSE)', fontweight='bold')
        axes[0].set_title('Elbow Method: Inertia', fontweight='bold')
        axes[0].grid(alpha=0.3)
    
    # Silhouette Score
    axes[1].plot(k_values, results_df['silhouette_score'], marker='o', 
                linewidth=2, markersize=8, color='#4ECDC4')
    axes[1].set_xlabel('Number of Clusters (k)', fontweight='bold')
    axes[1].set_ylabel('Silhouette Score', fontweight='bold')
    axes[1].set_title('Silhouette Score vs k', fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Davies-Bouldin Index
    axes[2].plot(k_values, results_df['davies_bouldin_score'], marker='o',
                linewidth=2, markersize=8, color='#FF6B6B')
    axes[2].set_xlabel('Number of Clusters (k)', fontweight='bold')
    axes[2].set_ylabel('Davies-Bouldin Index', fontweight='bold')
    axes[2].set_title('Davies-Bouldin Index vs k (lower is better)', fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cluster_distribution(labels: np.ndarray, title: str = "Cluster Distribution",
                              figsize: Tuple[int, int] = (10, 6)):
    """
    Plot cluster size distribution
    
    Args:
        labels: Cluster labels
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_counts)))
    bars = ax.bar(cluster_counts.index, cluster_counts.values, color=colors,
                  edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Cluster ID', fontweight='bold', fontsize=12)
    ax.set_ylabel('Number of Samples', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_pca_clusters(X: np.ndarray, labels: np.ndarray, 
                     title: str = "PCA Visualization of Clusters",
                     figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize clusters in 2D using PCA
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        title: Plot title
        figsize: Figure size
    """
    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            # Noise points for DBSCAN
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c='gray', marker='x', s=30, alpha=0.5, label='Noise')
        else:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[colors[i]], s=50, alpha=0.6, 
                      edgecolors='black', linewidth=0.5,
                      label=f'Cluster {label}')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                  fontweight='bold', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', 
                  fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig
