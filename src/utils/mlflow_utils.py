"""
MLflow Utilities
Helper functions for MLflow experiment tracking
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator


def init_mlflow(experiment_name: str = "diabetes-clustering",
                tracking_uri: Optional[str] = None) -> str:
    """
    Initialize MLflow tracking
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: Optional custom tracking URI
        
    Returns:
        experiment_id: ID of the experiment
    """
    if tracking_uri is None:
        tracking_uri = Path("experiments/mlruns").absolute().as_uri()
    
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=Path("experiments/artifacts").absolute().as_uri()
            )
            print(f"✅ Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id
    
    except Exception as e:
        print(f"⚠️  Warning: {e}")
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return experiment.experiment_id if experiment else None


def log_clustering_run(
    run_name: str,
    model: BaseEstimator,
    labels: np.ndarray,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    artifacts: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Log a clustering run to MLflow
    
    Args:
        run_name: Name for the run
        model: Trained clustering model
        labels: Cluster labels
        metrics: Dictionary of metrics
        params: Dictionary of parameters
        artifacts: Optional artifacts to log (figures, etc.)
        tags: Optional tags for the run
        
    Returns:
        run_id: MLflow run ID
    """
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                print(f"⚠️  Could not log param {key}: {e}")
        
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                try:
                    mlflow.log_metric(key, float(value))
                except Exception as e:
                    print(f"⚠️  Could not log metric {key}: {e}")
        
        # Log tags
        if tags:
            try:
                mlflow.set_tags(tags)
            except Exception as e:
                print(f"⚠️  Could not set tags: {e}")
        
        # Log cluster distribution
        try:
            cluster_dist = pd.Series(labels).value_counts().to_dict()
            for cluster_id, count in cluster_dist.items():
                if cluster_id != -1:  # Skip noise cluster for DBSCAN
                    mlflow.log_metric(f"cluster_{cluster_id}_size", int(count))
        except Exception as e:
            print(f"⚠️  Could not log cluster distribution: {e}")
        
        # Log model
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"⚠️  Could not log model: {e}")
        
        # Log artifacts
        if artifacts:
            artifact_dir = Path("temp_artifacts")
            artifact_dir.mkdir(exist_ok=True)
            
            for name, artifact in artifacts.items():
                try:
                    if isinstance(artifact, plt.Figure):
                        artifact_path = artifact_dir / f"{name}.png"
                        artifact.savefig(artifact_path, dpi=300, bbox_inches='tight')
                        mlflow.log_artifact(str(artifact_path))
                        artifact_path.unlink()
                    elif isinstance(artifact, (str, Path)):
                        mlflow.log_artifact(str(artifact))
                except Exception as e:
                    print(f"⚠️  Could not log artifact {name}: {e}")
            
            try:
                artifact_dir.rmdir()
            except:
                pass
        
        run_id = mlflow.active_run().info.run_id
        print(f"  ✅ Logged: {run_name} (ID: {run_id[:8]}...)")
        
        return run_id


def compare_runs(experiment_name: str, 
                metric: str = "silhouette_score",
                top_n: int = 10) -> pd.DataFrame:
    """Compare runs in an experiment"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    
    if len(runs) == 0:
        return pd.DataFrame()
    
    cols = ['run_id', 'start_time']
    if 'tags.mlflow.runName' in runs.columns:
        cols.append('tags.mlflow.runName')
    
    param_cols = [c for c in runs.columns if c.startswith('params.')]
    metric_cols = [c for c in runs.columns if c.startswith('metrics.')]
    cols.extend(param_cols + metric_cols)
    cols = [c for c in cols if c in runs.columns]
    
    return runs[cols].head(top_n)
