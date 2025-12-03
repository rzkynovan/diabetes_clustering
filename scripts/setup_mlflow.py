# ============================================================================
# MLFLOW SETUP & INITIALIZATION
# ============================================================================
import mlflow
import mlflow.sklearn
from pathlib import Path

# Setup MLflow tracking URI
MLFLOW_TRACKING_URI = Path("experiments/mlruns").absolute().as_uri()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f"✅ MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

# Create experiment for baseline clustering
EXPERIMENT_NAME = "diabetes-baseline-clustering"

try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=Path("experiments/artifacts").absolute().as_uri(),
            tags={
                "project": "Diabetes Readmission Clustering",
                "stage": "baseline",
                "team": "research",
                "version": "1.0"
            }
        )
        print(f"✅ Created new experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"✅ Using existing experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
except Exception as e:
    print(f"⚠️  Error setting up MLflow: {e}")
    print("   Creating experiment with default settings...")
    mlflow.set_experiment(EXPERIMENT_NAME)

print(f"\n📊 MLflow UI: Run 'mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}'")
print(f"   Then open: http://localhost:5000")
