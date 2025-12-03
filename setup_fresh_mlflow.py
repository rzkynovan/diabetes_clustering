# setup_fresh_mlflow.py
import mlflow
from pathlib import Path
import shutil

# Get absolute paths
project_root = Path.cwd().resolve()
mlruns_dir = project_root / 'experiments' / 'mlruns'
artifacts_dir = project_root / 'experiments' / 'artifacts'

print("🧹 Fresh MLflow Setup")
print("=" * 60)
print(f"Project root: {project_root}")
print(f"MLruns dir: {mlruns_dir}")

# Ensure directories exist
mlruns_dir.mkdir(parents=True, exist_ok=True)
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Set tracking URI with absolute path
tracking_uri = f"file:///{mlruns_dir.as_posix()}"
mlflow.set_tracking_uri(tracking_uri)

print(f"\n✅ Tracking URI set to: {tracking_uri}")

# Create fresh experiment
EXPERIMENT_NAME = "baseline-clustering-v2"

# Delete if exists
try:
    existing = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if existing:
        mlflow.delete_experiment(existing.experiment_id)
        print(f"🗑️  Deleted old experiment: {EXPERIMENT_NAME}")
except:
    pass

# Create new experiment WITHOUT custom artifact_location
experiment_id = mlflow.create_experiment(
    name=EXPERIMENT_NAME,
    tags={"project": "diabetes_clustering", "phase": "baseline", "version": "v2"}
)

print(f"\n✅ Created new experiment: {EXPERIMENT_NAME}")
print(f"   Experiment ID: {experiment_id}")

# Verify experiment
exp = mlflow.get_experiment(experiment_id)
print(f"   Artifact Location: {exp.artifact_location}")
print(f"\n🔍 Verifying artifact location is correct...")

# Check if path is nested (the problem!)
if "experiments/mlruns/experiments" in exp.artifact_location:
    print("   ❌ ERROR: Nested path detected!")
    print("   ⚠️  Please restart Python kernel and try again")
else:
    print("   ✅ Path is correct!")

print(f"\n💡 To start MLflow UI:")
print(f"   mlflow ui --backend-store-uri file://{mlruns_dir.as_posix()} --port 5002")

# Test logging a run
print(f"\n🧪 Testing with a sample run...")
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="test_setup") as run:
    mlflow.log_param("test", "setup")
    mlflow.log_metric("success", 1.0)
    print(f"   ✅ Test run logged: {run.info.run_id}")

# Verify physical directory
run_dir = mlruns_dir / experiment_id / run.info.run_id
if run_dir.exists():
    print(f"   ✅ Run directory exists: {run_dir}")
else:
    print(f"   ❌ Run directory NOT found: {run_dir}")

print("\n" + "=" * 60)
print("✅ Setup complete! You can now run your notebook.")
print("=" * 60)
