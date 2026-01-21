"""
MLflow All-in-One (macOS friendly)
Author: Yikai

Summary:
1) Sets MLflow tracking to a folder on your Desktop so your runs always show up in the same place.
2) (Optional) Cleans previous runs by deleting the `mlruns` folder.
3) Starts the MLflow UI server (http://127.0.0.1:5000) as a background process.
4) Trains a simple scikit-learn model (RandomForest on Iris dataset).
5) Logs parameters, metrics, and the trained model into MLflow.
6) Prints what to do next (open the browser, where files are stored, etc.)

IMPORTANT NOTES:
- You still need MLflow + scikit-learn installed in your Python environment:
    pip install mlflow scikit-learn
- If port 5000 is already in use, change MLFLOW_PORT below.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# -----------------------------
# USER SETTINGS (edit if needed)
# -----------------------------

# Where your MLflow runs will be stored (on your Desktop).
# This ensures the UI and your training always point to the same storage.
DESKTOP_DIR = Path.home() / "Desktop"
MLRUNS_DIR = DESKTOP_DIR / "mlruns"

# MLflow UI address
MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = 5000

# If True, deletes previous runs on Desktop/mlruns each time you run this script.
# Set to False if you want to keep your old runs.
CLEAN_PREVIOUS_RUNS = True

# Experiment name shown in MLflow UI
EXPERIMENT_NAME = "iris-demo"

# -----------------------------
# Helper functions
# -----------------------------

def explain(msg: str) -> None:
    """Print friendly explanations."""
    print(f"\n[INFO] {msg}")

def ensure_dependencies():
    """Check that MLflow and scikit-learn are importable."""
    explain("Checking that required Python packages are installed (mlflow, scikit-learn)...")
    try:
        import mlflow  # noqa: F401
    except Exception as e:
        print("\n[ERROR] Could not import 'mlflow'.")
        print("Install it with: pip install mlflow")
        raise e

    try:
        import sklearn  # noqa: F401
    except Exception as e:
        print("\n[ERROR] Could not import 'scikit-learn'.")
        print("Install it with: pip install scikit-learn")
        raise e

def clean_mlruns_folder():
    """Optionally remove old MLflow data to start fresh."""
    if CLEAN_PREVIOUS_RUNS and MLRUNS_DIR.exists():
        explain(f"Cleaning previous MLflow runs by deleting: {MLRUNS_DIR}")
        shutil.rmtree(MLRUNS_DIR, ignore_errors=True)

def set_tracking_uri():
    """
    Force MLflow to store runs in a specific folder.
    Without this, MLflow defaults to ./mlruns in your current folder,
    which can cause confusion if you run scripts from different directories.
    """
    import mlflow

    explain("Setting MLflow tracking URI to a fixed Desktop folder so runs are always in one place.")
    # MLflow expects a file URI format like: file:///Users/you/Desktop/mlruns
    tracking_uri = MLRUNS_DIR.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    explain(f"MLflow tracking URI is now: {tracking_uri}")

def start_mlflow_ui():
    """
    Start `mlflow ui` in the background so you don't need a second terminal.
    We point the backend store to the same folder used by tracking.
    """
    explain("Starting MLflow UI in the background...")

    # Use sys.executable to ensure we use the same Python environment running this script.
    # We run: python -m mlflow ui --host 127.0.0.1 --port 5000 --backend-store-uri file:///.../mlruns
    backend_store_uri = MLRUNS_DIR.resolve().as_uri()

    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--host", MLFLOW_HOST,
        "--port", str(MLFLOW_PORT),
        "--backend-store-uri", backend_store_uri,
    ]

    # Start MLflow UI as a background process.
    # stdout/stderr are suppressed to keep output clean. If you want logs, remove DEVNULL lines.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # keeps it running independently
    )

    # Give the server a moment to start
    time.sleep(2)

    explain(f"MLflow UI should be available at: http://{MLFLOW_HOST}:{MLFLOW_PORT}")
    explain("If it doesn't open, the port may already be used. Change MLFLOW_PORT in this file.")

    return proc

def train_and_log():
    """
    Train a simple model and log everything to MLflow.
    This is the core MLflow Tracking workflow:
      - start a run
      - log params
      - log metrics
      - log model artifact
    """
    explain("Training a simple RandomForest model and logging it to MLflow...")

    import mlflow
    import mlflow.sklearn
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Create/use an experiment name (groups runs in the UI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    X, y = load_iris(return_X_y=True)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Parameters for the model (we will log these)
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

    with mlflow.start_run(run_name="rf-iris"):
        # Train
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Print helpful info
        run = mlflow.active_run()
        run_id = run.info.run_id if run else "(unknown)"

        explain("Training complete.")
        print(f"\nResult metric: accuracy = {acc:.4f}")
        print(f"MLflow Run ID: {run_id}")
        print(f"Experiment name: {EXPERIMENT_NAME}")
        print(f"Runs stored at: {MLRUNS_DIR}")
        print(f"Open the UI here: http://{MLFLOW_HOST}:{MLFLOW_PORT}")

def main():
    explain("Starting MLflow all-in-one script...")

    # 1) Ensure you have packages
    ensure_dependencies()

    # 2) (Optional) clean old runs
    clean_mlruns_folder()

    # 3) Ensure mlruns directory exists
    explain(f"Ensuring MLflow storage folder exists: {MLRUNS_DIR}")
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    # 4) Set tracking uri so training writes to Desktop/mlruns
    set_tracking_uri()

    # 5) Start MLflow UI in background
    ui_proc = start_mlflow_ui()

    # 6) Train and log a run
    train_and_log()

    # 7) Keep script polite: UI continues running, but we explain how to stop it
    explain("Done! The MLflow UI server is still running in the background.")
    print(
        "\nTo stop the MLflow UI later, you can either:\n"
        "1) Close/restart your computer, OR\n"
        "2) Find and kill the process:\n"
        "   - In Terminal:  lsof -i :5000\n"
        "   - Then:         kill -9 <PID>\n"
    )

if __name__ == "__main__":
    main()

# For using - open the Terminal, in Terminal:
# cd ~/Desktop
# python3 mlflow_all_in_one_Mac.py
