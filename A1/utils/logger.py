import mlflow
import os
import time 
import subprocess
import requests


def is_mlflow_running(uri):
    try:
        # MLflow server exposes /api/2.0/mlflow/experiments/list
        r = requests.get(f"{uri}/api/2.0/mlflow/experiments/list", timeout=3)
        if r.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        return False
    return False

def start_mlflow_server():
    # Example: starts server as a background process
    subprocess.Popen([
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", "5000",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns"
    ])
    time.sleep(5)

def init_mlflow():
    # if not is_mlflow_running(os.getenv("MLFLOW_TRACKING_URI")):
    #     start_mlflow_server()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Real-Time Sentiment Analyzer")

def start_run(company_name):
    return mlflow.start_run(run_name=f"Sentiment_Analysis_{company_name}")
