import mlflow
import os

def init_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Real-Time Sentiment Analyzer")

def start_run(company_name):
    return mlflow.start_run(run_name=f"Sentiment_Analysis_{company_name}")
