# Real-Time Market Sentiment Analyzer

This project analyzes real-time sentiment on public companies using LangChain, GPT-4o, and mlflow.

## Setup

```bash
pip install -r requirements.txt


## Start MLFlow server
export $(cat .env | xargs)
mlflow server   --host 0.0.0.0   --port ${MLFLOW_TRACKING_PORT}


## Launch Streamlit UI

```bash
streamlit run streamlit_app/app.py
