Agentic RAG System (Azure + LangGraph)

This project implements an Agentic Retrieval-Augmented Generation (RAG) pipeline using:

- Azure OpenAI (GPT-4 mini & Embeddings)
- Pinecone (vector database)
- LangGraph (agentic flow control)
- Google Gemini (self-critique)
- MLflow (observability)

Built for Assignment 3: Agentic RAG System with Azure OpenAI & LangGraph

## Installation

1. **Clone the repo**
    git clone https://github.com/SriramTredence/genai_architech.git
    cd genai_architech/A3

2. Install the requirements
    pip install -r requirements.txt

3. Run mlflow server
    mlflow server (run in bash)

4. Run the streamlit app
    streamlit run streamlit_app.py