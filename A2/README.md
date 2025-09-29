# Personalized Course Recommendation Engine

This project is a semantic search-powered Course Recommendation Engine that suggests the top-5 most relevant online courses based on a user's completed courses and interests. It uses embedding models to represent course descriptions and user queries in vector space, and retrieves the closest matches using a vector database. A Streamlit UI is provided for interactive exploration.

---

## Features

- Embedding-based semantic search using pre-trained models
- Fast vector similarity matching with a vector database
- Personalized recommendations based on course history and interests
- Streamlit UI for intuitive user interaction
- Evaluation of recommendations on 5 sample user profiles

---

## Tech Stack

- Python
- Sentence Transformers / OpenAI Embeddings
- FAISS or Chroma (Vector DB)
- Streamlit (UI)

## Installation

1. **Clone the repo**
    git clone https://github.com/SriramTredence/genai_architech.git
    cd genai_architech/A2
    
2. Install the requirements
    pip install -r requirements.txt

3. Run streamlit 
    streamlit run streamlit_app/app.py
