# load_data.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load dataset
DATA_URL = "https://raw.githubusercontent.com/Bluedata-Consulting/GAAPB01-training-code-base/refs/heads/main/Assignments/assignment2dataset.csv"
df = pd.read_csv(DATA_URL)

# Combine title and description
df["full_text"] = df["title"] + ". " + df["description"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings
embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "course_index.faiss")

# Save metadata
df.to_pickle("course_metadata.pkl")

# Save embedding model name (optional, if switching between models)
with open("model_name.txt", "w") as f:
    f.write("all-MiniLM-L6-v2")

print("✅ Vector DB and metadata saved.")
