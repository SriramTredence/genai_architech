import pandas as pd
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss

# Load model globally
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load and preprocess course data
df = pd.read_csv("assignment2dataset.csv")

# Combine title + description into one text
df["full_text"] = df["title"] + ". " + df["description"]

# Generate course embeddings
course_embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=True)

# Store embeddings in FAISS index
dimension = course_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(course_embeddings)

# Mapping from index position to course ID
index_to_course_id = dict(enumerate(df["course_id"]))


def recommend_courses(profile: str, completed_ids: List[str]) -> List[Tuple[str, float]]:
    """
    Returns a list of (course_id, similarity_score) for the top-5 recommendations.
    """
    # Get course titles of completed IDs
    completed_titles = df[df["course_id"].isin(completed_ids)]["title"].tolist()
    completed_text = " ".join(completed_titles)

    # Combine with user profile
    query_text = completed_text + " " + profile
    query_embedding = model.encode([query_text])

    # Search in FAISS
    D, I = index.search(query_embedding, k=10)  # get more and filter later

    # Collect top results excluding completed_ids
    recommendations = []
    for idx, score in zip(I[0], D[0]):
        course_id = index_to_course_id[idx]
        if course_id not in completed_ids:
            similarity = 1 / (1 + score)  # convert L2 to similarity (optional)
            recommendations.append((course_id, similarity))
        if len(recommendations) == 5:
            break

    return recommendations
