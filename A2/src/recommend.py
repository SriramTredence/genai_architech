# recommend.py

import pandas as pd
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

# Load resources
df = pd.read_pickle("course_metadata.pkl")
print(df.head())
index = faiss.read_index("course_index.faiss")

with open("model_name.txt", "r") as f:
    model_name = f.read().strip()

model = SentenceTransformer(model_name)

# Map index position to course ID
index_to_course_id = dict(enumerate(df["course_id"]))

# Map course title -> course ID (to convert completed titles to IDs)
title_to_id = dict(zip(df["title"], df["course_id"]))


def recommend_courses(profile: str, completed_ids: List[str]) -> List[Tuple[str, float]]:
    """
    Given a user profile and list of completed course_ids,
    return a list of top-5 recommended (course_id, similarity_score).
    """
    # Get titles of completed courses
    completed_titles = df[df["course_id"].isin(completed_ids)]["title"].tolist()
    completed_text = " ".join(completed_titles)
    # Convert completed titles to IDs, ignoring any unknown titles
    print(f"title_to_id: {title_to_id}")
    print(f"completed titles: {completed_titles}")
    # completed_ids = [title_to_id[title] for title in completed_titles if title in title_to_id]
    print(f"Completed course ids {completed_ids}")
    # Build the query
    query_text = completed_text + " " + profile
    query_embedding = model.encode([query_text])

    # Perform similarity search
    D, I = index.search(query_embedding, k=20)

    # Prepare results
    recommendations = []
    for idx, score in zip(I[0], D[0]):
        course_id = index_to_course_id[idx]
        print(f"course {course_id}")
        if course_id not in completed_ids:
            similarity = 1 / (1 + score)  # Convert L2 to pseudo similarity
            recommendations.append((course_id, similarity))
        if len(recommendations) == 10:
            break

    return recommendations
