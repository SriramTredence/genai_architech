import streamlit as st
import os,sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_fol = os.path.abspath(os.path.dirname(parent_dir))
sys.path.append(parent_fol)
from src.recommend import recommend_courses, df

st.set_page_config(page_title="Course Recommender", layout="centered")
st.title("Personalized Course Recommender")

# Input: Completed course IDs
course_options = df["course_id"] + " - " + df["title"]
course_id_map = dict(zip(course_options, df["course_id"]))

selected_courses = st.multiselect("Select completed courses:", options=course_options)

# Input: User profile blurb
profile = st.text_area("Tell us about your interests:", height=150)

# Trigger
if st.button("Get Recommendations"):
    if (not selected_courses) or (not profile.strip()):
        st.error("Please select at least one completed course and provide a profile description")
        st.stop()
    completed_ids = [course_id_map[c] for c in selected_courses]
    recommendations = recommend_courses(profile, completed_ids)

    st.subheader("Top 5 Recommended Courses")
    for course_id, score in recommendations[:5]:
        course = df[df["course_id"] == course_id].iloc[0]
        st.markdown(f"### {course['title']}")
        st.markdown(f"**Course ID**: {course_id}")
        st.markdown(f"**Similarity Score**: `{score:.4f}`")
        st.markdown(f"**Description**: {course['description']}")
        st.markdown("---")
