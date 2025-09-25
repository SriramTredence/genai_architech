import streamlit as st
import json
import altair as alt
import pandas as pd
import os,sys
parent_dir = os.path.abspath(os.path.dirname(__file__))
parent_fol = os.path.abspath(os.path.dirname(parent_dir))
sys.path.append(parent_dir)
sys.path.append(parent_fol)
from main import run_chain
from components.sent_db import load_sentiment_history

st.title("Real-Time Market Sentiment Analyzer")

company = st.text_input("Enter Company Name", "Microsoft")

if st.button("Analyze"):
    with st.spinner("Running sentiment analysis..."):
        result = run_chain(company)
        print("result: ", result)
        st.subheader("Sentiment Summary")
        st.json(result.dict())

        st.subheader("Sentiment Trend")
        history = load_sentiment_history(company)
        if history:
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            chart = alt.Chart(df).mark_line(point=True).encode(
                x="timestamp:T",
                y="score:Q",
                color=alt.value("blue"),
                tooltip=["timestamp", "sentiment", "score"]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No historical data yet.")