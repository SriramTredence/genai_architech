import streamlit as st
from agentic_rag_azure import graph, GraphState

st.set_page_config(page_title="Agentic RAG System", layout="wide")

st.title("Agentic RAG System with Azure + LangGraph")
st.markdown("Ask technical questions and get citation-backed answers.")

# Input question
question = st.text_input("Enter your question:", placeholder="e.g., What are best practices for caching?")

if st.button("Run Agentic RAG") and question:
    with st.spinner("Running pipeline..."):
        initial_state = GraphState({"question": question})
        final_state = graph.invoke(initial_state)

        # Retrieved snippets
        st.subheader("Retrieved Snippets")
        for i, snippet in enumerate(final_state["retrieved"]):
            st.markdown(f"**{snippet['doc_id']}** — _{snippet['source']}_")
            st.write(snippet['text'])
            st.markdown("---")

        # Initial Answer
        st.subheader("Initial Answer")
        st.markdown(final_state.get("initial_answer", "N/A"))

        # Critique
        st.subheader("Self-Critique Result")
        critique = final_state.get("critique_result", "UNKNOWN")
        if critique == "COMPLETE":
            st.success("Answer is COMPLETE – No refinement needed.")
        else:
            st.warning("Answer was INCOMPLETE – Refinement applied.")

        # Final Answer
        st.subheader("Final Answer")
        st.markdown(final_state.get("final_answer", final_state.get("initial_answer", "N/A")))

        st.info("MLflow logs have been saved for this run.")
        st.markdown("### MLflow Link")
        st.markdown(f"[Open Run →](http://localhost:5000/#/runs/{final_state.get("run_id")})")