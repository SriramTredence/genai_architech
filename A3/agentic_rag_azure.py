import os
import mlflow
from typing import List
from langgraph.graph import StateGraph, END
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage
from azure.core.credentials import AzureKeyCredential
from pinecone import Pinecone
import google.generativeai as genai
from index_kb import embed_texts
from dotenv import load_dotenv
load_dotenv()
# Load environment variables
AZURE_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_CHATCOMPLETION_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4-mini")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-index")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_KEY = os.getenv("AZURE_EMBEDDING_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
# Init clients
azure_client = ChatCompletionsClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_API_KEY))
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

# Define graph state
class GraphState(dict): 
    question: str
    retrieved: list
    initial_answer: str
    critique_result: str
    final_answer: str

#Retriever Node
def retrieve_top_k(state: GraphState, k=5):
    query = state["question"]
    
    print(f'Query: {query}')
    embedding = embed_texts([query])
    results = index.query(vector=embedding, top_k=k, include_metadata=True)

    snippets = [{
        "doc_id": match["id"],
        "text": match["metadata"]["answer_snippet"],
        "source": match["metadata"]["question"]
    } for match in results["matches"]]

    state["retrieved"] = snippets
    print(f"snippets: {snippets}")
    return state

#LLM Answer Node
def generate_answer(state: GraphState,status="init"):
    question = state["question"]
    snippets = state["retrieved"]

    context = "\n".join([f"[{s['doc_id']}]: {s['text']}" for s in snippets])
    prompt = f"""You are an expert assistant. Use the following knowledge base entries to answer the question.

Context:
{context}

Question:
{question}

Provide a concise, helpful answer and cite sources like [KB001]."""

    # chat_input = ChatCompletionsOptions(
    #     messages=[ChatMessage(role="user", content=prompt)],
    #     temperature=0
    # )
    # response = azure_client.chat( deployment_name=AZURE_DEPLOYMENT_NAME, options=chat_input)
    # answer = response.choices[0].message.content

    response = azure_client.complete(messages=[
        SystemMessage(content=prompt)],
        temperature=0)
    answer = response.choices[0].message.content

    if status=="init":
        state["initial_answer"] = answer
    else:
        state["final_answer"]=answer
    print(f"answer: {answer}")
    return state

#Gemini Self-Critique Node
def critique_answer(state: GraphState):
    prompt = f"""Review the answer below. If it fully and accurately answers the question, respond with 'COMPLETE'. If it is incomplete, respond only with 'REFINE'.

Question: {state['question']}

Answer:
{state['initial_answer']}
"""
    gemini_response = gemini.generate_content(prompt)
    critique = gemini_response.text.strip().upper()

    state["critique_result"] = "REFINE" if "REFINE" in critique else "COMPLETE"
    print(f"critique_result: {state["critique_result"]}")
    return state

#Refinement Node
def refine_answer(state: GraphState):
    if state["critique_result"] == "COMPLETE":
        state["final_answer"] = state["initial_answer"]
        return state

    # Retrieve 1 more snippet
    query = state["question"]
    existing_ids = [s["doc_id"] for s in state["retrieved"]]
    embedding = embed_texts([query])[0]

    results = index.query(vector=embedding, top_k=10, include_metadata=True)
    extra = next((m for m in results["matches"] if m["id"] not in existing_ids), None)
    if not extra:
        state["final_answer"] = state["initial_answer"]
        return state

    state["retrieved"].append({
        "doc_id": extra["id"],
        "text": extra["metadata"]["answer_snippet"],
        "source": extra["metadata"]["question"]
    })

    # Re-generate answer
    print(f"extra: {extra}")
    return generate_answer(state,status="final")

def init_mlflow():
    # if not is_mlflow_running(os.getenv("MLFLOW_TRACKING_URI")):
    #     start_mlflow_server()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("Assignment_3")

def start_run(question):
    return mlflow.start_run(run_name=f"RAG_Solution_for_{question}")

# Logger Node
def log_to_mlflow(state: GraphState):
    init_mlflow()
    mlflow.start_run()
    run = mlflow.active_run()
    print(f"run_id: {run.info.run_id}; status: {run.info.status}")
    mlflow.log_param("question", state["question"])
    mlflow.log_param("critique", state["critique_result"])
    mlflow.log_text(state["initial_answer"], "initial_answer.txt")
    mlflow.log_text(state.get("final_answer", state["initial_answer"]), "final_answer.txt")

    for i, doc in enumerate(state["retrieved"]):
        mlflow.log_dict(doc, f"retrieved_snippet_{i}.json")

    mlflow.end_run()
    # print("Logging Result")
    return state

#LangGraph Flow
builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_top_k)
builder.add_node("answer", generate_answer)
builder.add_node("critique", critique_answer)
builder.add_node("refine", refine_answer)
builder.add_node("log", log_to_mlflow)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", "critique")
builder.add_conditional_edges("critique", 
    path=lambda state: state["critique_result"],
    path_map={
        "COMPLETE": "log",
        "REFINE": "refine"
    }
)
builder.add_edge("refine", "log")
builder.add_edge("log", END)

#Run Pipeline
graph = builder.compile()

def ask_agentic_question(question: str):
    initial_state = GraphState({"question": question})
    final_state = graph.invoke(initial_state)
    return final_state#["final"]

# Example
if __name__ == "__main__":
    # response = ask_agentic_question("What are best practices for caching?")
    response = ask_agentic_question("What are performance tuning tips?")
    
    print("\n Final Answer: \n", response)