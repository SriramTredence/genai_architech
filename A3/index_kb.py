import json
import os
# from azure.ai.inference import TextAnalyticsClient  # for embeddings, or azure‑ai‑inference
from azure.core.credentials import AzureKeyCredential
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()
# === Configuration (set via env or constants) ===
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_KEY = os.getenv("AZURE_EMBEDDING_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "kb-index")

KB_JSON = "A3/self_critique_loop_dataset.json"  # or path to KB file

# === Embedding helper ===
# from azure.ai.inference import TextAnalyticsClient  # or azure.ai.openai depending on SDK version

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Call Azure embedding endpoint for a batch of texts.
    Return list of embedding vectors.
    """
    # This depends on your Azure SDK version: if you use azure-ai-inference or azure‑ai‑openai etc.

    client = EmbeddingsClient(AZURE_EMBEDDING_ENDPOINT, credential=AzureKeyCredential(AZURE_EMBEDDING_KEY))
    resp = client.embed(input=texts)
    # resp.data is list of embedding objects
    return [e.embedding for e in resp.data]

def load_kb():
    with open(KB_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)
    return items

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print(f"indices: {pc.list_indexes()}")
    if PINECONE_INDEX_NAME not in pc.list_indexes():
        pc.create_index(name=PINECONE_INDEX_NAME,spec=ServerlessSpec(cloud='aws', region='us-east-1'),dimension=1536)
    idx = pc.Index(PINECONE_INDEX_NAME)
    return idx

def index_kb():
    kb = load_kb()
    # batch embed
    texts = [entry["question"] for entry in kb]
    embeddings = embed_texts(texts)
    idx = init_pinecone()

    # upsert in batches
    to_upsert = []
    for entry, emb in zip(kb, embeddings):
        # metadata can include id, maybe original text
        meta = entry
        to_upsert.append((entry["doc_id"], emb, meta))
    # upsert
    idx.upsert(vectors=to_upsert)
    print("Indexed {} KB entries".format(len(kb)))

if __name__ == "__main__":
    index_kb()
