##TASK 2 — TEXT CHUNKING, EMBEDDING, & INDEXING
#Embeddings & vector store

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks):
    vectors = model.encode(chunks, show_progress_bar=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))
    return index, vectors

def save_index(index, metadata, vectors, path="vector_store/faiss_index.pkl"):
    with open(path, "wb") as f:
        pickle.dump((index, metadata, vectors), f)


#Sample script to run indexing:
import pandas as pd
from src.chunking import chunk_text
from src.embed_index import build_faiss_index, save_index

df = pd.read_csv("data/processed/filtered_complaints.csv")
all_chunks, metadata = [], []
for _, r in df.iterrows():
    for chunk in chunk_text(r["clean_narrative"]):
        all_chunks.append(chunk)
        metadata.append({
            "id": r["Complaint ID"],
            "product": r["Product"]
        })

index, vectors = build_faiss_index(all_chunks)
save_index(index, metadata, vectors)
