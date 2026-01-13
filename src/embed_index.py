##TASK 2 — TEXT CHUNKING, EMBEDDING, & INDEXING
#Embeddings 

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from src.chunking import chunk_text

# Chroma client setup
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="vector_store"
))

collection = client.get_or_create_collection(name="complaints")

def build_chroma_index(csv_path: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        chunks = chunk_text(row["Consumer complaint narrative"])
        for c in chunks:
            emb = model.encode(c).tolist()
            collection.add(
                documents=[c],
                embeddings=[emb],
                metadatas=[{"product": row["Product"], "id": row["Complaint ID"]}]
            )
    collection.persist()
    print("Index built and persisted!")

if __name__ == "__main__":
    build_chroma_index("data/processed/filtered_complaints.csv")
