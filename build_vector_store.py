# save as build_vector_store.py in project root
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import os

def chunk_text(text: str, size: int = 500, overlap: int = 50):
    """Simple text chunking function"""
    chunks = []
    start = 0
    text = str(text)
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

def build_vector_store():
    print("Step 1: Loading data...")
    data_path = "data/processed/filtered_complaints.csv"
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} complaints")
    print(f"   Products: {df['Product'].unique()}")
    
    print("\nStep 2: Initializing ChromaDB...")
    # Clear existing vector store if any
    if os.path.exists("vector_store"):
        import shutil
        shutil.rmtree("vector_store")
        print("   Cleared existing vector store")
    
    client = chromadb.PersistentClient(path="vector_store")
    collection = client.get_or_create_collection(name="complaints")
    
    print("Step 3: Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Step 4: Processing complaints...")
    batch_docs = []
    batch_embs = []
    batch_metas = []
    batch_ids = []
    
    for idx, row in df.iterrows():
        narrative = str(row.get("Consumer complaint narrative", ""))
        if not narrative or narrative.lower() == "nan":
            continue
            
        # Chunk the narrative
        chunks = chunk_text(narrative, size=500, overlap=50)
        
        for chunk_idx, chunk in enumerate(chunks):
            # Generate embedding
            embedding = model.encode(chunk).tolist()
            
            # Prepare metadata
            metadata = {
                "complaint_id": str(row.get("Complaint ID", idx)),
                "product": str(row.get("Product", "Unknown")),
                "issue": str(row.get("Issue", "")),
                "company": str(row.get("Company", "")),
                "state": str(row.get("State", "")),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks)
            }
            
            # Add to batch
            batch_docs.append(chunk)
            batch_embs.append(embedding)
            batch_metas.append(metadata)
            batch_ids.append(f"doc_{idx}_chunk_{chunk_idx}")
            
            # Insert in batches of 100
            if len(batch_docs) >= 100:
                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
                batch_docs, batch_embs, batch_metas, batch_ids = [], [], [], []
        
        if idx % 100 == 0:
            print(f"   Processed {idx+1}/{len(df)} complaints")
    
    # Add remaining chunks
    if batch_docs:
        collection.add(
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas,
            ids=batch_ids
        )
    
    print(f"\nStep 5: Done! Vector store built.")
    print(f"   Total chunks: {collection.count()}")
    print(f"   Saved to: vector_store/")

if __name__ == "__main__":
    build_vector_store()