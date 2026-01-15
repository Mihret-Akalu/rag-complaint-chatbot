##TASK 3 — RAG PIPELINE


import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

# Initialize Chroma client
client = chromadb.PersistentClient(path="vector_store")

# Safely get collection
try:
    collection = client.get_collection("complaints")
    print(f"✓ Loaded 'complaints' collection with {collection.count()} chunks")
except:
    print("✗ Collection not found. Please build vector store first.")
    collection = None

# Initialize models
model = SentenceTransformer('all-MiniLM-L6-v2')
llm = ChatOpenAI(
    temperature=0.0,
    model="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here")  # Set your key
)

def retrieve_chunks(query: str, k: int = 5):
    """Retrieve top-k chunks from vector store."""
    if not collection:
        return ["Vector store not initialized. Run build_vector_store.py first."]
    
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    return results["documents"][0] if results["documents"] else []

def answer_question(question: str):
    """Generate answer using retrieved context."""
    context_chunks = retrieve_chunks(question)
    
    if not context_chunks or "not initialized" in context_chunks[0]:
        return "Please build the vector store first by running build_vector_store.py", []
    
    # Format context
    context_text = "\n".join([f"• {chunk}" for chunk in context_chunks])
    
    # Create prompt
    prompt = f"""You are a financial analyst at CrediTrust Financial. 
Answer the question using ONLY the following customer complaint excerpts.
If the context doesn't contain relevant information, say so.

Customer Complaint Excerpts:
{context_text}

Question: {question}

Based on the complaints above, answer:"""
    
    # Generate response
    try:
        response = llm.invoke(prompt)
        return response.content, context_chunks
    except Exception as e:
        return f"Error generating answer: {str(e)}", context_chunks