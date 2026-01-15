import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
import sys
import time

# ===== CONFIGURATION =====
OLLAMA_MODEL = "llama3.2"  # Change to "mistral" or "llama2" if needed
VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "complaints"
TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_DELAY = 2

# ===== INITIALIZE COMPONENTS =====
def initialize_components():
    """Initialize all components with error handling"""
    components = {}
    
    # 1. Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        print(f"✓ Loaded '{COLLECTION_NAME}' collection with {collection.count()} chunks")
        components['collection'] = collection
    except Exception as e:
        print(f"✗ ChromaDB Error: {e}")
        print("  Run: python build_vector_store.py")
        components['collection'] = None
    
    # 2. Initialize Embedding Model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        components['embedding_model'] = embedding_model
        print("✓ Loaded embedding model")
    except Exception as e:
        print(f"✗ Embedding Model Error: {e}")
        components['embedding_model'] = None
    
    # 3. Initialize LLM (Ollama)
    try:
        llm = Ollama(
            model=OLLAMA_MODEL,
            temperature=TEMPERATURE,
            num_predict=512,  # Limit response length
            # timeout=30,  # Optional: set timeout
        )
        # Test the connection
        test_response = llm.invoke("Hello")
        print(f"✓ Connected to Ollama model: {OLLAMA_MODEL}")
        components['llm'] = llm
    except Exception as e:
        print(f"✗ Ollama Error: {e}")
        print("  Make sure:")
        print("  1. Ollama is running (check system tray)")
        print(f"  2. Model is downloaded: 'ollama pull {OLLAMA_MODEL}'")
        print("  3. You're using the correct model name")
        components['llm'] = None
    
    return components

# Initialize once when module loads
components = initialize_components()
collection = components['collection']
embedding_model = components['embedding_model']
llm = components['llm']

# ===== CORE FUNCTIONS =====
def retrieve_chunks(query: str, k: int = 5):
    """Retrieve top-k relevant chunks from vector store."""
    if not collection or not embedding_model:
        return ["System not properly initialized. Check errors above."]
    
    try:
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Query with metadata filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        if results['documents'] and results['documents'][0]:
            return results['documents'][0]
        else:
            return ["No relevant complaints found."]
            
    except Exception as e:
        print(f"Retrieval error: {e}")
        return [f"Error retrieving data: {str(e)}"]

def format_context(chunks):
    """Format retrieved chunks for the prompt."""
    if not chunks or "Error" in chunks[0] or "not initialized" in chunks[0]:
        return "No relevant complaint data available."
    
    formatted = []
    for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks max
        # Clean and truncate chunk
        clean_chunk = chunk.strip()
        if len(clean_chunk) > 300:
            clean_chunk = clean_chunk[:300] + "..."
        
        formatted.append(f"[Excerpt {i+1}] {clean_chunk}")
    
    return "\n\n".join(formatted)

def generate_answer(prompt: str, max_retries: int = MAX_RETRIES):
    """Generate answer with retry logic."""
    if not llm:
        return "LLM not available. Please check Ollama setup."
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return f"Failed to generate answer after {max_retries} attempts: {str(e)}"
    
    return "Unknown error in answer generation."

def answer_question(question: str, k: int = 5):
    """
    Main function: answer question using RAG pipeline.
    Returns (answer, sources, metadata)
    """
    print(f"\n🔍 Processing: '{question}'")
    
    # Step 1: Retrieve relevant chunks
    context_chunks = retrieve_chunks(question, k)
    
    # Step 2: Format context
    context_text = format_context(context_chunks)
    
    # Step 3: Create optimized prompt
    prompt = f"""You are a helpful financial complaint analyst at CrediTrust.
Your task is to analyze customer complaints and provide clear, concise answers.

CUSTOMER COMPLAINT EXCERPTS:
{context_text}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the complaint excerpts above.
2. Summarize the main issues mentioned by customers.
3. Be specific about products, issues, and customer sentiments.
4. If excerpts don't contain relevant information, say: "Based on the available complaints, I don't have specific information about this."
5. Keep your answer under 150 words.

ANALYSIS:"""
    
    # Step 4: Generate answer
    answer = generate_answer(prompt)
    
    # Step 5: Return results
    return answer, context_chunks

# ===== TEST FUNCTION =====
def test_pipeline():
    """Test the RAG pipeline with sample questions."""
    test_questions = [
        "What are common credit card complaints?",
        "Why are customers unhappy with billing?",
        "What savings account issues do people report?"
    ]
    
    print("\n" + "="*60)
    print("🧪 TESTING RAG PIPELINE")
    print("="*60)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer, sources = answer_question(question)
        print(f"A: {answer[:150]}...")
        print(f"  Sources: {len(sources)}")
        if sources and len(sources) > 0:
            print(f"  Sample: {sources[0][:100]}...")
        print("-" * 40)

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Run tests if script is executed directly
    test_pipeline()
    
    # Example single question
    # answer, sources = answer_question("What are common credit card complaints?")
    # print(f"\nFinal answer: {answer}")