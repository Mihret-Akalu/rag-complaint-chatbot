
# Intelligent Complaint Analysis Chatbot

A RAG‑powered chatbot that turns CFPB customer complaint narratives into actionable insights. Users can ask plain‑English questions (e.g., “Why are people unhappy with Credit Cards?”) and get concise, evidence‑backed answers.

## 🚀 Features
- **Task 1:** EDA & text cleaning of complaint narratives  
- **Task 2:** Chunking & semantic embeddings (ChromaDB/FAISS)  
- **Task 3:** Retrieval + generation pipeline (LLM answers with context)  
- **Task 4:** Gradio UI for interactive querying

## 📁 Structure
```

data/              # Raw & processed datasets
notebooks/         # EDA & preprocessing notebooks
src/               # Core logic (preprocessing, chunking, embedding, RAG)
tests/             # Unit & integration tests
app.py             # Gradio UI
requirements.txt   # Dependencies

````

## 🚩 Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
````

Add your API key in a `.env` file (e.g., `OPENAI_API_KEY=your_key_here`).

## ▶️ Run

Build the index and run the app:

```bash
python -m src.embed_index
python app.py
```

## 🧪 Test

```bash
pytest
```

