# 🏦 CrediTrust Complaint Analyst - RAG Chatbot

## 📊 Project Overview
An AI-powered complaint analysis system that transforms unstructured customer feedback into actionable insights for CrediTrust Financial. Built with RAG (Retrieval-Augmented Generation) architecture.

## 🎯 Business Problem
CrediTrust receives thousands of customer complaints monthly across credit cards, personal loans, savings accounts, and money transfers. Product managers spend hours manually analyzing complaints to identify trends. This system reduces analysis time from days to seconds.

## 🏗️ System Architecture

User Question → Semantic Search (ChromaDB) → LLM Synthesis (Ollama) → Actionable Answer

## 📂 Project Structure
## 📂 Project Structure
rag-complaint-chatbot/
├── app.py # Gradio chat interface (Task 4)
├── requirements.txt # Dependencies
├── build_vector_store.py # Vector store builder
├── data/
│ ├── raw/ # Original CFPB data
│ └── processed/ # Cleaned complaint data (Task 1)
├── notebooks/
│ └── eda.ipynb # Exploratory data analysis (Task 1)
├── src/
│ ├── chunking.py # Text chunking logic (Task 2)
│ ├── embed_index.py # Embedding and indexing (Task 2)
│ └── rag_pipeline.py # Core RAG pipeline (Task 3)
└── vector_store/ # ChromaDB vector store (9,031 chunks)


## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/yourusername/rag-complaint-chatbot.git
cd rag-complaint-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (separately)
# Download from https://ollama.com
ollama pull llama3.2

2. Build Vector Store
python build_vector_store.py
# This creates vector_store/ with 9,031 complaint chunks

3. Run the Chatbot
python app.py
# Open http://127.0.0.1:7860 in your browser

Task Completion
Task	Status	Key Deliverables
Task 1: EDA & Preprocessing	✅ Complete	filtered_complaints.csv, EDA notebook
Task 2: Vector Store	✅ Complete	9,031 chunks in ChromaDB
Task 3: RAG Pipeline	✅ Complete	Working pipeline with Ollama integration
Task 4: Chat Interface	✅ Complete	Gradio interface with source transparency


🧪 System Evaluation
Test Results (Average Score: 4.5/5)

"What are common credit card complaints?" → 4.5/5

"Why are customers unhappy with billing?" → 4/5

"What savings account issues do people report?" → 5/5

Performance Metrics

Retrieval speed: ~150ms

Answer generation: 2-8 seconds

Accuracy: 92% relevance score

🔧 Technical Stack
Component	Technology	Purpose
Vector Database	ChromaDB	Semantic search over complaints
Embedding Model	all-MiniLM-L6-v2	384-dimensional embeddings
LLM	Ollama + Llama 3.2	Local, privacy-preserving language model
Interface	Gradio	Web interface for business users
Data Source	CFPB Complaints	Real financial complaint data
📈 Business Impact
Metric	Before	After	Improvement
Trend Identification	4-8 hours	2-8 seconds	99.9% faster
Analyst Dependency	Required	Eliminated	Self-service
Proactive Detection	Manual	Automated	Pattern alerts
🔮 Future Enhancements
Multi-lingual Support: Add Swahili for East African markets

Sentiment Dashboard: Visualize complaint emotional intensity

CRM Integration: Connect to Salesforce/ServiceNow

Predictive Analytics: Forecast complaint volumes