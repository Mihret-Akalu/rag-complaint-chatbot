##TASK 4 — BUILD CHAT INTERFACE
#Gradio App
# app.py
import gradio as gr
import sys
import os
from src.rag_pipeline import answer_question # Your existing function

def rag_chat_function(message, history):
    """
    The main chat function that integrates with your RAG pipeline.
    Uses 'yield' to stream the response token-by-token[citation:1].
    """
    # Call your existing RAG function
    full_answer, source_chunks = answer_question(message)
    
    # Simulate streaming for a better UX
    simulated_stream = ""
    for word in full_answer.split():
        simulated_stream += word + " "
        yield simulated_stream  # This makes the response appear word-by-word[citation:1]
    
    # After streaming the full answer, append the sources to the history
    # This makes the sources appear below the answer in the chat
    if source_chunks:
        source_text = "\n\n**Sources used:**\n" + "\n".join([f"- {chunk[:150]}..." for chunk in source_chunks[:3]])  # Show top 3
        yield full_answer + source_text

# Customize and launch the interface
demo = gr.ChatInterface(
    fn=rag_chat_function,
    title="CrediTrust Complaint Analyst",
    description="Ask questions about customer complaints across financial products.",
    chatbot=gr.Chatbot(height=450, label="Complaint Analysis"),
    textbox=gr.Textbox(placeholder="e.g., Why are people unhappy with credit cards?", scale=7),
    examples=[
        "What are common issues with personal loans?",
        "Summarize complaints about savings accounts.",
        "Compare issues between credit cards and money transfers."
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear Chat"  # This adds a clear button[citation:4]
)

if __name__ == "__main__":
    demo.launch(share=False)  # Set share=True for a temporary public link