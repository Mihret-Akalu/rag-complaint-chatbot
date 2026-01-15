## TASK 4 — BUILD CHAT INTERFACE
# app.py - Minimal working version
import gradio as gr
import sys
sys.path.append('.')
from src.rag_pipeline import answer_question

# This is the minimal working pattern
def chat_fn(message):
    answer, sources = answer_question(message)
    return answer

# Use Interface instead of ChatInterface
demo = gr.Interface(
    fn=chat_fn,
    inputs=gr.Textbox(label="Your Question"),
    outputs= gr.Textbox(label="Answer", scale=4, lines=8),
    title="CrediTrust Complaint Analyst",
    examples=[
        "What are common credit card complaints?",
        "Why are customers unhappy with billing?"
    ]
)

if __name__ == "__main__":
    demo.launch()