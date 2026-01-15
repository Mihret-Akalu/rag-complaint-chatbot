import pandas as pd
from src.chunking import chunk_text

def test_chunk_length():
    text = "A"*1200
    chunks = chunk_text(text)
    assert all(len(c) <= 500 for c in chunks)
