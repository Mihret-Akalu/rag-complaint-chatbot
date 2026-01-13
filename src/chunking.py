##TASK 2 — TEXT CHUNKING, EMBEDDING, & INDEXING

#Chunking logic
def chunk_text(text: str, size: int = 500, overlap: int = 50):
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks
