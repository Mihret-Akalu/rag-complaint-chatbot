from src.rag_pipeline import retrieve_chunks

def test_retrieve_type():
    result = retrieve_chunks("sample query", k=3)
    assert isinstance(result, list)
