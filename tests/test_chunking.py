from rag.ingest import chunk_text

def test_chunk_text_nonempty():
    chunks = chunk_text("Hello world. " * 200, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c.strip() for c in chunks)

def test_chunk_overlap_constraint():
    try:
        chunk_text("abc", chunk_size=10, chunk_overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
