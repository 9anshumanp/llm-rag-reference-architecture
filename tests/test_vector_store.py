from pathlib import Path
import shutil
import numpy as np
from rag.vector_store import FaissVectorStore, VectorRecord

def test_faiss_persist_roundtrip(tmp_path: Path):
    store_dir = tmp_path / "store"
    store = FaissVectorStore(dimension=3, dir_path=store_dir)

    embs = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    recs = [
        VectorRecord(id="a", text="alpha", meta={"doc_id":"d", "chunk_id":0}),
        VectorRecord(id="b", text="beta", meta={"doc_id":"d", "chunk_id":1}),
    ]
    store.add(embs, recs)
    store.persist()

    store2 = FaissVectorStore(dimension=3, dir_path=store_dir)
    hits = store2.search([0.0, 1.0, 0.0], k=1)
    assert len(hits) == 1
    assert hits[0][0].text == "alpha"
