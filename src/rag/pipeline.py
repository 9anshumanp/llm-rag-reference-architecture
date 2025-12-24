from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

from rag.ingest import load_documents, build_chunks
from rag.vector_store import FaissVectorStore, VectorRecord
from rag.prompt import build_messages
from rag.tracing import get_tracer

log = logging.getLogger("rag.pipeline")
tracer = get_tracer("rag.pipeline")

@dataclass(frozen=True)
class IndexStats:
    n_docs: int
    n_chunks: int
    index_size: int

class RAGApp:
    def __init__(self, settings, embedder, retriever, generator, store: FaissVectorStore):
        self._s = settings
        self._embedder = embedder
        self._retriever = retriever
        self._generator = generator
        self._store = store

    def index(self) -> IndexStats:
        with tracer.start_as_current_span("index") as span:
            docs = load_documents(self._s.data_dir)
            chunks = build_chunks(docs, chunk_size=self._s.chunk_size, chunk_overlap=self._s.chunk_overlap)

            span.set_attribute("n_docs", len(docs))
            span.set_attribute("n_chunks", len(chunks))
            log.info("Loaded documents", extra={"component": "pipeline", "event": "docs_loaded"})

            # Embed in batches to keep request size reasonable
            texts = [c.text for c in chunks]
            batch = 64
            embeddings_all: List[List[float]] = []
            for i in range(0, len(texts), batch):
                embeddings_all.extend(self._embedder.embed(texts[i:i+batch]))

            records = [
                VectorRecord(
                    id=f"{c.doc_id}:{c.chunk_id}",
                    text=c.text,
                    meta={"doc_id": c.doc_id, "chunk_id": c.chunk_id, "source_path": c.source_path},
                )
                for c in chunks
            ]

            self._store.add(embeddings_all, records)
            self._store.persist()

            stats = IndexStats(n_docs=len(docs), n_chunks=len(chunks), index_size=self._store.size)
            log.info("Index complete", extra={"component": "pipeline", "event": "index_complete"})
            return stats

    def ask(self, query: str) -> str:
        with tracer.start_as_current_span("ask") as span:
            q_emb = self._embedder.embed([query])[0]
            retrieved = self._retriever.retrieve(q_emb)
            chunks = [
                {"text": r.text, **r.meta, "score": r.score}
                for r in retrieved
            ]
            messages = build_messages(query, chunks)
            span.set_attribute("n_context", len(chunks))
            return self._generator.complete(messages)
