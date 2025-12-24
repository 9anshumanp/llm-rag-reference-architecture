from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import logging
from rag.tracing import get_tracer

log = logging.getLogger("rag.retriever")
tracer = get_tracer("rag.retriever")

@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float
    meta: Dict[str, Any]

class Retriever:
    def __init__(self, store, top_k: int):
        self._store = store
        self._top_k = top_k

    def retrieve(self, query_embedding: List[float]) -> List[RetrievedChunk]:
        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("top_k", self._top_k)
            hits = self._store.search(query_embedding, k=self._top_k)
            out = [RetrievedChunk(text=r.text, score=score, meta=r.meta) for (r, score) in hits]
            span.set_attribute("n_hits", len(out))
            return out
