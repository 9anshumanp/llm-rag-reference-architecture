from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

log = logging.getLogger("rag.vector_store")

@dataclass(frozen=True)
class VectorRecord:
    id: str
    text: str
    meta: Dict[str, Any]

class FaissVectorStore:
    def __init__(self, dimension: int, dir_path: Path):
        self._dim = dimension
        self._dir = dir_path
        self._dir.mkdir(parents=True, exist_ok=True)

        self._index_path = self._dir / "index.faiss"
        self._meta_path = self._dir / "meta.jsonl"

        self._index = faiss.IndexFlatL2(self._dim)
        self._records: List[VectorRecord] = []

        if self._index_path.exists() and self._meta_path.exists():
            self._load()

    def _load(self) -> None:
        log.info("Loading FAISS index", extra={"component": "vector_store", "event": "load"})
        self._index = faiss.read_index(str(self._index_path))
        self._records = []
        with self._meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self._records.append(VectorRecord(id=obj["id"], text=obj["text"], meta=obj["meta"]))

    def persist(self) -> None:
        log.info("Persisting FAISS index", extra={"component": "vector_store", "event": "persist"})
        faiss.write_index(self._index, str(self._index_path))
        with self._meta_path.open("w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps({"id": r.id, "text": r.text, "meta": r.meta}, ensure_ascii=False) + "\n")

    @property
    def size(self) -> int:
        return len(self._records)

    def add(self, embeddings: List[List[float]], records: List[VectorRecord]) -> None:
        if not embeddings:
            return
        if len(embeddings) != len(records):
            raise ValueError("Embeddings and records length mismatch")

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"Expected embeddings shape (n, {self._dim})")

        self._index.add(vectors)
        self._records.extend(records)

    def search(self, query_embedding: List[float], k: int) -> List[Tuple[VectorRecord, float]]:
        if self.size == 0:
            return []

        q = np.asarray([query_embedding], dtype="float32")
        distances, indices = self._index.search(q, k)
        results: List[Tuple[VectorRecord, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= self.size:
                continue
            results.append((self._records[idx], float(dist)))
        return results
