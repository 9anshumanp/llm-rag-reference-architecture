from __future__ import annotations
from typing import List
import logging
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from rag.hashing import sha256_json
from rag.cache import DiskCache, cached_call
from rag.tracing import get_tracer

log = logging.getLogger("rag.embeddings")
tracer = get_tracer("rag.embeddings")

class EmbeddingService:
    def __init__(self, client: OpenAI, model: str, cache: DiskCache | None, ttl_s: int | None, max_retries: int, timeout_s: float):
        self._client = client
        self._model = model
        self._cache = cache
        self._ttl_s = ttl_s
        self._max_retries = max_retries
        self._timeout_s = timeout_s

    def embed(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self._model, "texts": texts}
        cache_key = f"emb:{sha256_json(payload)}"

        def _call():
            return self._embed_uncached(texts)

        if self._cache:
            return cached_call(self._cache, cache_key, _call, ttl_s=self._ttl_s)
        return _call()

    def _embed_uncached(self, texts: List[str]) -> List[List[float]]:
        with tracer.start_as_current_span("embeddings.create") as span:
            span.set_attribute("model", self._model)
            span.set_attribute("n_texts", len(texts))
            vectors = self._embed_with_retry(texts)
            span.set_attribute("dim", len(vectors[0]) if vectors else 0)
            return vectors

    def _retry_decorator(self):
        return retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries if self._max_retries > 0 else 1),
            wait=wait_exponential_jitter(initial=0.5, max=20),
            retry=retry_if_exception_type(Exception),
        )

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        @self._retry_decorator()
        def _do():
            resp = self._client.embeddings.create(
                model=self._model,
                input=texts,
                timeout=self._timeout_s,
            )
            return [item.embedding for item in resp.data]
        return _do()
