from __future__ import annotations
import logging
from typing import Any, Dict, List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from rag.hashing import sha256_json
from rag.cache import DiskCache, cached_call
from rag.tracing import get_tracer

log = logging.getLogger("rag.generator")
tracer = get_tracer("rag.generator")

class Generator:
    def __init__(self, client: OpenAI, model: str, cache: DiskCache | None, ttl_s: int | None, max_retries: int, timeout_s: float):
        self._client = client
        self._model = model
        self._cache = cache
        self._ttl_s = ttl_s
        self._max_retries = max_retries
        self._timeout_s = timeout_s

    def complete(self, messages: List[Dict[str, Any]]) -> str:
        payload = {"model": self._model, "messages": messages}
        cache_key = f"chat:{sha256_json(payload)}"

        def _call():
            return self._complete_uncached(messages)

        if self._cache:
            return cached_call(self._cache, cache_key, _call, ttl_s=self._ttl_s)
        return _call()

    def _retry_decorator(self):
        return retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries if self._max_retries > 0 else 1),
            wait=wait_exponential_jitter(initial=0.8, max=30),
            retry=retry_if_exception_type(Exception),
        )

    def _complete_uncached(self, messages: List[Dict[str, Any]]) -> str:
        with tracer.start_as_current_span("chat.completions.create") as span:
            span.set_attribute("model", self._model)
            span.set_attribute("n_messages", len(messages))
            text = self._complete_with_retry(messages)
            span.set_attribute("n_chars", len(text))
            return text

    def _complete_with_retry(self, messages: List[Dict[str, Any]]) -> str:
        @self._retry_decorator()
        def _do():
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                timeout=self._timeout_s,
            )
            return resp.choices[0].message.content or ""
        return _do()
