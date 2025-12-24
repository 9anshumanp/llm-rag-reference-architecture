from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from aiolimiter import AsyncLimiter

from rag.cache import DiskCache, cached_call
from rag.hashing import sha256_json
from rag.tracing import get_tracer
from rag.metrics import Usage, TokenEstimator, CostModel

log = logging.getLogger("rag.async")
tracer = get_tracer("rag.async")

class AsyncEmbeddingService:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        cache: DiskCache | None,
        ttl_s: int | None,
        max_retries: int,
        timeout_s: float,
        limiter: Optional[AsyncLimiter] = None,
    ):
        self._client = client
        self._model = model
        self._cache = cache
        self._ttl_s = ttl_s
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._limiter = limiter

    async def embed(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self._model, "texts": texts}
        cache_key = f"aemb:{sha256_json(payload)}"

        async def _call():
            return await self._embed_uncached(texts)

        if self._cache:
            # diskcache is sync; keep it simple for reference impl
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit
            val = await _call()
            self._cache.set(cache_key, val, ttl_s=self._ttl_s)
            return val
        return await _call()

    def _retry_decorator(self):
        return retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries if self._max_retries > 0 else 1),
            wait=wait_exponential_jitter(initial=0.5, max=20),
            retry=retry_if_exception_type(Exception),
        )

    async def _embed_uncached(self, texts: List[str]) -> List[List[float]]:
        with tracer.start_as_current_span("async.embeddings.create") as span:
            span.set_attribute("model", self._model)
            span.set_attribute("n_texts", len(texts))
            if self._limiter:
                await self._limiter.acquire()

            @self._retry_decorator()
            async def _do():
                resp = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    timeout=self._timeout_s,
                )
                return [item.embedding for item in resp.data]

            vectors = await _do()
            span.set_attribute("dim", len(vectors[0]) if vectors else 0)
            return vectors

class AsyncGenerator:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        cache: DiskCache | None,
        ttl_s: int | None,
        max_retries: int,
        timeout_s: float,
        limiter: Optional[AsyncLimiter] = None,
    ):
        self._client = client
        self._model = model
        self._cache = cache
        self._ttl_s = ttl_s
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._limiter = limiter

        self._est = TokenEstimator(model)
        self._cost = CostModel(model)

    async def complete(self, messages: List[Dict[str, Any]]) -> tuple[str, Usage, float]:
        payload = {"model": self._model, "messages": messages}
        cache_key = f"achat:{sha256_json(payload)}"

        async def _call():
            return await self._complete_uncached(messages)

        if self._cache:
            hit = self._cache.get(cache_key)
            if hit is not None:
                text, usage_dict = hit
                usage = Usage(**usage_dict)
                cost = self._cost.estimate(usage).total_usd
                return text, usage, cost
            text, usage = await _call()
            self._cache.set(cache_key, (text, usage.__dict__), ttl_s=self._ttl_s)
            cost = self._cost.estimate(usage).total_usd
            return text, usage, cost

        text, usage = await _call()
        cost = self._cost.estimate(usage).total_usd
        return text, usage, cost

    def _retry_decorator(self):
        return retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries if self._max_retries > 0 else 1),
            wait=wait_exponential_jitter(initial=0.8, max=30),
            retry=retry_if_exception_type(Exception),
        )

    async def _complete_uncached(self, messages: List[Dict[str, Any]]) -> tuple[str, Usage]:
        with tracer.start_as_current_span("async.chat.completions.create") as span:
            span.set_attribute("model", self._model)
            if self._limiter:
                await self._limiter.acquire()

            # Best-effort token estimate for prompt
            prompt_text = "\n".join([m.get("content","") for m in messages])
            prompt_tokens = self._est.count(prompt_text)

            @self._retry_decorator()
            async def _do():
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.2,
                    timeout=self._timeout_s,
                )
                return resp

            resp = await _do()
            text = resp.choices[0].message.content or ""

            completion_tokens = self._est.count(text)
            usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

            span.set_attribute("prompt_tokens_est", usage.prompt_tokens)
            span.set_attribute("completion_tokens_est", usage.completion_tokens)
            span.set_attribute("total_tokens_est", usage.total_tokens)
            return text, usage
