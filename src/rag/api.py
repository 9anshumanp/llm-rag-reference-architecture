from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI

from rag.config import get_settings
from rag.logging import configure_logging
from rag.tracing import configure_tracing, TracingConfig
from rag.cache import DiskCache
from rag.vector_store import FaissVectorStore
from rag.retriever import Retriever
from rag.embeddings import EmbeddingService
from rag.generator import Generator
from rag.pipeline import RAGApp
from rag.async_services import AsyncEmbeddingService, AsyncGenerator
from rag.prompt import build_messages
from rag.metrics import Usage, CostModel, TokenEstimator

app = FastAPI(title="RAG Reference API", version="0.2.0")

class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    answer: str
    context_chunks: int
    prompt_tokens_est: int
    completion_tokens_est: int
    total_tokens_est: int
    cost_est_usd: float

class IndexResponse(BaseModel):
    docs: int
    chunks: int
    index_size: int

# Lazy singleton (simple, reference impl)
_state = {}

def _init_once():
    if _state:
        return
    s = get_settings()
    configure_logging()
    configure_tracing(TracingConfig(service_name=s.service_name, otlp_endpoint=s.otlp_endpoint))

    cache = None
    if s.enable_cache:
        s.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = DiskCache(str(s.cache_dir))

    # Persisted store
    store = FaissVectorStore(dimension=1536, dir_path=s.vector_dir)
    retriever = Retriever(store=store, top_k=s.top_k)

    # Sync app for indexing (simpler and reliable)
    client = OpenAI()
    embedder = EmbeddingService(client, s.embedding_model, cache, s.embedding_cache_ttl_s, s.max_retries, s.request_timeout_s)
    generator = Generator(client, s.llm_model, cache, s.completion_cache_ttl_s, s.max_retries, s.request_timeout_s)
    rag = RAGApp(settings=s, embedder=embedder, retriever=retriever, generator=generator, store=store)

    # Async for /ask for concurrency + rate limiting
    limiter = AsyncLimiter(max_rate=3, time_period=1)  # default 3 rps, tune as needed
    aclient = AsyncOpenAI()
    aembedder = AsyncEmbeddingService(aclient, s.embedding_model, cache, s.embedding_cache_ttl_s, s.max_retries, s.request_timeout_s, limiter=limiter)
    agenerator = AsyncGenerator(aclient, s.llm_model, cache, s.completion_cache_ttl_s, s.max_retries, s.request_timeout_s, limiter=limiter)

    _state.update({"s": s, "cache": cache, "store": store, "retriever": retriever, "rag": rag, "aembedder": aembedder, "agenerator": agenerator})

@app.get("/health")
def health():
    _init_once()
    return {"status": "ok"}

@app.post("/index", response_model=IndexResponse)
def index():
    _init_once()
    rag: RAGApp = _state["rag"]
    stats = rag.index()
    return IndexResponse(docs=stats.n_docs, chunks=stats.n_chunks, index_size=stats.index_size)

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    _init_once()
    retriever: Retriever = _state["retriever"]
    aembedder: AsyncEmbeddingService = _state["aembedder"]
    agenerator: AsyncGenerator = _state["agenerator"]
    s = _state["s"]

    q_emb = (await aembedder.embed([req.query]))[0]
    retrieved = retriever.retrieve(q_emb)
    chunks = [{"text": r.text, **r.meta, "score": r.score} for r in retrieved]
    messages = build_messages(req.query, chunks)

    answer, usage, cost = await agenerator.complete(messages)

    return AskResponse(
        answer=answer,
        context_chunks=len(chunks),
        prompt_tokens_est=usage.prompt_tokens,
        completion_tokens_est=usage.completion_tokens,
        total_tokens_est=usage.total_tokens,
        cost_est_usd=cost,
    )
