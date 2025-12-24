from __future__ import annotations
import typer
from rich import print
from openai import OpenAI

from rag.config import get_settings
from rag.logging import configure_logging
from rag.tracing import configure_tracing, TracingConfig
from rag.cache import DiskCache
from rag.embeddings import EmbeddingService
from rag.generator import Generator
from rag.vector_store import FaissVectorStore
from rag.retriever import Retriever
from rag.pipeline import RAGApp

app = typer.Typer(add_completion=False)

def build_app() -> RAGApp:
    s = get_settings()
    configure_logging()
    configure_tracing(TracingConfig(service_name=s.service_name, otlp_endpoint=s.otlp_endpoint))

    # Caches
    cache = None
    if s.enable_cache:
        s.cache_dir.mkdir(parents=True, exist_ok=True)
        cache = DiskCache(str(s.cache_dir))

    client = OpenAI()

    embedder = EmbeddingService(
        client=client,
        model=s.embedding_model,
        cache=cache,
        ttl_s=s.embedding_cache_ttl_s,
        max_retries=s.max_retries,
        timeout_s=s.request_timeout_s,
    )

    generator = Generator(
        client=client,
        model=s.llm_model,
        cache=cache,
        ttl_s=s.completion_cache_ttl_s,
        max_retries=s.max_retries,
        timeout_s=s.request_timeout_s,
    )

    # NOTE: embedding dimensions vary by model; OpenAI text-embedding-3-small is 1536.
    store = FaissVectorStore(dimension=1536, dir_path=s.vector_dir)
    retriever = Retriever(store=store, top_k=s.top_k)

    return RAGApp(settings=s, embedder=embedder, retriever=retriever, generator=generator, store=store)

@app.command()
def index() -> None:
    """Index documents under RAG_DATA_DIR into a FAISS store."""
    rag = build_app()
    stats = rag.index()
    print(f"[bold green]Indexed[/bold green] docs={stats.n_docs} chunks={stats.n_chunks} index_size={stats.index_size}")

@app.command()
def ask(query: str) -> None:
    """Ask a question using RAG over the indexed corpus."""
    rag = build_app()
    answer = rag.ask(query)
    print(answer)

if __name__ == "__main__":
    app()
