from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Paths
    data_dir: Path = Field(default=Path("data/documents"))
    vector_dir: Path = Field(default=Path(".rag_store"))
    cache_dir: Path = Field(default=Path(".rag_cache"))

    # Models
    embedding_model: str = Field(default="text-embedding-3-small")
    llm_model: str = Field(default="gpt-4o-mini")

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=50)

    # Chunking
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)

    # Runtime
    request_timeout_s: float = Field(default=30.0, ge=5.0, le=120.0)
    max_retries: int = Field(default=6, ge=0, le=10)

    # Tracing (OTLP)
    otlp_endpoint: str | None = Field(default=None)
    service_name: str = Field(default="rag-reference")

    # Caching
    enable_cache: bool = Field(default=True)
    embedding_cache_ttl_s: int = Field(default=60 * 60 * 24 * 14)  # 14 days
    completion_cache_ttl_s: int = Field(default=60 * 60 * 24 * 7)  # 7 days

def get_settings() -> RAGSettings:
    return RAGSettings()
