# LLM RAG Reference Architecture (Production-lean)

A production-oriented reference implementation for Retrieval-Augmented Generation (RAG) systems using Large Language Models (LLMs).

This repository is intentionally **engineering-first**:
- Modular components (ingestion → embeddings → retrieval → generation)
- **Caching** for embeddings and completions (disk-backed)
- **Retries** with exponential backoff for network calls
- **Tracing** via OpenTelemetry (console exporter by default)
- Structured logging and configuration via environment variables
- Runnable CLI for indexing and querying

## Quickstart

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure
Set an OpenAI API key:
```bash
export OPENAI_API_KEY="..."
```

Optional tuning:
```bash
export RAG_DATA_DIR="data/documents"
export RAG_VECTOR_DIR=".rag_store"
export RAG_TOP_K="5"
```

### 3) Add documents
Put `.txt`, `.md`, or `.pdf` files in `data/documents/`.

### 4) Index
```bash
python -m rag.cli index
```

### 5) Query
```bash
python -m rag.cli ask "What is RAG and why is it useful?"
```

## Notes

- Vector store uses FAISS and persists under `.rag_store/`.
- Embeddings and chat completions are cached under `.rag_cache/` (diskcache).
- Traces are printed to console by default; you can wire OTLP exporters if desired.

## Disclaimer
Reference architecture for educational and architectural guidance.


## API Service (FastAPI)

```bash
uvicorn rag.api:app --reload
```

Endpoints:
- `POST /index`
- `POST /ask` `{ "query": "..." }`

## Docker

```bash
docker build -t rag-ref .
docker run -p 8000:8000 -e OPENAI_API_KEY=... rag-ref
```

## Evaluation

```bash
python -c "from pathlib import Path; from rag.cli import build_app; from rag.eval import load_cases, run_eval; app=build_app(); cases=load_cases(Path('eval/golden.json')); print(run_eval(app, cases))"
```
