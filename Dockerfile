FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for faiss + building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY data /app/data
COPY README.md /app/README.md

ENV RAG_DATA_DIR=/app/data/documents
ENV RAG_VECTOR_DIR=/app/.rag_store
ENV RAG_CACHE_DIR=/app/.rag_cache
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
