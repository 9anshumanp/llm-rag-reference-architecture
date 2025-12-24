from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple
import re
from pypdf import PdfReader

@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    source_path: str
    chunk_id: int
    text: str

def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for path in sorted(data_dir.glob("**/*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            text = _read_text_file(path)
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        text = normalize_text(text)
        if text.strip():
            docs.append((str(path), text))
    return docs

def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    # Sentence-ish splitting while keeping things deterministic
    parts = re.split(r"(?<=[.!?])\s+|\n\n+", text)
    parts = [p.strip() for p in parts if p and p.strip()]

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append(" ".join(cur).strip())
            cur = []
            cur_len = 0

    for part in parts:
        plen = len(part)
        if cur_len + plen + 1 <= chunk_size:
            cur.append(part)
            cur_len += plen + 1
            continue

        flush()

        # If single part is huge, hard-slice it
        if plen > chunk_size:
            start = 0
            while start < plen:
                end = min(start + chunk_size, plen)
                chunks.append(part[start:end].strip())
                start = max(end - chunk_overlap, start + 1)
        else:
            cur.append(part)
            cur_len = plen + 1

    flush()

    # Apply overlap by merging trailing text when possible (simple and robust)
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        prev_tail = ""
        for c in chunks:
            if prev_tail:
                overlapped.append((prev_tail + " " + c).strip())
            else:
                overlapped.append(c)
            prev_tail = c[-chunk_overlap:]
        chunks = overlapped

    return [c for c in chunks if c.strip()]

def build_chunks(docs: List[Tuple[str, str]], chunk_size: int, chunk_overlap: int) -> List[DocumentChunk]:
    out: List[DocumentChunk] = []
    for (path, text) in docs:
        doc_id = Path(path).name
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, c in enumerate(chunks):
            out.append(DocumentChunk(doc_id=doc_id, source_path=path, chunk_id=i, text=c))
    return out
