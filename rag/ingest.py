from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
import fitz  # PyMuPDF

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

@dataclass
class Chunk:
    text: str
    doc_id: str
    page_start: int
    page_end: int
    start_tok: int
    end_tok: int

def _encode_tokens(text: str) -> list[int]:
    if _ENC is None:
        # Fallback: simple char-based slicing in ~4*token heuristic
        return list(text)
    return _ENC.encode(text)

def _decode_tokens(toks: list[int]) -> str:
    if _ENC is None:
        return "".join(toks)  # type: ignore
    return _ENC.decode(toks)

def chunk_text(text: str, doc_id: str, max_tokens: int = 600, overlap: int = 100) -> list[Chunk]:
    toks = _encode_tokens(text)
    out: list[Chunk] = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + max_tokens)
        piece = _decode_tokens(toks[i:j])
        out.append(Chunk(
            text=piece, doc_id=doc_id,
            page_start=0, page_end=0,
            start_tok=i, end_tok=j
        ))
        if j >= len(toks): break
        i = max(0, j - overlap)
    return out

def extract_pdf_chunks(pdf_path: str, max_tokens=600, overlap=100) -> list[Chunk]:
    doc = fitz.open(pdf_path)
    chunks: list[Chunk] = []
    for pno in range(len(doc)):
        page = doc[pno]
        txt = page.get_text("text")
        doc_id = pdf_path
        # page-wise token chunking (bessere Lokalität der Zitate)
        toks = _encode_tokens(txt)
        i = 0
        while i < len(toks):
            j = min(len(toks), i + max_tokens)
            piece = _decode_tokens(toks[i:j])
            chunks.append(Chunk(
                text=piece, doc_id=doc_id,
                page_start=pno+1, page_end=pno+1,
                start_tok=i, end_tok=j
            ))
            if j >= len(toks): break
            i = max(0, j - overlap)
    return chunks

def to_dicts(chunks: Iterable[Chunk]) -> list[dict]:
    return [asdict(c) for c in chunks]
