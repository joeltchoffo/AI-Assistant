#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG CLI — PDFs -> Chunks -> Embeddings -> FAISS -> Ask

Befehle:
  1) Ingest (Index bauen):
     python rag_cli.py ingest --data_dir data/papers --index_dir indices/faiss --chunk_size 500 --overlap 100

  2) Frage stellen:
     python rag_cli.py ask --index_dir indices/faiss --question "What problem does the paper address?" --k 5

Funktioniert offline. Optional: Wenn OPENAI_API_KEY gesetzt ist, wird für die finale Antwort ein LLM genutzt.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from glob import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# --- Konsole auf UTF-8 setzen (verhindert UnicodeEncodeError auf Windows) ---
try:
    sys.stdout.reconfigure(encoding="utf-8")  # Python 3.7+
except Exception:
    pass

import numpy as np
from tqdm import tqdm

# PDF
import fitz  # PyMuPDF

# Embeddings
from sentence_transformers import SentenceTransformer

# FAISS
import faiss

# Optional LLM (nur genutzt, wenn OPENAI_API_KEY existiert)
USE_LLM = False
try:
    from openai import OpenAI
    USE_LLM = True
except Exception:
    USE_LLM = False


# -------------------------
# Konfiguration (Defaults)
# -------------------------
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
DEFAULT_MAX_CTX_CHARS = 2000


# ---------------
# Hilfsstrukturen
# ---------------
@dataclass
class Chunk:
    text: str
    source: str
    start_word: int
    end_word: int


# ---------------
# PDF -> Text
# ---------------
def pdf_to_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")
    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts).strip()


# ---------------
# Chunking
# ---------------
def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def pdf_to_chunks(pdf_path: Path, size: int, overlap: int) -> List[Chunk]:
    raw = pdf_to_text(pdf_path)
    words = raw.split()
    if not words:
        return []
    chunks: List[Chunk] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        text = " ".join(words[start:end])
        chunks.append(Chunk(text=text, source=pdf_path.name, start_word=start, end_word=end))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


# ---------------
# Embeddings
# ---------------
class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=normalize)
        return np.array(vecs, dtype="float32")


# ---------------
# FAISS Store
# ---------------
class FaissStore:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.meta_path = index_path.with_suffix(".meta.jsonl")
        self.index: Optional[faiss.IndexFlatIP] = None
        self.meta: List[Dict] = []

    def _ensure_index(self, dim: int):
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        assert embeddings.ndim == 2, "Embeddings müssen 2D sein (N, dim)."
        self._ensure_index(embeddings.shape[1])
        self.index.add(embeddings)
        self.meta.extend(metadatas)

    def save(self):
        assert self.index is not None, "Index nicht initialisiert."
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_path: Path) -> "FaissStore":
        meta_path = index_path.with_suffix(".meta.jsonl")
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Index oder Metadaten fehlen bei {index_path}")
        store = cls(index_path)
        store.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            store.meta = [json.loads(line) for line in f if line.strip()]
        return store

    def search(self, query_vec: np.ndarray, k: int = DEFAULT_TOP_K) -> Tuple[np.ndarray, np.ndarray]:
        assert self.index is not None, "Index nicht geladen/gebaut."
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        sims, idxs = self.index.search(query_vec, k)
        return sims[0], idxs[0]


# ---------------
# Ingest
# ---------------
def ingest_pdfs_to_faiss(
    data_dir: Path,
    index_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> Path:
    pdf_paths = [Path(p) for p in glob(str(data_dir / "*.pdf"))]
    if not pdf_paths:
        raise FileNotFoundError(f"Keine PDFs in {data_dir} gefunden.")

    print(f"[Ingest] PDFs gefunden: {len(pdf_paths)}")
    all_chunks: List[Chunk] = []
    for pdf in tqdm(pdf_paths, desc="PDFs lesen & chunken"):
        try:
            chs = pdf_to_chunks(pdf, size=chunk_size, overlap=overlap)
            all_chunks.extend(chs)
        except Exception as e:
            print(f"[warn] Fehler beim Lesen {pdf}: {e}")

    if not all_chunks:
        raise RuntimeError("Aus PDFs wurde kein Text extrahiert (leer?).")

    texts = [c.text for c in all_chunks]
    metas = [dict(text=c.text, source=c.source, start=c.start_word, end=c.end_word) for c in all_chunks]

    embedder = Embedder(embed_model)
    embs = embedder.encode(texts, normalize=True)

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "papers.index.faiss"
    store = FaissStore(index_path)
    store.add(embs, metas)
    store.save()

    print(f"[Ingest] Fertig. Index: {index_path} | Einträge: {store.index.ntotal}")
    return index_path


# ---------------
# Retrieval
# ---------------
class Retriever:
    def __init__(self, index_path: Path, embed_model: str = DEFAULT_EMBED_MODEL):
        self.store = FaissStore.load(index_path)
        self.embedder = Embedder(embed_model)

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[Dict]:
        q_vec = self.embedder.encode([query], normalize=True)
        sims, idxs = self.store.search(q_vec, k=k)
        results: List[Dict] = []
        for sim, idx in zip(sims, idxs):
            if idx == -1:
                continue
            m = dict(self.store.meta[idx])
            m["score"] = float(sim)
            results.append(m)
        return results


# ---------------
# Antwort-Generator (offline & optional LLM)
# ---------------
"""
def format_context(hits: List[Dict], max_chars: int = DEFAULT_MAX_CTX_CHARS) -> str:
    buf = []
    total = 0
    for i, h in enumerate(hits, 1):
        snippet = h["text"].strip()
        part = f"[{i}] {snippet}\n(source: {h['source']})\n---\n"
        if total + len(part) > max_chars:
            break
        buf.append(part)
        total += len(part)
    return "".join(buf)
"""
def format_context(hits: List[Dict], max_chars: int = DEFAULT_MAX_CTX_CHARS) -> str:
    """
    Baut einen Kontext-String aus den Top-k Treffern.
    - Schneidet jeden Treffer zu, wenn er zu lang ist.
    - Garantiert, dass mindestens der erste Treffer enthalten ist,
      selbst wenn er allein schon größer als max_chars wäre.
    """
    if max_chars <= 0 or not hits:
        return ""

    buf = []
    total = 0

    for i, h in enumerate(hits, 1):
        snippet = (h.get("text") or "").strip()
        header = f"[{i}] "
        footer = f"\n(source: {h.get('source','unknown')})\n---\n"

        # maximal erlaubter Platz für den Textkörper dieses Treffers
        remaining = max_chars - total - len(header) - len(footer)

        if remaining <= 0:
            # kein Platz mehr – wenn noch nichts drin ist, nimm eine harte Kürzung
            if not buf and snippet:
                trimmed = snippet[: max(0, max_chars - len(header) - len(footer))]
                buf.append(header + trimmed + footer)
            break

        # falls der Ausschnitt zu lang ist: kürzen
        body = snippet if len(snippet) <= remaining else snippet[:remaining]

        # falls durch Kürzung gar nichts übrig wäre: erzwinge minimalen Ausschnitt beim ersten Treffer
        if not body and not buf and snippet:
            body = snippet[: max(1, max_chars - len(header) - len(footer))]

        # wenn immer noch nichts übrig: breche ab
        if not body:
            break

        part = header + body + footer
        buf.append(part)
        total += len(part)

        if total >= max_chars:
            break

    return "".join(buf)



def answer_offline(question: str, hits: List[Dict], max_chars: int = DEFAULT_MAX_CTX_CHARS) -> str:
    ctx = format_context(hits, max_chars=max_chars)
    if not ctx:
        return "(Offline) Keine passenden Auszüge gefunden."
    return "(Offline) Relevante Auszüge aus deinen PDFs:\n\n" + ctx


def answer_with_llm(question: str, hits: List[Dict], max_chars: int = DEFAULT_MAX_CTX_CHARS) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not (USE_LLM and api_key):
        return answer_offline(question, hits, max_chars=max_chars)

    ctx = format_context(hits, max_chars=max_chars)
    if not ctx:
        return "(LLM) Keine passenden Auszüge gefunden."

    system = "You are a helpful research assistant. Answer using ONLY the provided context. If missing, say 'Not found.' Cite sources as [1], [2]. Keep it concise."
    user = f"""Question: {question}

Context:
{ctx}
"""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM Fehler: {e})\n\n" + answer_offline(question, hits, max_chars=max_chars)


# ---------------
# CLI
# ---------------
def main():
    ap = argparse.ArgumentParser(description="RAG CLI (offline-fähig)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_ing = sub.add_parser("ingest", help="PDFs einlesen & Index bauen")
    ap_ing.add_argument("--data_dir", type=Path, required=True, help="Ordner mit PDFs")
    ap_ing.add_argument("--index_dir", type=Path, required=True, help="Zielordner für FAISS-Index")
    ap_ing.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    ap_ing.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    ap_ing.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL)

    ap_ask = sub.add_parser("ask", help="Frage stellen gegen bestehenden Index")
    ap_ask.add_argument("--index_dir", type=Path, required=True, help="Ordner, in dem der FAISS-Index liegt")
    ap_ask.add_argument("--question", type=str, required=True)
    ap_ask.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    ap_ask.add_argument("--max_context_chars", type=int, default=DEFAULT_MAX_CTX_CHARS)
    ap_ask.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL)
    ap_ask.add_argument("--show_sources", action="store_true", help="Zeigt die Top-k Quellen & Scores zusätzlich an")

    args = ap.parse_args()

    if args.cmd == "ingest":
        ingest_pdfs_to_faiss(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            embed_model=args.embed_model,
        )
        return

    if args.cmd == "ask":
        index_path = args.index_dir / "papers.index.faiss"
        retriever = Retriever(index_path=index_path, embed_model=args.embed_model)
        hits = retriever.retrieve(args.question, k=args.k)

        if args.show_sources:
            print("\n=== Treffer (Top-k) ===")
            for i, h in enumerate(hits, 1):
                print(f"[{i}] score={h['score']:.4f} source={h['source']}")
                print(h["text"][:300].strip().replace("\n", " "), "...\n")

        # Antwort offline (oder, falls OPENAI_API_KEY vorhanden, via LLM)
        ans = answer_with_llm(args.question, hits, max_chars=args.max_context_chars)
        print("\n=== Antwort ===")
        print(ans)
        return


if __name__ == "__main__":
    main()
