#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit UI für deinen lokalen RAG-Assistenten (offline)
- Lädt bestehenden FAISS-Index (gebaut per ingest)
- Fragt Top-k Chunks ab
- Zeigt kuratierte Auszüge + Quellen an
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

# ---------------- Config ----------------
DEFAULT_INDEX_DIR = Path("indices/faiss")
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cleaning-Heuristiken (gleiche Idee wie im CLI)
URL_RE = re.compile(r"https?://\S+")
MULTISPACE_RE = re.compile(r"[ \t]+")

def clean_snippet(s: str) -> str:
    if not s: return s
    lines = []
    for line in s.splitlines():
        url_count = len(URL_RE.findall(line))
        if url_count >= 3 and len(line) > 300:
            line = URL_RE.sub("", line)
        lines.append(line)
    s2 = "\n".join(lines)
    s2 = MULTISPACE_RE.sub(" ", s2)
    return s2.strip()

# ------------- FAISS Store --------------
class FaissStore:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.meta_path = index_path.with_suffix(".meta.jsonl")
        self.index = None
        self.meta: List[Dict] = []

    def load(self):
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Index oder Metadaten fehlen bei {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.meta = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    import json
                    self.meta.append(json.loads(line))

    def search(self, qvec: np.ndarray, k: int):
        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)
        sims, idxs = self.index.search(qvec, k)
        return sims[0], idxs[0]

# ------------- Embedder -----------------
@st.cache_resource(show_spinner=False)
def get_embedder(name: str):
    return SentenceTransformer(name)

# ------------- Loader -------------------
@st.cache_resource(show_spinner=False)
def load_store(index_dir: Path):
    index_path = index_dir / "papers.index.faiss"
    store = FaissStore(index_path)
    store.load()
    return store

# ------------- Retrieval ----------------
def retrieve(store: FaissStore, embedder: SentenceTransformer, query: str, k: int, min_score: float):
    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")
    sims, idxs = store.search(q_vec, k=k)
    hits = []
    for sim, idx in zip(sims, idxs):
        if idx == -1: 
            continue
        if sim < min_score:
            continue
        m = dict(store.meta[idx])
        m["score"] = float(sim)
        hits.append(m)
    return hits

def format_context(hits: List[Dict], max_chars: int, per_snippet_chars: int) -> str:
    if max_chars <= 0 or not hits:
        return ""
    buf, total = [], 0
    for i, h in enumerate(hits, 1):
        snippet = clean_snippet((h.get("text") or "").strip())
        if per_snippet_chars > 0 and len(snippet) > per_snippet_chars:
            snippet = snippet[:per_snippet_chars]
        header = f"[{i}] "
        footer = f"\n(source: {h.get('source','unknown')})\n---\n"
        remaining = max_chars - total - len(header) - len(footer)
        if remaining <= 0:
            if not buf and snippet:
                trimmed = snippet[: max(0, max_chars - len(header) - len(footer))]
                buf.append(header + trimmed + footer)
            break
        body = snippet if len(snippet) <= remaining else snippet[:remaining]
        if not body:
            if not buf and snippet:
                body = snippet[: max(1, max_chars - len(header) - len(footer))]
            else:
                break
        part = header + body + footer
        buf.append(part)
        total += len(part)
        if total >= max_chars:
            break
    return "".join(buf)

# ---------------- UI --------------------
st.set_page_config(page_title="🎓 Local RAG Assistant", page_icon="🎓")
st.title("🎓 Local Research Assistant (RAG, offline)")

with st.sidebar:
    st.subheader("Index & Modelle")
    index_dir_str = st.text_input("Index directory", str(DEFAULT_INDEX_DIR))
    embed_model = st.text_input("Embedding model", DEFAULT_EMBED_MODEL)
    st.divider()
    st.subheader("Retrieval-Parameter")
    k = st.slider("Top-k Treffer", 1, 20, 8)
    min_score = st.slider("Min. Score (Filter)", 0.0, 0.5, 0.05, 0.01)
    per_snippet_chars = st.slider("Max Zeichen pro Snippet", 200, 3000, 1200, 50)
    max_context_chars = st.slider("Max Gesamt-Kontext", 1000, 20000, 12000, 500)

index_dir = Path(index_dir_str)

try:
    store = load_store(index_dir)
    embedder = get_embedder(embed_model)
    st.success(f"Index geladen: {index_dir}/papers.index.faiss  |  Einträge: {store.index.ntotal}")
except Exception as e:
    st.error(f"Problem beim Laden: {e}")
    st.stop()

query = st.text_input("Deine Frage", value="Was sind die Lernziele der Vorlesung Informationssicherheit?")
ask = st.button("Fragen")

if ask and query.strip():
    with st.spinner("Suche relevante Textstellen..."):
        hits = retrieve(store, embedder, query.strip(), k=k, min_score=min_score)

    if not hits:
        st.warning("Keine Treffer. Erhöhe k oder senke den Min-Score.")
    else:
        st.markdown("### Antwort (offline, kuratierte Auszüge)")
        ctx = format_context(hits, max_chars=max_context_chars, per_snippet_chars=per_snippet_chars)
        st.code(ctx, language="markdown")

        st.markdown("### Quellen")
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}]** `{h.get('source','?')}` — Score: {h['score']:.4f}")
            st.write(h["text"][:400] + ("..." if len(h["text"])>400 else ""))
            st.divider()
