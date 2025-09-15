from __future__ import annotations
import os, json
import numpy as np
import faiss
from typing import Tuple, List, Dict
from .embeddings import Embedder

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def build_index(chunks: List[dict], index_dir: str = "indices/faiss", model_name: str | None = None) -> Tuple[str, str]:
    _ensure_dir(index_dir)
    meta_path = os.path.join(index_dir, "meta.jsonl")
    index_path = os.path.join(index_dir, "index.faiss")

    embedder = Embedder(model_name) if model_name else Embedder()
    texts = [c["text"] for c in chunks]
    embs = embedder.embed_texts(texts)  # shape [N, D], normalized float32
    d = embs.shape[1]

    index = faiss.IndexFlatIP(d)  # inner product == cosine (norm=1)
    index.add(embs)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            rec = {**c, "_id": i}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # speichere auch die Modelinfo
    with open(os.path.join(index_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"embedder": embedder.model_name, "dim": d}, f)

    return index_path, meta_path

def load_index(index_dir: str = "indices/faiss") -> Tuple[faiss.Index, List[Dict], Embedder]:
    index_path = os.path.join(index_dir, "index.faiss")
    meta_path  = os.path.join(index_dir, "meta.jsonl")
    cfg_path   = os.path.join(index_dir, "config.json")

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        raise FileNotFoundError("FAISS-Index oder Metadatei fehlt. Bitte zuerst ingesten/builden.")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    embedder = Embedder(cfg.get("embedder"))

    metas: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))

    index = faiss.read_index(index_path)
    return index, metas, embedder
