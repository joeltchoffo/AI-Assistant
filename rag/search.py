from __future__ import annotations
import numpy as np
from typing import List, Dict
from .store_faiss import load_index

def search(query: str, top_k: int = 5, index_dir: str = "indices/faiss") -> List[Dict]:
    index, metas, embedder = load_index(index_dir)
    qv = embedder.embed_query(query).reshape(1, -1)  # shape [1, D]
    scores, idxs = index.search(qv, top_k)          # inner product scores
    hits: List[Dict] = []
    for score, ix in zip(scores[0], idxs[0]):
        if ix < 0: 
            continue
        m = metas[int(ix)]
        hits.append({
            "score": float(score),
            "text": m["text"],
            "doc_id": m["doc_id"],
            "page_start": m["page_start"],
            "page_end": m["page_end"],
            "_id": m["_id"],
        })
    return hits
