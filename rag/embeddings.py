from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-large"

@dataclass
class Embedder:
    model_name: str = DEFAULT_MODEL

    def __post_init__(self):
        self._model = SentenceTransformer(self.model_name)

    @property
    def dim(self) -> int:
        # sentence-transformers liefert bei encode die Dim indirekt
        v = self._model.encode(["probe"], normalize_embeddings=True)
        return int(v.shape[1])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # L2-normalisierte Embeddings → inner product ~ cosine
        arr = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(arr, dtype="float32")

    def embed_query(self, q: str) -> np.ndarray:
        # E5 nutzt Prefix "query: "
        return self.embed_texts([f"query: {q}"])[0]
