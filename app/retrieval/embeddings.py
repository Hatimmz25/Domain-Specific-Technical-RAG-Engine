from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingEngine:
    """
    Wrapper around SentenceTransformers to generate L2-normalized dense vector embeddings.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        print(f"[INFO] Initializing EmbeddingEngine with model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Modern sentence-transformers method compatibility check
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)