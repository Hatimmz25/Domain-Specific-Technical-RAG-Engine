from typing import List, Tuple, Dict, Any
import torch
from sentence_transformers import CrossEncoder

from app.config import settings
from app.ingestion.loaders import Document


class CrossEncoderReranker:
    """
    Second-stage cross-encoder re-ranking engine.
    Scores and re-ranks top-N candidate chunks into top-K high-precision context documents.
    """

    def __init__(self, model_name: str = None, top_k: int = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.top_k = top_k or settings.RERANK_TOP_K
        
        # Detect CUDA GPU availability for rapid cross-encoder scoring
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Initializing CrossEncoderReranker with model: {self.model_name} on device: {self.device}")
        
        self.model = CrossEncoder(self.model_name, max_length=512, device=self.device)

    def rerank(
        self, query: str, candidates: List[Document]
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks candidate documents for a given query using Cross-Encoder joint attention.

        Args:
            query (str): User question or query string.
            candidates (List[Document]): Initial candidate documents fetched from FAISS.

        Returns:
            List[Dict[str, Any]]: Top-K re-ranked candidates formatted as dictionaries with relevance scores.
        """
        if not candidates:
            return []

        # Construct (query, candidate_text) pairs for Cross-Encoder evaluation
        pairs = [[query, doc.page_content] for doc in candidates]

        # Compute cross-encoder relevance scores
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Apply sigmoid normalization if model outputs unnormalized logits
        if len(raw_scores) > 0 and (max(raw_scores) > 1.0 or min(raw_scores) < 0.0):
            scores = 1.0 / (1.0 + torch.exp(-torch.tensor(raw_scores)).numpy())
        else:
            scores = raw_scores

        # Pair candidates with normalized relevance scores
        reranked_results = []
        for doc, score in zip(candidates, scores):
            reranked_results.append({
                "document": doc,
                "score": float(score),
                "metadata": doc.metadata
            })

        # Sort candidates descending by Cross-Encoder score
        reranked_results.sort(key=lambda x: x["score"], reverse=True)

        # Return Top-K context chunks
        return reranked_results[: self.top_k]