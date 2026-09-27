import json
import time
from pathlib import Path
from typing import List, Dict, Any

from app.config import settings
from app.evaluation.metrics import RetrievalMetrics


class RetrievalEvaluator:
    """
    Evaluates retrieval configurations against the golden dataset.
    """

    def __init__(self, eval_dataset_path: Path = None):
        self.dataset_path = eval_dataset_path or (
            settings.BASE_DIR / settings.DATA_EVALUATION_DIR / "eval_dataset.json"
            if hasattr(settings, "DATA_EVALUATION_DIR")
            else settings.BASE_DIR / "data/evaluation/eval_dataset.json"
        )
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Loads evaluation QA dataset from JSON file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_retrieval_pipeline(
        self,
        retriever_func,
        k: int = 5,
        use_reranker: bool = False,
        reranker_func = None
    ) -> Dict[str, Any]:
        """
        Runs full evaluation loop across all questions in the golden dataset.

        Args:
            retriever_func: Function accepting query string and returning candidate (Document, score) pairs.
            k (int): Top-K evaluation parameter.
            use_reranker (bool): Whether to evaluate stage-2 re-ranking.
            reranker_func: Optional re-ranker function accepting query and candidate list.

        Returns:
            Dict[str, Any]: Aggregated precision, recall, MRR, hit rate, and latency metrics.
        """
        precisions = []
        recalls = []
        mrr_scores = []
        hit_rates = []
        latencies_ms = []

        # Filter out questions where relevant_chunk_ids is intentionally empty (e.g., absent info test cases)
        eval_items = [item for item in self.dataset if item.get("relevant_chunk_ids")]

        for item in eval_items:
            query = item["question"]
            relevant_ids = item["relevant_chunk_ids"]

            start_time = time.perf_counter()

            # Execute Candidate Retrieval
            raw_candidates = retriever_func(query)
            candidate_docs = [doc for doc, score in raw_candidates]

            # Execute Re-Ranking if enabled
            if use_reranker and reranker_func and candidate_docs:
                reranked_output = reranker_func(query, candidate_docs)
                retrieved_ids = [
                    res["metadata"].get("chunk_id") for res in reranked_output
                ]
            else:
                retrieved_ids = [
                    doc.metadata.get("chunk_id") for doc in candidate_docs
                ]

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Compute metrics for current query
            p_k = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k=k)
            r_k = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k=k)
            mrr = RetrievalMetrics.reciprocal_rank(retrieved_ids, relevant_ids, k=k)
            hit = RetrievalMetrics.hit_rate_at_k(retrieved_ids, relevant_ids, k=k)

            precisions.append(p_k)
            recalls.append(r_k)
            mrr_scores.append(mrr)
            hit_rates.append(hit)
            latencies_ms.append(latency_ms)

        num_queries = len(eval_items)
        avg_precision = sum(precisions) / num_queries if num_queries > 0 else 0.0
        avg_recall = sum(recalls) / num_queries if num_queries > 0 else 0.0
        avg_mrr = sum(mrr_scores) / num_queries if num_queries > 0 else 0.0
        avg_hit_rate = sum(hit_rates) / num_queries if num_queries > 0 else 0.0
        avg_latency = sum(latencies_ms) / num_queries if num_queries > 0 else 0.0

        return {
            "total_evaluated_queries": num_queries,
            "k": k,
            "precision_at_k": round(avg_precision, 4),
            "recall_at_k": round(avg_recall, 4),
            "mrr": round(avg_mrr, 4),
            "hit_rate_at_k": round(avg_hit_rate, 4),
            "avg_latency_ms": round(avg_latency, 2)
        }