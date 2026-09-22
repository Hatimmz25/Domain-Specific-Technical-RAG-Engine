from typing import List, Set
import numpy as np

class ExtendedRetrievalMetrics:
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if k <= 0 or not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        rel_set = set(relevant_ids)
        hits = sum(1 for cid in top_k if cid in rel_set)
        return hits / float(k)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        rel_set = set(relevant_ids)
        hits = sum(1 for cid in top_k if cid in rel_set)
        return hits / float(len(rel_set))

    @staticmethod
    def mrr_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        rel_set = set(relevant_ids)
        for rank, cid in enumerate(retrieved_ids[:k], start=1):
            if cid in rel_set:
                return 1.0 / float(rank)
        return 0.0

    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        rel_set = set(relevant_ids)
        return 1.0 if any(cid in rel_set for cid in retrieved_ids[:k]) else 0.0

    @staticmethod
    def compute_p95(latencies_ms: List[float]) -> float:
        if not latencies_ms:
            return 0.0
        return float(np.percentile(latencies_ms, 95))