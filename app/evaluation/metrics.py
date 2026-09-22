from typing import List, Dict, Any, Set


class RetrievalMetrics:
    """
    Computes Precision@K, Recall@K, MRR, and Hit Rate@K for retrieval evaluation.
    """

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Computes Precision@K: (Relevant Retrieved in Top K) / K.
        """
        if k <= 0:
            return 0.0
        top_k_retrieved = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        if not relevant_set:
            return 0.0
        
        relevant_retrieved = sum(1 for cid in top_k_retrieved if cid in relevant_set)
        return relevant_retrieved / float(k)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Computes Recall@K: (Relevant Retrieved in Top K) / Total Ground Truth Relevant.
        """
        relevant_set = set(relevant_ids)
        if not relevant_set:
            return 0.0
        
        top_k_retrieved = retrieved_ids[:k]
        relevant_retrieved = sum(1 for cid in top_k_retrieved if cid in relevant_set)
        return relevant_retrieved / float(len(relevant_set))

    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Computes Reciprocal Rank (1 / rank of first relevant item in top-K).
        """
        relevant_set = set(relevant_ids)
        if not relevant_set:
            return 0.0

        top_k_retrieved = retrieved_ids[:k]
        for rank, cid in enumerate(top_k_retrieved, start=1):
            if cid in relevant_set:
                return 1.0 / float(rank)
        return 0.0

    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Computes Hit Rate@K: 1.0 if at least one relevant item is in top-K, else 0.0.
        """
        relevant_set = set(relevant_ids)
        if not relevant_set:
            return 0.0

        top_k_retrieved = retrieved_ids[:k]
        return 1.0 if any(cid in relevant_set for cid in top_k_retrieved) else 0.0