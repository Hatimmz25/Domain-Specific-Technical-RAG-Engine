import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.retriever import DenseRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.evaluation.evaluator import RetrievalEvaluator


def execute_benchmark_experiments():
    """
    Executes automated benchmark comparison across 3 pipeline configurations:
    - Experiment A: Dense Retrieval Only (FAISS Top 5)
    - Experiment B: Dense Retrieval Candidate Over-fetching (FAISS Top 15)
    - Experiment C: Dense Retrieval (Top 15) + Cross-Encoder Re-Ranking (Top 5)
    """
    print("=" * 70)
    print("      RAG ENGINE RETRIEVAL BENCHMARK EXPERIMENTS")
    print("=" * 70)

    evaluator = RetrievalEvaluator()
    
    # Initialize Core Vector Store & Reranker
    vector_store = FAISSVectorStore()
    vector_store.load_index()

    retriever_exp_a = DenseRetriever(
        vector_store=vector_store,
        top_k=5,
        similarity_threshold=0.1
    )

    retriever_exp_b_c = DenseRetriever(
        vector_store=vector_store,
        top_k=15,
        similarity_threshold=0.1
    )

    reranker = CrossEncoderReranker(
        model_name=settings.RERANKER_MODEL,
        top_k=5
    )

    # 1. Run Experiment A: Dense Retrieval (K=5)
    print("\n[Running Exp A] Dense Retrieval Only (FAISS Top 5)...")
    res_a = evaluator.evaluate_retrieval_pipeline(
        retriever_func=retriever_exp_a.get_relevant_documents_with_scores,
        k=5,
        use_reranker=False
    )

    # 2. Run Experiment B: Dense Retrieval Candidate Over-fetching (N=15, evaluated at K=5)
    print("[Running Exp B] Dense Retrieval Candidate Over-fetching (FAISS Top 15)...")
    res_b = evaluator.evaluate_retrieval_pipeline(
        retriever_func=retriever_exp_b_c.get_relevant_documents_with_scores,
        k=5,
        use_reranker=False
    )

    # 3. Run Experiment C: Dense Retrieval (Top 15) + Cross-Encoder Re-Ranking (Top 5)
    print("[Running Exp C] Dense Retrieval (Top 15) + Cross-Encoder Re-Ranking (Top 5)...")
    res_c = evaluator.evaluate_retrieval_pipeline(
        retriever_func=retriever_exp_b_c.get_relevant_documents_with_scores,
        k=5,
        use_reranker=True,
        reranker_func=reranker.rerank
    )

    # Print Final Benchmark Table
    print("\n" + "=" * 70)
    print(f"{'Method / Experiment':<35} | {'Precision@5':<11} | {'Recall@5':<9} | {'MRR':<6} | {'Hit Rate':<8} | {'Latency (ms)':<10}")
    print("-" * 70)
    print(f"{'Exp A: Dense Retrieval (Top 5)':<35} | {res_a['precision_at_k']*100:>9.1f}% | {res_a['recall_at_k']*100:>7.1f}% | {res_a['mrr']:>6.4f} | {res_a['hit_rate_at_k']*100:>6.1f}% | {res_a['avg_latency_ms']:>10.1f}")
    print(f"{'Exp B: Dense Candidate (Top 15)':<35} | {res_b['precision_at_k']*100:>9.1f}% | {res_b['recall_at_k']*100:>7.1f}% | {res_b['mrr']:>6.4f} | {res_b['hit_rate_at_k']*100:>6.1f}% | {res_b['avg_latency_ms']:>10.1f}")
    print(f"{'Exp C: Dense (Top 15) + Re-Rank':<35} | {res_c['precision_at_k']*100:>9.1f}% | {res_c['recall_at_k']*100:>7.1f}% | {res_c['mrr']:>6.4f} | {res_c['hit_rate_at_k']*100:>6.1f}% | {res_c['avg_latency_ms']:>10.1f}")
    print("=" * 70)


if __name__ == "__main__":
    execute_benchmark_experiments()