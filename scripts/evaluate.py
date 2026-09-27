import sys
import json
import logging
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.evaluation.evaluator import RetrievalEvaluator
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.retriever import DenseRetriever
from app.retrieval.reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    logging.info("Starting Automated RAG Engine Evaluation Benchmark...")

    vector_store = FAISSVectorStore()
    try:
        vector_store.load_index()
    except Exception as e:
        logging.error(f"Failed to load index for evaluation: {e}")
        return

    retriever = DenseRetriever(
        vector_store=vector_store,
        top_k=settings.RETRIEVAL_TOP_K,
        similarity_threshold=0.1,
    )
    reranker = CrossEncoderReranker(
        model_name=settings.RERANKER_MODEL, top_k=settings.RERANK_TOP_K
    )

    evaluator = RetrievalEvaluator()

    logging.info("Evaluating Stage-1: Dense Retrieval Only...")
    metrics_dense = evaluator.evaluate_retrieval_pipeline(
        retriever_func=retriever.get_relevant_documents_with_scores,
        k=5,
        use_reranker=False,
    )

    logging.info("Evaluating Stage-2: Dense Retrieval + Cross-Encoder Re-Ranking...")
    metrics_rerank = evaluator.evaluate_retrieval_pipeline(
        retriever_func=retriever.get_relevant_documents_with_scores,
        k=5,
        use_reranker=True,
        reranker_func=reranker.rerank,
    )

    report = {
        "dense_retrieval_only": metrics_dense,
        "dense_plus_reranking": metrics_rerank,
    }

    out_dir = settings.BASE_DIR / settings.DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_report.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info(f"Evaluation Complete! Results saved to: {out_file}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()