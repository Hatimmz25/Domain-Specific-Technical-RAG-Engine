import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.retriever import DenseRetriever
from app.retrieval.reranker import CrossEncoderReranker


def test_reranker_pipeline():
    """
    Tests end-to-end retrieval and re-ranking:
    Query -> FAISS Dense Search (Top N=15) -> Cross-Encoder Reranker (Top K=5).
    """
    print("=" * 60)
    print("      Testing Candidate Retrieval & Cross-Encoder Re-Ranking")
    print("=" * 60)

    # Step 1: Load FAISS vector store
    vector_store = FAISSVectorStore()
    vector_store.load_index()

    # Step 2: Dense retriever fetching Top N candidates
    retriever = DenseRetriever(
        vector_store=vector_store,
        top_k=settings.RETRIEVAL_TOP_K,
        similarity_threshold=0.2
    )

    # Step 3: Initialize Cross-Encoder Re-Ranker
    reranker = CrossEncoderReranker(
        model_name=settings.RERANKER_MODEL,
        top_k=settings.RERANK_TOP_K
    )

    test_query = "How do I create a POST endpoint in FastAPI?"
    print(f"\nQuery: '{test_query}'")
    print("-" * 50)

    # Stage 1: Dense Retrieval
    candidate_pairs = retriever.get_relevant_documents_with_scores(test_query)
    candidate_docs = [doc for doc, score in candidate_pairs]
    print(f"[Stage 1] Dense Retrieval fetched {len(candidate_docs)} candidates from FAISS.")

    # Stage 2: Cross-Encoder Re-Ranking
    reranked_output = reranker.rerank(query=test_query, candidates=candidate_docs)
    print(f"[Stage 2] Cross-Encoder Re-Ranker selected Top {len(reranked_output)} chunks:\n")

    for rank, item in enumerate(reranked_output, start=1):
        doc = item["document"]
        score = item["score"]
        print(f"Rank #{rank} [Re-Ranker Score: {score:.4f}]")
        print(f"  Chunk ID : {doc.metadata.get('chunk_id')}")
        print(f"  Source   : {doc.metadata.get('file_name')}")
        print(f"  Snippet  : {doc.page_content[:130]}...\n")


if __name__ == "__main__":
    test_reranker_pipeline()