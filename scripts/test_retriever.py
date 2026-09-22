import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.retriever import DenseRetriever


def test_dense_retriever():
    """
    Validates DenseRetriever integration with FAISS vector store and score cutoffs.
    """
    print("=" * 60)
    print("      Testing Dense candidate Retriever & Threshold Filter")
    print("=" * 60)

    # 1. Initialize and load FAISS vector store
    vector_store = FAISSVectorStore()
    vector_store.load_index()

    # 2. Instantiate DenseRetriever (N = RETRIEVAL_TOP_K)
    retriever = DenseRetriever(
        vector_store=vector_store,
        top_k=settings.RETRIEVAL_TOP_K,
        similarity_threshold=0.3
    )

    # Test Query 1: Relevant domain query
    query_in_domain = "How do I create a POST endpoint in FastAPI?"
    print(f"\n[Test 1] Domain Query: '{query_in_domain}'")
    print("-" * 50)
    
    # Retrieve using get_relevant_documents_with_scores
    candidates = retriever.get_relevant_documents_with_scores(query_in_domain)
    print(f"Retrieved {len(candidates)} candidate chunk(s) above cutoff threshold (0.3):")
    for rank, (doc, score) in enumerate(candidates, start=1):
        print(f"  Rank #{rank} [Similarity: {score:.4f}]")
        print(f"  Chunk ID : {doc.metadata.get('chunk_id')}")
        print(f"  Source   : {doc.metadata.get('file_name')}")
        print(f"  Snippet  : {doc.page_content[:120]}...\n")

    # Test Query 2: Out-of-domain query (Testing quality cutoff)
    query_out_of_domain = "What is the capital city of France?"
    print(f"\n[Test 2] Out-of-Domain Query: '{query_out_of_domain}'")
    print("-" * 50)
    
    out_candidates = retriever.get_relevant_documents_with_scores(query_out_of_domain)
    print(f"Retrieved {len(out_candidates)} candidate chunk(s) above threshold (0.3).")
    if not out_candidates:
        print("[SUCCESS] Quality guardrail working: All out-of-domain chunks filtered out!")

    # Test Query 3: Standard LangChain BaseRetriever API compatibility
    print("\n[Test 3] Testing LangChain standard `.invoke()` API interface:")
    print("-" * 50)
    lc_docs = retriever.invoke(query_in_domain)
    print(f"Retrieved {len(lc_docs)} standard LangChain Document object(s).")
    print(f"Sample Metadata Attached: {lc_docs[0].metadata}")


if __name__ == "__main__":
    test_dense_retriever()