import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.retrieval.vector_store import FAISSVectorStore


def run_search_test():
    """Tests loading persistent FAISS index and running query similarity search."""
    print("=" * 60)
    print("      Testing FAISS Vector Search & Metadata Retrieval")
    print("=" * 60)

    vector_store = FAISSVectorStore()
    vector_store.load_index()

    test_queries = [
        "How do I create a POST endpoint in FastAPI?",
        "What is the difference between COPY and ADD in Docker?"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        results = vector_store.search(query, top_k=2)

        for rank, (doc, score) in enumerate(results, start=1):
            print(f"Rank #{rank} [Similarity Score: {score:.4f}]")
            print(f"Source Document : {doc.metadata.get('file_name')}")
            print(f"Chunk ID        : {doc.metadata.get('chunk_id')}")
            print(f"Content Snippet : {doc.page_content[:150]}...")
            print()


if __name__ == "__main__":
    run_search_test()