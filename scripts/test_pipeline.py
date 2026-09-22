import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.pipeline import TechnicalRAGEngine


def run_pipeline_test():
    """Validates master RAG engine orchestration across domain and out-of-domain queries."""
    print("=" * 60)
    print("      Testing Master RAG Engine Orchestrator")
    print("=" * 60)

    # Instantiate engine with MockLLM for rapid testing
    engine = TechnicalRAGEngine(use_mock_llm=True)

    queries = [
        "How do I create a POST endpoint in FastAPI?",
        "What is the capital city of France?"
    ]

    for q in queries:
        print(f"\nUser Query: '{q}'")
        print("-" * 50)
        response = engine.ask(q)

        print(f"Generated Answer:\n{response['answer']}\n")
        print("Sources Cited:")
        for src in response["sources"]:
            print(f" - {src['file_name']} (Chunk ID: {src['chunk_id']}, Score: {src['score']})")

        print("\nLatency Metrics (ms):")
        print(json.dumps(response["latency_ms"], indent=2))
        print("=" * 60)


if __name__ == "__main__":
    run_pipeline_test()