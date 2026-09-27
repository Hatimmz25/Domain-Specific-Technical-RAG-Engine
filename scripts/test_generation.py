import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.generation.prompts import build_grounded_prompt
from app.generation.llm import MockLLMProvider
from app.ingestion.loaders import Document


def test_generation_layer():
    """
    Validates prompt construction and LLM generation interface behavior.
    """
    print("=" * 60)
    print("      Testing Grounded Prompt Design & LLM Interface")
    print("=" * 60)

    # 1. Simulate retrieved candidate context block
    dummy_doc = Document(
        page_content="To create a POST endpoint in FastAPI, use the @app.post() decorator with a Pydantic model.",
        metadata={"file_name": "fastapi_tutorial.md", "chunk_id": "doc_fastapi_001"}
    )
    mock_context = [{
        "document": dummy_doc,
        "score": 0.924,
        "metadata": dummy_doc.metadata
    }]

    # 2. Test Grounded Prompt Construction
    query = "How do I create a POST endpoint in FastAPI?"
    prompt = build_grounded_prompt(query, mock_context)

    print("\n--- Constructed Grounded Prompt ---")
    print(prompt)
    print("-" * 50)

    # 3. Test LLM Generation via Provider Interface
    llm_provider = MockLLMProvider()
    response = llm_provider.generate(prompt)

    print("\n--- LLM Generated Output ---")
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    test_generation_layer()