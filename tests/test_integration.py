import pytest
from app.ingestion.loaders import Document
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.retriever import DenseRetriever
from app.pipeline import TechnicalRAGEngine


def test_full_rag_pipeline_normal_flow():
    """Tests normal flow: Ingestion -> Retrieval -> Reranking -> Grounded Prompt -> Citation."""
    doc = Document(
        page_content="To create a POST endpoint in FastAPI, use @app.post('/path') with a Pydantic model.",
        metadata={
            "source": "fastapi.md",
            "file_name": "fastapi.md",
            "chunk_id": "chunk_fastapi_001",
        },
    )

    store = FAISSVectorStore()
    store.add_documents([doc])

    engine = TechnicalRAGEngine(vector_store=store, use_mock_llm=True)
    response = engine.ask("How do I create a POST endpoint in FastAPI?")

    assert response["question"] == "How do I create a POST endpoint in FastAPI?"
    assert len(response["sources"]) > 0
    assert response["sources"][0]["chunk_id"] == "chunk_fastapi_001"
    assert response["latency_ms"]["total"] >= 0.0


def test_rag_pipeline_empty_index():
    """Verifies RAG engine gracefully handles queries when vector store is empty."""
    empty_store = FAISSVectorStore()  # No documents added
    engine = TechnicalRAGEngine(vector_store=empty_store, use_mock_llm=True)

    response = engine.ask("How do I create a POST endpoint in FastAPI?")

    assert (
        "couldn't find sufficient information"
        in response["answer"].lower()
    )
    assert response["sources"] == []
    assert response["retrieved_context"] == []


def test_rag_pipeline_out_of_domain_query():
    """Verifies score cutoffs filter out low-confidence results for out-of-domain questions."""
    doc = Document(
        page_content="FastAPI is a Python web framework for building REST APIs.",
        metadata={
            "source": "fastapi.md",
            "file_name": "fastapi.md",
            "chunk_id": "c1",
        },
    )

    store = FAISSVectorStore()
    store.add_documents([doc])

    engine = TechnicalRAGEngine(vector_store=store, use_mock_llm=True)

    # Query completely unrelated to Python/FastAPI
    response = engine.ask("What is the capital city of France?")

    assert (
        "couldn't find sufficient information"
        in response["answer"].lower()
    )
    assert response["sources"] == []
    assert response["retrieved_context"] == []


def test_rag_pipeline_empty_query():
    """Verifies handling of blank or whitespace-only query strings."""
    store = FAISSVectorStore()
    engine = TechnicalRAGEngine(vector_store=store, use_mock_llm=True)

    response = engine.ask("   ")
    assert response["answer"] == "Query string cannot be empty."
    assert response["sources"] == []
    assert response["latency_ms"]["total"] == 0.0