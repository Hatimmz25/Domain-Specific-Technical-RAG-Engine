import pytest
from fastapi.testclient import TestClient

from main import app
from app.api.routes import get_rag_engine
from app.pipeline import TechnicalRAGEngine
from app.ingestion.loaders import Document
from app.retrieval.vector_store import FAISSVectorStore


@pytest.fixture
def mock_engine():
    store = FAISSVectorStore()
    docs = [
        Document(
            page_content="FastAPI is a modern Python web framework for building APIs.",
            metadata={
                "source": "fastapi_doc.md",
                "file_name": "fastapi_doc.md",
                "chunk_id": "doc_fastapi_001",
                "file_hash": "hash_fastapi_001",
            },
        )
    ]
    store.build_index(docs)
    return TechnicalRAGEngine(vector_store=store, use_mock_llm=True)


@pytest.fixture
def client(mock_engine):
    app.dependency_overrides[get_rag_engine] = lambda: mock_engine
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_check_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_system_stats_endpoint(client):
    response = client.get("/api/v1/stats")
    assert response.status_code == 200


def test_query_endpoint_success(client):
    payload = {"question": "How do I create a POST endpoint in FastAPI?"}
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200


def test_query_endpoint_validation_error(client):
    response = client.post("/api/v1/query", json={"question": "hi"})
    assert response.status_code == 422


def test_upload_document_endpoint(client):
    file_content = b"# Docker Guide\nDocker containerizes applications cleanly."
    files = {"file": ("docker_guide.md", file_content, "text/markdown")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 200


def test_upload_unsupported_file_type(client):
    files = {"file": ("malicious.exe", b"\x00\x01\x02", "application/octet-stream")}
    response = client.post("/api/v1/upload", files=files)
    assert response.status_code == 500