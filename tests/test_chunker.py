import pytest
from app.ingestion.loaders import Document
from app.ingestion.chunker import RecursiveChunker, SemanticChunker


def test_recursive_chunker_basic():
    text = "Sentence one. " * 50  # ~700 characters
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split_text(text)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 250


def test_semantic_chunker_metadata_inheritance():
    doc = Document(
        page_content="FastAPI is a web framework for Python. It uses Pydantic for validation. Docker containerizes apps.",
        metadata={"source": "test.md", "document_id": "doc_123"}
    )
    
    chunker = SemanticChunker(max_chunk_size=200, breakpoint_percentile_threshold=50)
    chunked_docs = chunker.split_documents([doc])
    
    assert len(chunked_docs) >= 1
    for chunk in chunked_docs:
        assert chunk.metadata["source"] == "test.md"
        assert chunk.metadata["document_id"] == "doc_123"
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata