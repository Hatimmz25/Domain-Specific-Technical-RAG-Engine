import pytest
import numpy as np
from app.ingestion.loaders import Document
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.retriever import DenseRetriever


def test_faiss_vector_store_build_and_search():
    docs = [
        Document(page_content="FastAPI POST endpoint creation.", metadata={"chunk_id": "c1", "file_name": "f1.md"}),
        Document(page_content="Docker COPY vs ADD command comparison.", metadata={"chunk_id": "c2", "file_name": "f2.txt"}),
    ]
    
    store = FAISSVectorStore()
    store.build_index(docs)
    
    assert store.index.ntotal == 2
    
    results = store.search("FastAPI endpoint", top_k=1)
    assert len(results) == 1
    retrieved_doc, score = results[0]
    assert "FastAPI" in retrieved_doc.page_content
    assert score > 0.0


def test_dense_retriever_threshold_filtering():
    docs = [
        Document(page_content="FastAPI framework path parameters.", metadata={"chunk_id": "c1", "file_name": "f1.md"}),
    ]
    
    store = FAISSVectorStore()
    store.build_index(docs)
    
    # Set high similarity threshold to verify filtering
    retriever = DenseRetriever(vector_store=store, top_k=5, similarity_threshold=0.99)
    results = retriever.get_relevant_documents_with_scores("Random query about cooking")
    
    assert len(results) == 0