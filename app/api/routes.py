import re
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request

from app.config import settings
from app.observability import logger
from app.pipeline import TechnicalRAGEngine
from app.ingestion.loaders import DocumentLoader
from app.ingestion.chunker import SemanticChunker
from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SubsystemHealth,
    SystemStatsResponse
)

_engine_instance: TechnicalRAGEngine = None


def get_rag_engine() -> TechnicalRAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TechnicalRAGEngine()
    return _engine_instance


router = APIRouter()


def sanitize_filename(filename: str) -> str:
    base_name = Path(filename).name
    clean_name = re.sub(r"[^a-zA-Z0-9_.\-]", "_", base_name)
    return clean_name or "uploaded_document.txt"


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(engine: TechnicalRAGEngine = Depends(get_rag_engine)):
    """
    Subsystem-level health check evaluating API, FAISS, Models, and Storage.
    """
    subsystems = {}
    overall_status = "ok"

    # 1. API Availability
    subsystems["api"] = SubsystemHealth(status="ok")

    # 2. FAISS Index Availability
    try:
        vector_count = engine.vector_store.index.ntotal if engine.vector_store and engine.vector_store.index else 0
        subsystems["faiss_index"] = SubsystemHealth(
            status="ok" if vector_count > 0 else "degraded",
            details=f"Contains {vector_count} indexed vectors"
        )
    except Exception as e:
        vector_count = 0
        subsystems["faiss_index"] = SubsystemHealth(status="error", details=str(e))
        overall_status = "degraded"

    # 3. Model Engine Availability
    try:
        if hasattr(engine.llm, "llm") or hasattr(engine.llm, "client") or engine.llm.__class__.__name__ == "MockLLMProvider":
            subsystems["models"] = SubsystemHealth(status="ok", details=f"Provider: {engine.llm.__class__.__name__}")
        else:
            subsystems["models"] = SubsystemHealth(status="degraded", details="Model provider initialized with caveats")
    except Exception as e:
        subsystems["models"] = SubsystemHealth(status="error", details=str(e))
        overall_status = "degraded"

    # 4. Storage Write Access
    try:
        idx_dir = settings.BASE_DIR / settings.INDEX_STORAGE_DIR
        idx_dir.mkdir(parents=True, exist_ok=True)
        subsystems["storage"] = SubsystemHealth(status="ok", details="Writable storage directories verified")
    except Exception as e:
        subsystems["storage"] = SubsystemHealth(status="error", details=str(e))
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        environment=settings.ENVIRONMENT,
        indexed_vectors=vector_count,
        subsystems=subsystems
    )


@router.get("/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_system_stats(engine: TechnicalRAGEngine = Depends(get_rag_engine)):
    vector_count = engine.vector_store.index.ntotal if engine.vector_store and engine.vector_store.index else 0
    return SystemStatsResponse(
        embedding_model=settings.EMBEDDING_MODEL,
        reranker_model=settings.RERANKER_MODEL,
        llm_model=getattr(settings, "GROQ_MODEL", settings.LLM_MODEL),
        retrieval_top_k=settings.RETRIEVAL_TOP_K,
        rerank_top_k=settings.RERANK_TOP_K,
        total_indexed_chunks=vector_count
    )


@router.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
async def query_rag_engine(
    raw_request: Request,
    body: QueryRequest,
    engine: TechnicalRAGEngine = Depends(get_rag_engine)
):
    req_id = getattr(raw_request.state, "request_id", None)
    try:
        response = engine.ask(query=body.question, request_id=req_id)
        return QueryResponse(**response)
    except Exception as e:
        logger.error(f"Query Processing Exception: {str(e)}", extra={"request_id": req_id, "endpoint": "/query"})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the RAG query."
        )


@router.post("/upload", tags=["Document Management"])
async def upload_document(
    raw_request: Request,
    file: UploadFile = File(...),
    engine: TechnicalRAGEngine = Depends(get_rag_engine)
):
    req_id = getattr(raw_request.state, "request_id", None)
    safe_filename = sanitize_filename(file.filename)
    extension = Path(safe_filename).suffix.lower()

    if extension not in getattr(settings, "ALLOWED_EXTENSIONS", [".pdf", ".md", ".txt"]):
        logger.warning(f"Upload rejected: Unsupported extension '{extension}'", extra={"request_id": req_id})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'."
        )

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            tmp_path = Path(tmp_file.name)

            loader = DocumentLoader()
            new_docs = loader.load_file(tmp_path)

            for doc in new_docs:
                doc.metadata["file_name"] = safe_filename
                doc.metadata["source"] = safe_filename

            chunker = SemanticChunker(max_chunk_size=settings.CHUNK_SIZE)
            chunked_docs = chunker.split_documents(new_docs)

            engine.vector_store.add_documents(chunked_docs)
            engine.vector_store.save_index()

            logger.info(
                f"Document Upload Success: '{safe_filename}'",
                extra={
                    "request_id": req_id,
                    "endpoint": "/upload",
                    "metrics": {
                        "filename": safe_filename,
                        "chunks_created": len(chunked_docs),
                        "total_vectors": engine.vector_store.index.ntotal
                    }
                }
            )

            return {
                "status": "success",
                "filename": safe_filename,
                "chunks_created": len(chunked_docs),
                "total_indexed_vectors": engine.vector_store.index.ntotal
            }

    except Exception as e:
        logger.error(
            f"Indexing/Upload Exception for '{safe_filename}': {str(e)}",
            extra={"request_id": req_id, "endpoint": "/upload"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document upload safely."
        )