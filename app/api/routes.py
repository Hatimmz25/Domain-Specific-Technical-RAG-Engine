import re
import logging
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.pipeline import TechnicalRAGEngine
from app.ingestion.loaders import DocumentLoader
from app.ingestion.chunker import SemanticChunker
from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SystemStatsResponse
)

logger = logging.getLogger("rag_security")

# Rate Limiter setup
limiter = Limiter(key_func=get_remote_address)

# Global singleton storage
_engine_instance: TechnicalRAGEngine = None


def get_rag_engine() -> TechnicalRAGEngine:
    global _engine_instance
    if _engine_instance is None:
        logger.info("Initializing TechnicalRAGEngine singleton for API Server...")
        _engine_instance = TechnicalRAGEngine()
    return _engine_instance


router = APIRouter()


def sanitize_filename(filename: str) -> str:
    """Strips path traversal indicators and restricts characters."""
    base_name = Path(filename).name
    clean_name = re.sub(r"[^a-zA-Z0-9_.\-]", "_", base_name)
    return clean_name or "uploaded_document.txt"


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(engine: TechnicalRAGEngine = Depends(get_rag_engine)):
    vector_count = engine.vector_store.index.ntotal if engine.vector_store and engine.vector_store.index else 0
    return HealthResponse(
        status="ok",
        environment=settings.ENVIRONMENT,
        indexed_vectors=vector_count
    )


@router.get("/stats", response_model=SystemStatsResponse, tags=["System"])
async def get_system_stats(engine: TechnicalRAGEngine = Depends(get_rag_engine)):
    vector_count = engine.vector_store.index.ntotal if engine.vector_store and engine.vector_store.index else 0
    return SystemStatsResponse(
        embedding_model=settings.EMBEDDING_MODEL,
        reranker_model=settings.RERANKER_MODEL,
        llm_model=settings.LLM_MODEL,
        retrieval_top_k=settings.RETRIEVAL_TOP_K,
        rerank_top_k=settings.RERANK_TOP_K,
        total_indexed_chunks=vector_count
    )


@router.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
@limiter.limit(settings.RATE_LIMIT_QUERY)
async def query_rag_engine(
    request: Request,
    body: QueryRequest,
    engine: TechnicalRAGEngine = Depends(get_rag_engine)
):
    try:
        response = engine.ask(query=body.question)
        return QueryResponse(**response)
    except Exception as e:
        logger.error(f"Error during query execution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request. Please try again."
        )


@router.post("/upload", tags=["Document Management"])
@limiter.limit(settings.RATE_LIMIT_UPLOAD)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    engine: TechnicalRAGEngine = Depends(get_rag_engine)
):
    # 1. Filename Sanitization & Extension Verification
    safe_filename = sanitize_filename(file.filename)
    extension = Path(safe_filename).suffix.lower()

    if extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{extension}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    # 2. Vector Cap Guardrail for Anonymous Demo
    current_vectors = engine.vector_store.index.ntotal if engine.vector_store and engine.vector_store.index else 0
    if current_vectors >= settings.MAX_INDEXED_CHUNKS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Public demo capacity limit reached. Reset the index or try again later."
        )

    # 3. Size-limited streaming download to temporary storage with automatic cleanup
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    read_bytes = 0

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=extension) as tmp_file:
            while chunk := await file.read(64 * 1024):
                read_bytes += len(chunk)
                if read_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB."
                    )
                tmp_file.write(chunk)
            
            tmp_file.flush()
            tmp_path = Path(tmp_file.name)

            # Ingest and Chunk
            loader = DocumentLoader()
            new_docs = loader.load_file(tmp_path)

            # Re-assign safe public filename metadata
            for doc in new_docs:
                doc.metadata["file_name"] = safe_filename
                doc.metadata["source"] = safe_filename

            chunker = SemanticChunker(max_chunk_size=settings.CHUNK_SIZE)
            chunked_docs = chunker.split_documents(new_docs)

            # Build and persist updated vector index
            engine.vector_store.add_documents(chunked_docs)
            engine.vector_store.save_index()

            return {
                "status": "success",
                "filename": safe_filename,
                "chunks_created": len(chunked_docs),
                "total_indexed_vectors": engine.vector_store.index.ntotal
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed processing upload '{safe_filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document upload safely."
        )