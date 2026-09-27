import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.observability import RequestContextMiddleware, logger
from app.api.routes import router as api_router

app = FastAPI(
    title="Domain-Specific Technical RAG Engine API",
    description="Automated technical-documentation Q&A engine using FAISS, Cross-Encoder Re-Ranking, and Qwen2.5/Llama-3.1.",
    version="1.0.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None
)

# Attach Correlation ID & Observability Middleware
app.add_middleware(RequestContextMiddleware)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "ALLOWED_CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Domain-Specific Technical RAG Engine API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=(settings.ENVIRONMENT == "development")
    )