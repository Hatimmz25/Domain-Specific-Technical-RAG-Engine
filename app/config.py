import os
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration management for the RAG engine using Pydantic Settings V2.
    """
    # System Environment
    ENVIRONMENT: str = Field(default="production", description="App runtime environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Security & Public Demo Limits
    ALLOWED_CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Origins permitted for CORS requests"
    )
    MAX_FILE_SIZE_MB: int = Field(default=5, description="Maximum allowed upload size per file in MB")
    MAX_INDEXED_CHUNKS: int = Field(default=1000, description="Global maximum chunk cap for public demo")
    ALLOWED_EXTENSIONS: List[str] = Field(default=[".pdf", ".md", ".txt"], description="Permitted file extensions")
    
    # Rate Limits (Slowapi syntax)
    RATE_LIMIT_QUERY: str = Field(default="15/minute", description="Query rate limit per client IP")
    RATE_LIMIT_UPLOAD: str = Field(default="5/minute", description="Upload rate limit per client IP")

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = Field(default=Path("data/raw"), description="Directory for raw documents")
    DATA_PROCESSED_DIR: Path = Field(default=Path("data/processed"), description="Directory for parsed artifacts")
    INDEX_STORAGE_DIR: Path = Field(default=Path("indexes"), description="Directory for FAISS index files")

    # Chunking Options
    CHUNK_SIZE: int = Field(default=500, description="Target chunk size in tokens/characters for chunkers")
    CHUNK_OVERLAP: int = Field(default=50, description="Overlap between consecutive chunks")
    SEMANTIC_CHUNKING_THRESHOLD_PERCENTILE: int = Field(
        default=95, description="Percentile threshold for sentence distance breaks in semantic chunking"
    )

    # Retrieval Models & Search
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model string for dense sentence vectorization"
    )
    RETRIEVAL_TOP_K: int = Field(
        default=15, description="Number of candidate chunks fetched by dense retrieval"
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.3, description="Minimum cosine similarity cutoff threshold for retrieval"
    )

    # Re-Ranking Model
    RERANKER_MODEL: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-Encoder model for context score re-ranking"
    )
    RERANK_TOP_K: int = Field(
        default=5, description="Number of final context chunks supplied to LLM after re-ranking"
    )

    # LLM Options
    LLM_PROVIDER: str = Field(
        default="groq",
        description="LLM provider switch: 'groq' for public cloud demo, 'llama_cpp' for local dev"
    )
    GROQ_API_KEY: str = Field(default="", description="API Key for Groq cloud inference")
    GROQ_MODEL: str = Field(default="llama-3.1-8b-instant", description="Groq model ID")
    
    LLM_MODEL: str = Field(
        default="bartowski/Qwen2.5-3B-Instruct-GGUF",
        description="HuggingFace repository repo ID for Qwen2.5 GGUF weights"
    )
    LLM_MODEL_FILE: str = Field(
        default="Qwen2.5-3B-Instruct-Q8_0.gguf",
        description="GGUF quantization weight filename"
    )
    LLM_TEMPERATURE: float = Field(default=0.1, description="Sampling temperature for LLM text generation")
    LLM_MAX_TOKENS: int = Field(default=512, description="Maximum generated token limit for demo responses")
    LLM_CONTEXT_WINDOW: int = Field(default=4096, description="LLM prompt context window capacity")

    # API Settings
    API_HOST: str = Field(default="0.0.0.0", description="FastAPI host address")
    API_PORT: int = Field(default=8000, description="FastAPI server port")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()