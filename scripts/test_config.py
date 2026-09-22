import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings


def verify_environment():
    """Validates configuration parameters and creates required project directories."""
    print("=" * 60)
    print("      Domain-Specific Technical RAG Engine Environment Check")
    print("=" * 60)

    print(f"[✓] Python Version: {sys.version.split()[0]}")
    print(f"[✓] App Environment: {settings.ENVIRONMENT}")
    print(f"[✓] Base Directory: {settings.BASE_DIR}")
    print(f"[✓] Embedding Model Configured: {settings.EMBEDDING_MODEL}")
    print(f"[✓] Re-Ranker Model Configured: {settings.RERANKER_MODEL}")
    print(f"[✓] LLM Target Model: {settings.LLM_MODEL} ({settings.LLM_MODEL_FILE})")
    print(f"[✓] Dense Retrieval Top-K: {settings.RETRIEVAL_TOP_K}")
    print(f"[✓] Re-Ranker Top-K: {settings.RERANK_TOP_K}")

    # Ensure required data and index directories exist
    dirs_to_create = [
        settings.BASE_DIR / settings.DATA_RAW_DIR,
        settings.BASE_DIR / settings.DATA_PROCESSED_DIR,
        settings.BASE_DIR / settings.INDEX_STORAGE_DIR,
    ]

    print("\n--- Checking Project Scaffolding Directories ---")
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        # Touch a .gitkeep file to retain directory structure in Git
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
        print(f"[✓] Confirmed Directory: {directory}")

    print("\n[SUCCESS] Environment setup and configuration layer initialized successfully!")


if __name__ == "__main__":
    verify_environment()