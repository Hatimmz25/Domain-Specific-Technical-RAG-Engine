import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.loaders import DocumentLoader
from app.ingestion.chunker import SemanticChunker, RecursiveChunker


def run_chunking_verification():
    """Validates both Semantic and Recursive fallback chunkers on ingested docs."""
    print("=" * 60)
    print("      Verifying Semantic & Recursive Chunking Architecture")
    print("=" * 60)

    raw_dir = settings.BASE_DIR / settings.DATA_RAW_DIR
    loader = DocumentLoader()
    raw_docs = loader.load_directory(raw_dir)

    if not raw_docs:
        print("[ERROR] No raw documents found. Run `python scripts/ingest.py` first.")
        return

    print(f"\n[✓] Loaded {len(raw_docs)} parent raw document(s).")

    # 1. Test Fallback Recursive Chunker
    print("\n--- Testing Recursive Chunker ---")
    rec_chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    rec_chunks = rec_chunker.split_text(raw_docs[0].page_content)
    print(f"Parent Doc 1 Char Length: {len(raw_docs[0].page_content)}")
    print(f"Recursive Chunker Created: {len(rec_chunks)} chunk(s)")
    for i, chunk in enumerate(rec_chunks[:2]):
        print(f"  [Chunk #{i+1} ({len(chunk)} chars)]: {chunk[:80]}...")

    # 2. Test Semantic Chunker
    print("\n--- Testing Semantic Chunker (Model: BAAI/bge-small-en-v1.5) ---")
    semantic_chunker = SemanticChunker(max_chunk_size=300, breakpoint_percentile_threshold=80)
    semantic_doc_chunks = semantic_chunker.split_documents(raw_docs)

    print(f"[SUCCESS] Semantic Chunker created {len(semantic_doc_chunks)} total chunk Document(s).")
    for idx, doc_chunk in enumerate(semantic_doc_chunks[:3], start=1):
        print(f"\n--- Semantic Chunk Document #{idx} ---")
        print(f"Chunk ID: {doc_chunk.metadata.get('chunk_id')}")
        print(f"Source: {doc_chunk.metadata.get('file_name')}")
        print(f"Position: {doc_chunk.metadata.get('chunk_index') + 1} of {doc_chunk.metadata.get('total_chunks')}")
        print(f"Character Count: {doc_chunk.metadata.get('chunk_char_length')}")
        print(f"Content:\n{doc_chunk.page_content}")


if __name__ == "__main__":
    run_chunking_verification()