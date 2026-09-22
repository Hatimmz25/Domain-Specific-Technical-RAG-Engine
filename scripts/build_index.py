import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.loaders import DocumentLoader
from app.ingestion.chunker import SemanticChunker
from app.retrieval.vector_store import FAISSVectorStore


def build_and_save_vector_index():
    """
    Complete CLI pipeline script:
    Load Data -> Semantic Chunk -> Embed & Build FAISS Index -> Persist to Disk.
    """
    print("=" * 60)
    print("      Building & Persisting FAISS Vector Index")
    print("=" * 60)

    raw_dir = settings.BASE_DIR / settings.DATA_RAW_DIR
    
    # Step 1: Load raw document files
    loader = DocumentLoader()
    raw_docs = loader.load_directory(raw_dir)
    print(f"[✓] Step 1: Loaded {len(raw_docs)} raw document file(s).")

    if not raw_docs:
        print("[ERROR] No input documents found in data/raw. Add files and retry.")
        return

    # Step 2: Chunk documents semantically
    chunker = SemanticChunker(max_chunk_size= settings.CHUNK_SIZE)
    chunked_docs = chunker.split_documents(raw_docs)
    print(f"[✓] Step 2: Generated {len(chunked_docs)} semantic chunk(s).")

    # Step 3: Embed chunks & build FAISS index
    vector_store = FAISSVectorStore()
    vector_store.build_index(chunked_docs)

    # Step 4: Persist index and mapping
    vector_store.save_index()
    print("\n[SUCCESS] Vector indexing pipeline executed successfully.")


if __name__ == "__main__":
    build_and_save_vector_index()