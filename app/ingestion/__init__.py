import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import faiss
import numpy as np

from app.config import settings
from app.ingestion.loaders import Document
from app.retrieval.embeddings import EmbeddingEngine

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Production FAISS Vector Store manager for indexing, saving, loading,
    and performing inner product (cosine similarity) search mapped to document chunks.
    Supports atomic index persistence and incremental vector addition.
    """

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.dimension = self.embedding_engine.dimension
        # IndexFlatIP = Flat Inner Product index (Exact cosine similarity when vectors are unit-normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        # ID Maps: integer_id -> Chunk Metadata Payload
        self.id_to_chunk_map: Dict[int, Dict[str, Any]] = {}
        # Document Registry: document_hash -> Metadata Manifest Record
        self.doc_registry: Dict[str, Dict[str, Any]] = {}

    def build_index(self, documents: List[Document]) -> None:
        """
        Resets and rebuilds the vector index from a fresh list of documents.
        """
        if not documents:
            logger.warning("No documents provided to build vector index.")
            return

        texts = [doc.page_content for doc in documents]
        logger.info(f"Vectorizing {len(texts)} document chunks...")
        embeddings = self.embedding_engine.embed_documents(texts)

        # Reset existing index and mappings
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_chunk_map = {}
        self.doc_registry = {}

        # Add vectors to FAISS
        self.index.add(embeddings)

        # Map FAISS integer IDs to Document objects
        for idx, doc in enumerate(documents):
            self.id_to_chunk_map[idx] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            file_hash = doc.metadata.get("file_hash")
            if file_hash and file_hash not in self.doc_registry:
                self.doc_registry[file_hash] = {
                    "document_id": doc.metadata.get("document_id"),
                    "file_name": doc.metadata.get("file_name"),
                    "file_type": doc.metadata.get("file_type"),
                }

        logger.info(f"Built FAISS Index with {self.index.ntotal} vectors.")

    def add_documents(self, documents: List[Document]) -> int:
        """
        Appends new vectors and chunk metadata without wiping existing state.
        Returns the count of newly added chunk vectors.
        """
        if not documents:
            logger.warning("No documents supplied for incremental indexing.")
            return 0

        existing_chunk_ids: Set[str] = {
            v["metadata"]["chunk_id"] 
            for v in self.id_to_chunk_map.values() 
            if "metadata" in v and "chunk_id" in v["metadata"]
        }

        new_docs = [d for d in documents if d.metadata.get("chunk_id") not in existing_chunk_ids]

        if not new_docs:
            logger.info("All documents provided are already indexed.")
            return 0

        texts = [doc.page_content for doc in new_docs]
        logger.info(f"Generating embeddings for {len(texts)} new chunk(s)...")
        embeddings = self.embedding_engine.embed_documents(texts)

        start_idx = self.index.ntotal
        self.index.add(embeddings)

        for offset, doc in enumerate(new_docs):
            faiss_id = start_idx + offset
            self.id_to_chunk_map[faiss_id] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            file_hash = doc.metadata.get("file_hash")
            if file_hash and file_hash not in self.doc_registry:
                self.doc_registry[file_hash] = {
                    "document_id": doc.metadata.get("document_id"),
                    "file_name": doc.metadata.get("file_name"),
                    "file_type": doc.metadata.get("file_type"),
                }

        logger.info(f"Added {len(new_docs)} vectors. Total vectors: {self.index.ntotal}")
        return len(new_docs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Performs inner-product dense vector search for a given query.
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty. Search returning empty results.")
            return []

        query_vector = self.embedding_engine.embed_query(query)
        actual_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS padding indicator for empty slots
                continue
            chunk_data = self.id_to_chunk_map.get(int(idx))
            if chunk_data:
                doc = Document(
                    page_content=chunk_data["page_content"],
                    metadata=chunk_data["metadata"]
                )
                results.append((doc, float(score)))

        return results

    def save_index(self, storage_dir: Optional[Path] = None) -> None:
        """
        Persists FAISS index file and chunk mapping JSON file to disk.
        """
        target_dir = Path(storage_dir or settings.BASE_DIR / settings.INDEX_STORAGE_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        mapping_file = target_dir / "chunk_map.json"
        manifest_file = target_dir / "manifest.json"

        # Write C++ FAISS index
        faiss.write_index(self.index, str(index_file))

        # Write metadata mapping
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(self.id_to_chunk_map, f, indent=2)

        # Write manifest registry
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.doc_registry, f, indent=2)

        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {target_dir}")

    def load_index(self, storage_dir: Optional[Path] = None) -> None:
        """
        Loads FAISS index and metadata maps from disk.
        """
        target_dir = Path(storage_dir or settings.BASE_DIR / settings.INDEX_STORAGE_DIR)
        index_file = target_dir / "index.faiss"
        mapping_file = target_dir / "chunk_map.json"
        manifest_file = target_dir / "manifest.json"

        if not index_file.exists() or not mapping_file.exists():
            raise FileNotFoundError(
                f"Index files not found at '{target_dir}'. Run index building first."
            )

        self.index = faiss.read_index(str(index_file))

        with open(mapping_file, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
            self.id_to_chunk_map = {int(k): v for k, v in raw_map.items()}

        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                self.doc_registry = json.load(f)

        logger.info(f"Loaded FAISS Index with {self.index.ntotal} vectors from {target_dir}")