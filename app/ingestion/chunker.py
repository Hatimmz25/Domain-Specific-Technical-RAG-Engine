import re
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

from app.ingestion.loaders import Document
from app.config import settings


class RecursiveChunker:
    """
    Fallback hierarchical text chunker that splits on natural structural separators
    (paragraphs, line breaks, sentences, words) while respecting max chunk boundaries.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Splits text recursively based on ordered separators."""
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find the first valid separator
        separator = separators[-1]
        for sep in separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break

        # Split text using chosen separator
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                other_seps = separators[separators.index(separator) + 1:] if separator in separators else []
                if other_seps:
                    final_chunks.extend(self._recursive_split(s, other_seps))
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merges small splits into chunks up to self.chunk_size with overlap."""
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            len_split = len(split) + (len(separator) if current_chunk else 0)
            if current_length + len_split > self.chunk_size and current_chunk:
                chunk_str = separator.join(current_chunk).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                # Keep overlap elements
                overlap_size = 0
                new_current = []
                for item in reversed(current_chunk):
                    if overlap_size + len(item) <= self.chunk_overlap:
                        new_current.insert(0, item)
                        overlap_size += len(item)
                    else:
                        break
                current_chunk = new_current
                current_length = sum(len(x) for x in current_chunk) + len(separator) * max(0, len(current_chunk) - 1)

            current_chunk.append(split)
            current_length += len_split

        if current_chunk:
            final_str = separator.join(current_chunk).strip()
            if final_str:
                chunks.append(final_str)

        return chunks


class SemanticChunker:
    """
    Splits text semantically based on sentence embedding distance shifts.
    Uses RecursiveChunker as a fallback if document is too short or embedding model fails.
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        max_chunk_size: int = 600,
        breakpoint_percentile_threshold: int = 95,
        buffer_size: int = 1
    ):
        self.max_chunk_size = max_chunk_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.buffer_size = buffer_size
        self.fallback_chunker = RecursiveChunker(chunk_size=max_chunk_size, chunk_overlap=50)
        self.embedding_model_name = embedding_model_name or settings.EMBEDDING_MODEL
        self._embedder = None

    @property
    def embedder(self):
        """Lazy loading wrapper for HuggingFace SentenceTransformer."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of Documents into semantically cohesive chunk Documents.
        Inherits and extends parent metadata.
        """
        chunked_docs = []
        for doc in documents:
            chunks = self._chunk_single_document(doc.page_content)
            total_chunks = len(chunks)

            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc.metadata.get('document_id', 'doc')}_chunk_{idx:03d}"
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "chunk_char_length": len(chunk_text)
                })
                chunked_docs.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return chunked_docs

    def _chunk_single_document(self, text: str) -> List[str]:
        """Processes a single text body into semantic chunks."""
        # Clean & split into candidate sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 2:
            return self.fallback_chunker.split_text(text)

        # Create combined sentence window buffers for local contextual comparison
        combined_sentences = self._combine_sentences(sentences, buffer_size=self.buffer_size)

        try:
            # Generate sentence window embeddings
            embeddings = self.embedder.encode(combined_sentences, show_progress_bar=False)
            
            # Compute cosine distances between adjacent sentence windows
            distances = []
            for i in range(len(embeddings) - 1):
                vec1 = embeddings[i]
                vec2 = embeddings[i + 1]
                # Cosine distance = 1 - cosine_similarity
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 == 0 or norm2 == 0:
                    sim = 0.0
                else:
                    sim = np.dot(vec1, vec2) / (norm1 * norm2)
                distances.append(1.0 - float(sim))

            if not distances:
                return self.fallback_chunker.split_text(text)

            # Determine distance cutoff threshold via percentile
            breakpoint_distance_threshold = np.percentile(distances, self.breakpoint_percentile_threshold)
            
            # Identify indices where distance exceeds threshold
            indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

            # Assemble chunks bounded by distance thresholds
            chunks = []
            start_index = 0

            for index in indices_above_thresh:
                # Group sentences up to split point
                group = sentences[start_index : index + 1]
                chunk_txt = " ".join(group).strip()
                
                # Check max size limit; if exceeded, apply recursive split fallback
                if len(chunk_txt) > self.max_chunk_size:
                    chunks.extend(self.fallback_chunker.split_text(chunk_txt))
                elif chunk_txt:
                    chunks.append(chunk_txt)
                
                start_index = index + 1

            # Append trailing sentences
            if start_index < len(sentences):
                group = sentences[start_index:]
                chunk_txt = " ".join(group).strip()
                if len(chunk_txt) > self.max_chunk_size:
                    chunks.extend(self.fallback_chunker.split_text(chunk_txt))
                elif chunk_txt:
                    chunks.append(chunk_txt)

            return chunks if chunks else self.fallback_chunker.split_text(text)

        except Exception as e:
            # If semantic processing fails for any reason, safely execute fallback chunking
            print(f"[WARNING] Semantic chunking failed ({str(e)}). Executing Recursive fallback.")
            return self.fallback_chunker.split_text(text)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits raw text into candidate sentences preserving punctuation boundary."""
        # Simple regex splitting on standard sentence terminators (. ! ?) followed by whitespace
        sentence_end = re.compile(r"(?<=[.!?])\s+")
        raw_sentences = sentence_end.split(text)
        return [s.strip() for s in raw_sentences if s and s.strip()]

    def _combine_sentences(self, sentences: List[str], buffer_size: int = 1) -> List[str]:
        """Combines neighboring sentences into windowed groups for robust embedding distance comparison."""
        combined = []
        for i in range(len(sentences)):
            start = max(0, i - buffer_size)
            end = min(len(sentences), i + buffer_size + 1)
            combined.append(" ".join(sentences[start:end]))
        return combined