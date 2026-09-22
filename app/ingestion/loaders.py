import hashlib
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field

from pypdf import PdfReader


@dataclass
class Document:
    """
    Standardized internal document object across the RAG engine.
    """
    page_content: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """
    Unified loader for technical documentation files (PDF, Markdown, TXT).
    Applies automatic metadata tagging and text normalization.
    """

    def __init__(self, cleaner=None):
        from app.ingestion.cleaner import TextCleaner
        self.cleaner = cleaner or TextCleaner()

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Loads a single document file and returns a list of Document objects.

        Args:
            file_path (str | Path): Path to the input target file.

        Returns:
            List[Document]: Extracted document segments with rich metadata.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Target document not found: {path}")

        file_extension = path.suffix.lower()

        if file_extension in [".md", ".markdown"]:
            return self._load_text_based(path, file_type="markdown")
        elif file_extension == ".txt":
            return self._load_text_based(path, file_type="txt")
        elif file_extension == ".pdf":
            return self._load_pdf(path)
        else:
            raise ValueError(f"Unsupported file format '{file_extension}' for path: {path}")

    def load_directory(self, dir_path: Union[str, Path]) -> List[Document]:
        """
        Recursively loads all supported technical documents (.md, .txt, .pdf) from a directory.
        """
        directory = Path(dir_path).resolve()
        if not directory.exists() or not directory.is_dir():
            raise NotADirectoryError(f"Target directory does not exist: {directory}")

        documents = []
        supported_extensions = ["*.md", "*.markdown", "*.txt", "*.pdf"]

        for ext in supported_extensions:
            for file_path in directory.rglob(ext):
                try:
                    loaded_docs = self.load_file(file_path)
                    documents.extend(loaded_docs)
                except Exception as e:
                    print(f"[WARNING] Failed to load {file_path}: {str(e)}")

        return documents

    def _load_text_based(self, path: Path, file_type: str) -> List[Document]:
        """Loads and processes plain text or Markdown files."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        cleaned_text = self.cleaner.clean(raw_text)
        if not cleaned_text:
            return []

        doc_id = self._generate_doc_id(path, cleaned_text)
        metadata = {
            "source": str(path),
            "file_name": path.name,
            "file_type": file_type,
            "document_id": doc_id,
            "char_length": len(cleaned_text),
        }

        return [Document(page_content=cleaned_text, metadata=metadata)]

    def _load_pdf(self, path: Path) -> List[Document]:
        """Loads and extracts page-by-page content from PDF documents."""
        reader = PdfReader(path)
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned_text = self.cleaner.clean(raw_text)

            if not cleaned_text:
                continue

            doc_id = self._generate_doc_id(path, f"page_{page_num}_{cleaned_text[:50]}")
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "file_type": "pdf",
                "page": page_num,
                "total_pages": len(reader.pages),
                "document_id": doc_id,
                "char_length": len(cleaned_text),
            }

            documents.append(Document(page_content=cleaned_text, metadata=metadata))

        return documents

    @staticmethod
    def _generate_doc_id(path: Path, content_sample: str) -> str:
        """Generates a deterministic hash ID for tracking document identity."""
        hasher = hashlib.sha256()
        hasher.update(str(path.name).encode("utf-8"))
        hasher.update(content_sample[:100].encode("utf-8"))
        return f"doc_{hasher.hexdigest()[:12]}"