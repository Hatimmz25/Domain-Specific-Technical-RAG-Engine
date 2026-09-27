import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.loaders import DocumentLoader


def create_sample_docs(raw_dir: Path):
    """Generates sample technical documentation files for testing ingestion."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    fastapi_sample = raw_dir / "fastapi_tutorial.md"
    docker_sample = raw_dir / "docker_guide.txt"

    if not fastapi_sample.exists():
        fastapi_sample.write_text(
            "# FastAPI Tutorial\n\n"
            "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+.\n\n"
            "## Creating a POST Endpoint\n\n"
            "To create a POST endpoint in FastAPI, use the `@app.post()` decorator:\n\n"
            "```python\n"
            "from fastapi import FastAPI\n"
            "from pydantic import BaseModel\n\n"
            "app = FastAPI()\n\n"
            "class Item(BaseModel):\n"
            "    name: str\n"
            "    price: float\n\n"
            "@app.post('/items/')\n"
            "def create_item(item: Item):\n"
            "    return {'item_name': item.name, 'item_price': item.price}\n"
            "```\n",
            encoding="utf-8"
        )

    if not docker_sample.exists():
        docker_sample.write_text(
            "Docker Commands and Instructions\n\n"
            "COPY vs ADD in Dockerfile:\n"
            "1. COPY: Copies local files or directories into the container image filesystem.\n"
            "2. ADD: Does everything COPY does, but also supports fetching files from remote URLs "
            "and automatically extracting .tar archives into the destination.\n\n"
            "Best Practice: Prefer COPY over ADD for transparency unless automatic tar extraction is required.\n",
            encoding="utf-8"
        )


def run_ingestion():
    """CLI script to test loading technical documentation from data/raw."""
    raw_dir = settings.BASE_DIR / settings.DATA_RAW_DIR
    print("=" * 60)
    print(f"      Running Document Ingestion Pipeline on '{raw_dir}'")
    print("=" * 60)

    # Seed sample documentation if directory is empty
    create_sample_docs(raw_dir)

    loader = DocumentLoader()
    documents = loader.load_directory(raw_dir)

    print(f"\n[SUCCESS] Total Documents Ingested: {len(documents)}\n")
    for idx, doc in enumerate(documents, start=1):
        print(f"--- Document #{idx} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"File Type: {doc.metadata.get('file_type')}")
        print(f"Document ID: {doc.metadata.get('document_id')}")
        print(f"Character Length: {doc.metadata.get('char_length')}")
        print(f"Content Preview:\n{doc.page_content[:150]}...\n")


if __name__ == "__main__":
    run_ingestion()