from typing import List, Tuple, Dict, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document as LCDocument
from pydantic import Field, ConfigDict

from app.config import settings
from app.ingestion.loaders import Document
from app.retrieval.vector_store import FAISSVectorStore


class DenseRetriever(BaseRetriever):
    """
    Production candidate generation retriever implementing LangChain's standard BaseRetriever interface.
    Retrieves Top-N candidate documents using FAISS inner-product vector search with similarity filtering.
    """

    vector_store: FAISSVectorStore = Field(default=None, description="FAISS vector store instance")
    top_k: int = Field(default_factory=lambda: settings.RETRIEVAL_TOP_K, description="Candidate count (N)")
    similarity_threshold: float = Field(
        default_factory=lambda: settings.SIMILARITY_THRESHOLD,
        description="Minimum score threshold cutoff"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[LCDocument]:
        results_with_scores = self.get_relevant_documents_with_scores(query)
        lc_docs = []

        for doc, score in results_with_scores:
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata["similarity_score"] = float(score)
            
            lc_docs.append(
                LCDocument(
                    page_content=doc.page_content,
                    metadata=metadata
                )
            )

        return lc_docs

    def get_relevant_documents_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            raise ValueError("FAISSVectorStore is not attached to DenseRetriever.")

        raw_results = self.vector_store.search(query=query, top_k=self.top_k)

        filtered_results = []
        for doc, score in raw_results:
            if score >= self.similarity_threshold:
                filtered_results.append((doc, score))

        return filtered_results