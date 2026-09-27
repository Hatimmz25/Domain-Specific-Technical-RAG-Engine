import time
import logging
from typing import Dict, Any, Optional

from app.config import settings
from app.retrieval.vector_store import FAISSVectorStore
from app.retrieval.retriever import DenseRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.generation.prompts import build_grounded_prompt
from app.generation.llm import LLMProvider, MockLLMProvider, LlamaCppLLMProvider, GroqLLMProvider

logger = logging.getLogger("rag_engine")


class TechnicalRAGEngine:
    """
    Master RAG Orchestrator unifying Dense Retrieval, Cross-Encoder Re-Ranking,
    Prompt Framing, and Grounded LLM Generation.
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        llm_provider: Optional[LLMProvider] = None,
        use_mock_llm: bool = False
    ):
        logger.info("Initializing TechnicalRAGEngine pipeline...")

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = FAISSVectorStore()
            try:
                self.vector_store.load_index()
            except Exception as e:
                logger.warning(f"Vector index not found at startup: {str(e)}")

        self.retriever = DenseRetriever(
            vector_store=self.vector_store,
            top_k=settings.RETRIEVAL_TOP_K,
            similarity_threshold=settings.SIMILARITY_THRESHOLD
        )

        self.reranker = CrossEncoderReranker(
            model_name=settings.RERANKER_MODEL,
            top_k=settings.RERANK_TOP_K
        )

        if llm_provider:
            self.llm = llm_provider
        elif use_mock_llm:
            self.llm = MockLLMProvider()
        elif settings.LLM_PROVIDER == "groq":
            self.llm = GroqLLMProvider()
        else:
            self.llm = LlamaCppLLMProvider()

        logger.info("TechnicalRAGEngine pipeline initialization complete.")

    def ask(self, query: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        start_total = time.perf_counter()

        if not query or not query.strip():
            return {
                "question": query,
                "answer": "Query string cannot be empty.",
                "sources": [],
                "retrieved_context": [],
                "latency_ms": {"retrieval": 0.0, "reranking": 0.0, "generation": 0.0, "total": 0.0},
                "retrieved_candidate_count": 0,
                "selected_chunk_count": 0
            }

        # Stage 1: Dense Retrieval
        start_retrieval = time.perf_counter()
        candidate_pairs = self.retriever.get_relevant_documents_with_scores(query)
        candidate_docs = [doc for doc, _ in candidate_pairs]
        time_retrieval = (time.perf_counter() - start_retrieval) * 1000

        # Stage 2: Cross-Encoder Re-Ranking
        start_rerank = time.perf_counter()
        if candidate_docs:
            raw_reranked = self.reranker.rerank(query=query, candidates=candidate_docs)
            reranked_chunks = [item for item in raw_reranked if item.get("score", 0.0) >= 0.1]
        else:
            reranked_chunks = []
        time_rerank = (time.perf_counter() - start_rerank) * 1000

        # Stage 3: Prompt Framing
        prompt = build_grounded_prompt(query=query, context_chunks=reranked_chunks)

        # Stage 4: LLM Generation
        start_gen = time.perf_counter()
        if not reranked_chunks:
            answer = "I couldn't find sufficient information in the indexed documentation to answer this question."
        else:
            try:
                answer = self.llm.generate(
                    prompt=prompt,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    temperature=settings.LLM_TEMPERATURE
                )
            except Exception as err:
                logger.error(f"LLM Generation Exception: {err}", extra={"request_id": request_id})
                answer = "An internal error occurred during model generation."

            if not answer or not answer.strip():
                answer = "I couldn't find sufficient information in the indexed documentation to answer this question."

        time_gen = (time.perf_counter() - start_gen) * 1000

        # Stage 5: Source Citation Assembly
        sources = []
        for chunk_data in reranked_chunks:
            meta = chunk_data.get("metadata", {})
            sources.append({
                "source": meta.get("file_name", "Unknown"),
                "file_name": meta.get("file_name", "Unknown"),
                "chunk_id": meta.get("chunk_id", "Unknown"),
                "score": round(chunk_data.get("score", 0.0), 4)
            })

        time_total = (time.perf_counter() - start_total) * 1000

        return {
            "question": query,
            "answer": answer,
            "sources": sources,
            "retrieved_context": [
                {
                    "chunk_id": item["metadata"].get("chunk_id"),
                    "content": item["document"].page_content,
                    "score": round(item["score"], 4)
                }
                for item in reranked_chunks
            ],
            "latency_ms": {
                "retrieval": round(time_retrieval, 2),
                "reranking": round(time_rerank, 2),
                "generation": round(time_gen, 2),
                "total": round(time_total, 2)
            },
            "retrieved_candidate_count": len(candidate_docs),
            "selected_chunk_count": len(reranked_chunks)
        }