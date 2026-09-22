from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.config import settings


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        pass


class LlamaCppLLMProvider(LLMProvider):
    def __init__(
        self,
        filename: Optional[str] = None,
        context_window: int = 4096
    ):
        self.filename = filename or settings.LLM_MODEL_FILE
        self.context_window = context_window or settings.LLM_CONTEXT_WINDOW
        self.model_path = settings.BASE_DIR / "models" / self.filename
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file '{self.filename}' was not found at {self.model_path}. "
                    f"Please verify your file placement in the models/ directory."
                )

            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is required for local quantized GGUF inference."
                )

            print(f"[INFO] Loading Llama C++ GGUF Model from {self.model_path}...")
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_window,
                n_threads=4,
                verbose=False
            )
            print("[SUCCESS] Loaded GGUF LLM Model successfully.")
        return self._llm

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>", "### User:", "<|im_start|>"],
            echo=False
        )
        return response["choices"][0]["text"].strip()


class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        return (
            "In LangChain, a Retriever is an interface that returns documents given an "
            "unstructured query. It is more general than a vector store and does not need "
            "to store documents, only return them."
        )