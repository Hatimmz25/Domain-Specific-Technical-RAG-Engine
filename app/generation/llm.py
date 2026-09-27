import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.config import settings


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        pass


class GroqLLMProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.model = model or settings.GROQ_MODEL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable is not configured.")
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("groq package is required for Groq Cloud API inference.")
        return self._client

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()


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
                    f"Model file '{self.filename}' was not found at {self.model_path}."
                )

            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError(
                    "llama-cpp-python is required for local quantized GGUF inference."
                )

            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_window,
                n_threads=4,
                verbose=False
            )
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
    """
    Mock LLM provider for rapid pipeline integration testing and CI without loading heavy model weights.
    """
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        return (
            "To create a POST endpoint in FastAPI, define a route using the `@app.post()` decorator "
            "and specify a Pydantic model class to validate the incoming request body payload."
        )
