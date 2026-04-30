"""Generator module with pluggable backends for RAG output."""

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import warnings

from .retriever import RetrievalResult
from .index_store import SearchResult


@dataclass
class GenerationResult:
    """Result of a generation operation."""
    answer: str
    cites: List[Dict[str, Any]]
    backend: str
    context_used: str
    context_truncated: bool = False


class BaseGenerationBackend(ABC):
    """Abstract base class for generation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend."""
        pass

    @abstractmethod
    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        max_context_chars: int = 8000
    ) -> GenerationResult:
        """Generate an answer based on query and retrieved context.

        Args:
            query: The user's question.
            retrieval_result: The retrieval results containing relevant chunks.
            max_context_chars: Maximum characters to include in context.

        Returns:
            GenerationResult containing the answer and citations.
        """
        pass

    @staticmethod
    def _format_citation(result: SearchResult, index: int) -> Dict[str, Any]:
        """Format a search result as a citation."""
        return {
            "chunk_id": result.chunk_id,
            "source": result.source,
            "file_path": result.file_path,
            "offset_start": result.offset_start,
            "offset_end": result.offset_end,
            "reference_index": index + 1
        }

    @staticmethod
    def _build_context(
        results: List[SearchResult],
        max_context_chars: int
    ) -> tuple:
        """Build context string from search results, with truncation handling.

        Returns:
            Tuple of (context_string, citations_list, is_truncated)
        """
        context_parts = []
        citations = []
        total_chars = 0
        is_truncated = False

        for idx, result in enumerate(results):
            citation = BaseGenerationBackend._format_citation(result, idx)
            citations.append(citation)

            chunk_header = f"\n--- Reference [{idx + 1}] ---\n"
            chunk_header += f"Source: {result.source}\n"
            chunk_header += f"Content: {result.content}\n"

            if total_chars + len(chunk_header) > max_context_chars:
                available = max_context_chars - total_chars
                if available > 0:
                    truncated_content = result.content[:available]
                    truncated_header = f"\n--- Reference [{idx + 1}] (truncated) ---\n"
                    truncated_header += f"Source: {result.source}\n"
                    truncated_header += f"Content: {truncated_content}\n"
                    context_parts.append(truncated_header)
                    total_chars += len(truncated_header)
                is_truncated = True
                break

            context_parts.append(chunk_header)
            total_chars += len(chunk_header)

        context_string = "".join(context_parts)
        return context_string, citations, is_truncated


class StubGenerationBackend(BaseGenerationBackend):
    """Stub generation backend that produces deterministic, testable output.

    This backend generates predictable answers based on the query and
    retrieved context. It always includes citations in a parsable format.
    """

    @property
    def name(self) -> str:
        return "stub"

    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        max_context_chars: int = 8000
    ) -> GenerationResult:
        """Generate a stub answer with citations.

        The answer format is deterministic and includes:
        1. A reference to the query
        2. A summary of how many chunks were retrieved
        3. Citations to all retrieved chunks
        """
        context, citations, is_truncated = self._build_context(
            retrieval_result.results,
            max_context_chars
        )

        if is_truncated:
            warnings.warn(
                "Context was truncated due to length limit. "
                f"Max chars: {max_context_chars}, used: {len(context)}",
                UserWarning
            )
            print(
                "WARNING: Context truncated - some references may be incomplete",
                file=sys.stderr
            )

        num_results = len(retrieval_result.results)

        if num_results == 0:
            answer = (
                f"No relevant information found for query: '{query}'. "
                "Please try a different question or add more documents to the index."
            )
        else:
            answer_parts = [
                f"Query: {query}",
                f"Found {num_results} relevant reference(s):",
            ]

            for idx, result in enumerate(retrieval_result.results):
                preview = result.content[:100]
                if len(result.content) > 100:
                    preview += "..."
                answer_parts.append(
                    f"  [{idx + 1}] {result.source}: {preview}"
                )

            answer_parts.append(
                "\nBased on the above references, here is a summary answer."
            )
            answer = "\n".join(answer_parts)

        return GenerationResult(
            answer=answer,
            cites=citations,
            backend=self.name,
            context_used=context,
            context_truncated=is_truncated
        )


class OpenAIGenerationBackend(BaseGenerationBackend):
    """OpenAI generation backend using the Chat Completions API."""

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT: 
1. Only use information from the provided context to answer the question.
2. If the answer is not in the context, say "I cannot find the answer in the provided documents."
3. Cite your sources using [1], [2], etc. where each number corresponds to a reference.
4. Be concise but thorough.

The context will be provided with references marked as [1], [2], etc."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize the OpenAI generation backend.

        Args:
            model: Name of the OpenAI chat model.
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            system_prompt: Custom system prompt. If None, uses default.

        Raises:
            ImportError: If openai package is not installed.
            ValueError: If API key is not provided.
        """
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        if not self._api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai>=1.0"
            )

    @property
    def name(self) -> str:
        return "openai"

    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        max_context_chars: int = 8000
    ) -> GenerationResult:
        """Generate an answer using OpenAI API."""
        context, citations, is_truncated = self._build_context(
            retrieval_result.results,
            max_context_chars
        )

        if is_truncated:
            warnings.warn(
                "Context was truncated due to length limit. "
                f"Max chars: {max_context_chars}, used: {len(context)}",
                UserWarning
            )
            print(
                "WARNING: Context truncated - some references may be incomplete",
                file=sys.stderr
            )

        if not context or not citations:
            user_prompt = (
                f"Question: {query}\n\n"
                "No relevant documents were found. Please inform the user."
            )
        else:
            user_prompt = (
                f"Question: {query}\n\n"
                f"Context:\n{context}\n\n"
                "Please answer the question based only on the context provided. "
                "Cite your sources using [1], [2], etc. format."
            )

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0
        )

        answer = response.choices[0].message.content or ""

        return GenerationResult(
            answer=answer,
            cites=citations,
            backend=self.name,
            context_used=context,
            context_truncated=is_truncated
        )


def get_generation_backend(backend: str = "stub", **kwargs) -> BaseGenerationBackend:
    """Get a generation backend instance.

    Args:
        backend: Name of the backend to use ('stub' or 'openai').
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An instance of BaseGenerationBackend.

    Raises:
        ValueError: If the backend is unknown.
    """
    if backend == "stub":
        return StubGenerationBackend(**kwargs)
    elif backend == "openai":
        return OpenAIGenerationBackend(**kwargs)
    else:
        raise ValueError(f"Unknown generation backend: {backend}")


def get_generation_backend_from_config(config: Dict[str, Any]) -> BaseGenerationBackend:
    """Get generation backend from configuration dictionary.

    Args:
        config: Configuration dict with 'backend' key and optional parameters.

    Returns:
        An instance of BaseGenerationBackend.
    """
    backend = config.get("backend", "stub")
    params = config.get("params", {})
    return get_generation_backend(backend, **params)
