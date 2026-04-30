"""Retriever and reranker module."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from .index_store import SearchResult, VectorIndexStore
from .embedding import BaseEmbeddingBackend


@dataclass
class RetrievalResult:
    """Result of a retrieval operation with optional reranking."""
    results: List[SearchResult]
    original_scores: List[float]
    reranked_scores: Optional[List[float]]
    was_reranked: bool
    query: str


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this reranker."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank search results.

        Args:
            query: The search query.
            results: Original search results.

        Returns:
            Reranked list of SearchResult objects with updated scores.
        """
        pass


class LengthPenaltyReranker(BaseReranker):
    """Reranker that penalizes very short or very long chunks.

    This reranker applies a Gaussian penalty based on chunk length,
    preferring chunks of moderate length that are more likely to
    contain complete, coherent information.
    """

    def __init__(
        self,
        optimal_length: int = 500,
        length_sigma: float = 300.0,
        weight: float = 0.3
    ):
        """Initialize the length penalty reranker.

        Args:
            optimal_length: The ideal chunk length (highest score).
            length_sigma: Controls how quickly scores drop off from optimal.
            weight: Weight of the length penalty vs original score (0-1).
        """
        self.optimal_length = optimal_length
        self.length_sigma = length_sigma
        self.weight = max(0.0, min(1.0, weight))

    @property
    def name(self) -> str:
        return "length_penalty"

    def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using length penalty.

        The new score is a weighted combination of:
        1. Normalized original score (distance converted to similarity)
        2. Length penalty (Gaussian around optimal length)
        """
        if not results:
            return []

        min_score = min(r.score for r in results)
        max_score = max(r.score for r in results)
        score_range = max_score - min_score if max_score != min_score else 1.0

        min_len = min(len(r.content) for r in results)
        max_len = max(len(r.content) for r in results)
        len_range = max_len - min_len if max_len != min_len else 1.0

        reranked = []
        for result in results:
            norm_score = 1.0 - (result.score - min_score) / score_range

            chunk_len = len(result.content)
            norm_len = (chunk_len - min_len) / len_range
            length_penalty = np.exp(
                -0.5 * ((chunk_len - self.optimal_length) / self.length_sigma) ** 2
            )

            combined_score = (
                (1.0 - self.weight) * norm_score +
                self.weight * length_penalty
            )

            new_result = SearchResult(
                chunk_id=result.chunk_id,
                score=combined_score,
                content=result.content,
                source=result.source,
                file_path=result.file_path,
                offset_start=result.offset_start,
                offset_end=result.offset_end,
                index=result.index
            )
            reranked.append(new_result)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


class TfIdfReranker(BaseReranker):
    """Reranker that uses term frequency matching between query and chunks.

    This reranker boosts scores for chunks that contain more query terms,
    especially rare terms.
    """

    def __init__(self, weight: float = 0.4):
        """Initialize the TF-IDF reranker.

        Args:
            weight: Weight of the TF match vs original score (0-1).
        """
        self.weight = max(0.0, min(1.0, weight))

    @property
    def name(self) -> str:
        return "tf_idf"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer that splits on whitespace and punctuation."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    @staticmethod
    def _term_frequency(tokens: List[str], term: str) -> float:
        """Calculate term frequency in a list of tokens."""
        if not tokens:
            return 0.0
        count = tokens.count(term.lower())
        return count / len(tokens)

    def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using query term frequency matching."""
        if not results:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return results

        min_score = min(r.score for r in results)
        max_score = max(r.score for r in results)
        score_range = max_score - min_score if max_score != min_score else 1.0

        all_chunk_tokens = [self._tokenize(r.content) for r in results]

        doc_freq: Dict[str, int] = {}
        for tokens in all_chunk_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        num_docs = len(results)

        reranked = []
        for idx, result in enumerate(results):
            norm_score = 1.0 - (result.score - min_score) / score_range

            chunk_tokens = all_chunk_tokens[idx]
            tf_idf_sum = 0.0

            for query_term in query_tokens:
                tf = self._term_frequency(chunk_tokens, query_term)
                if tf > 0:
                    df = doc_freq.get(query_term, 0)
                    idf = np.log((num_docs + 1) / (df + 1)) + 1
                    tf_idf_sum += tf * idf

            max_possible = len(query_tokens) * np.log(num_docs + 2)
            norm_tf_idf = tf_idf_sum / max_possible if max_possible > 0 else 0.0

            combined_score = (
                (1.0 - self.weight) * norm_score +
                self.weight * norm_tf_idf
            )

            new_result = SearchResult(
                chunk_id=result.chunk_id,
                score=combined_score,
                content=result.content,
                source=result.source,
                file_path=result.file_path,
                offset_start=result.offset_start,
                offset_end=result.offset_end,
                index=result.index
            )
            reranked.append(new_result)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


class Retriever:
    """Main retriever that combines vector search with optional reranking."""

    RERANKERS = {
        "length_penalty": LengthPenaltyReranker,
        "tf_idf": TfIdfReranker,
    }

    def __init__(
        self,
        index_store: VectorIndexStore,
        embedding_backend: BaseEmbeddingBackend,
        reranker: Optional[str] = "tf_idf",
        reranker_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize the retriever.

        Args:
            index_store: The vector index store to search.
            embedding_backend: Backend to embed queries.
            reranker: Name of reranker to use, or None for no reranking.
            reranker_kwargs: Additional kwargs for the reranker constructor.
        """
        self.index_store = index_store
        self.embedding_backend = embedding_backend
        self._reranker: Optional[BaseReranker] = None

        if reranker is not None:
            if reranker not in self.RERANKERS:
                raise ValueError(
                    f"Unknown reranker: {reranker}. "
                    f"Available: {list(self.RERANKERS.keys())}"
                )
            reranker_class = self.RERANKERS[reranker]
            kwargs = reranker_kwargs or {}
            self._reranker = reranker_class(**kwargs)

    @property
    def reranker_name(self) -> Optional[str]:
        """Get the name of the active reranker, if any."""
        return self._reranker.name if self._reranker else None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        enable_rerank: bool = True
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
            enable_rerank: Whether to apply reranking if a reranker is configured.

        Returns:
            RetrievalResult containing the (possibly reranked) results.
        """
        query_vector = self.embedding_backend.embed([query])[0]

        search_results = self.index_store.search(query_vector, top_k)

        original_scores = [r.score for r in search_results]

        if enable_rerank and self._reranker and search_results:
            reranked_results = self._reranker.rerank(query, search_results)
            reranked_scores = [r.score for r in reranked_results]
            return RetrievalResult(
                results=reranked_results,
                original_scores=original_scores,
                reranked_scores=reranked_scores,
                was_reranked=True,
                query=query
            )

        return RetrievalResult(
            results=search_results,
            original_scores=original_scores,
            reranked_scores=None,
            was_reranked=False,
            query=query
        )

    def set_reranker(self, reranker_name: Optional[str], **kwargs):
        """Set or change the reranker.

        Args:
            reranker_name: Name of reranker, or None to disable reranking.
            **kwargs: Additional kwargs for the reranker.
        """
        if reranker_name is None:
            self._reranker = None
        else:
            if reranker_name not in self.RERANKERS:
                raise ValueError(f"Unknown reranker: {reranker_name}")
            reranker_class = self.RERANKERS[reranker_name]
            self._reranker = reranker_class(**kwargs)
