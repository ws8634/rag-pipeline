"""Tests for the generator module."""

import numpy as np
import pytest
import warnings
import io
import sys

from llm_rag_pipeline.generator import (
    StubGenerationBackend,
    get_generation_backend,
    get_generation_backend_from_config,
    GenerationResult
)
from llm_rag_pipeline.retriever import RetrievalResult
from llm_rag_pipeline.index_store import SearchResult


class TestStubGenerationBackend:
    """Tests for StubGenerationBackend class."""

    def test_name_property(self):
        """Test that name property returns 'stub'."""
        backend = StubGenerationBackend()
        assert backend.name == "stub"

    def test_generate_no_results(self):
        """Test generating with no retrieval results."""
        backend = StubGenerationBackend()

        retrieval_result = RetrievalResult(
            results=[],
            original_scores=[],
            reranked_scores=None,
            was_reranked=False,
            query="test question"
        )

        gen_result = backend.generate(
            "test question",
            retrieval_result
        )

        assert isinstance(gen_result, GenerationResult)
        assert gen_result.backend == "stub"
        assert gen_result.cites == []
        assert "test question" in gen_result.answer
        assert gen_result.context_truncated is False

    def test_generate_with_results(self):
        """Test generating with retrieval results."""
        backend = StubGenerationBackend()

        search_results = [
            SearchResult(
                chunk_id="doc1:0:100",
                score=0.1,
                content="This is the first chunk of content about RAG systems.",
                source="doc1.txt",
                file_path="/tmp/doc1.txt",
                offset_start=0,
                offset_end=100,
                index=0
            ),
            SearchResult(
                chunk_id="doc2:50:150",
                score=0.2,
                content="This is the second chunk with more information about embeddings.",
                source="doc2.txt",
                file_path="/tmp/doc2.txt",
                offset_start=50,
                offset_end=150,
                index=1
            )
        ]

        retrieval_result = RetrievalResult(
            results=search_results,
            original_scores=[0.1, 0.2],
            reranked_scores=None,
            was_reranked=False,
            query="What is RAG?"
        )

        gen_result = backend.generate(
            "What is RAG?",
            retrieval_result
        )

        assert isinstance(gen_result, GenerationResult)
        assert gen_result.backend == "stub"
        assert len(gen_result.cites) == 2
        assert "What is RAG?" in gen_result.answer
        assert "doc1.txt" in gen_result.answer or "doc2.txt" in gen_result.answer

    def test_citations_include_chunk_ids(self):
        """Test that citations include chunk IDs and metadata."""
        backend = StubGenerationBackend()

        search_results = [
            SearchResult(
                chunk_id="test:0:50",
                score=0.1,
                content="Content here",
                source="test.txt",
                file_path="/tmp/test.txt",
                offset_start=0,
                offset_end=50,
                index=0
            )
        ]

        retrieval_result = RetrievalResult(
            results=search_results,
            original_scores=[0.1],
            reranked_scores=None,
            was_reranked=False,
            query="test"
        )

        gen_result = backend.generate("test", retrieval_result)

        assert len(gen_result.cites) == 1
        citation = gen_result.cites[0]
        assert citation["chunk_id"] == "test:0:50"
        assert citation["source"] == "test.txt"
        assert citation["file_path"] == "/tmp/test.txt"
        assert citation["offset_start"] == 0
        assert citation["offset_end"] == 50
        assert "reference_index" in citation

    def test_context_truncation_warning(self):
        """Test that context truncation generates a warning."""
        backend = StubGenerationBackend()

        long_content = "x" * 1000

        search_results = [
            SearchResult(
                chunk_id=f"doc{i}:0:1000",
                score=0.1 + i * 0.01,
                content=long_content,
                source=f"doc{i}.txt",
                file_path=f"/tmp/doc{i}.txt",
                offset_start=0,
                offset_end=1000,
                index=i
            )
            for i in range(10)
        ]

        retrieval_result = RetrievalResult(
            results=search_results,
            original_scores=[r.score for r in search_results],
            reranked_scores=None,
            was_reranked=False,
            query="test"
        )

        captured_stderr = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured_stderr

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gen_result = backend.generate(
                    "test",
                    retrieval_result,
                    max_context_chars=500
                )

                if gen_result.context_truncated:
                    assert len(w) >= 1
                    assert any("truncated" in str(warning.message).lower() for warning in w)

                    stderr_output = captured_stderr.getvalue()
                    assert "truncated" in stderr_output.lower()
        finally:
            sys.stderr = old_stderr


class TestGetGenerationBackend:
    """Tests for get_generation_backend function."""

    def test_get_stub_backend(self):
        """Test getting stub backend."""
        backend = get_generation_backend("stub")
        assert isinstance(backend, StubGenerationBackend)

    def test_get_unknown_backend(self):
        """Test getting an unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown generation backend"):
            get_generation_backend("unknown_backend")


class TestGetGenerationBackendFromConfig:
    """Tests for get_generation_backend_from_config function."""

    def test_stub_from_config(self):
        """Test creating stub backend from config."""
        config = {
            "backend": "stub",
            "params": {}
        }
        backend = get_generation_backend_from_config(config)
        assert isinstance(backend, StubGenerationBackend)

    def test_default_backend(self):
        """Test that default backend is stub."""
        config = {}
        backend = get_generation_backend_from_config(config)
        assert isinstance(backend, StubGenerationBackend)
