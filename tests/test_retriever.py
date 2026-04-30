"""Tests for the retriever and reranker module."""

import numpy as np
import pytest
import tempfile
import os

from llm_rag_pipeline.retriever import (
    LengthPenaltyReranker,
    TfIdfReranker,
    Retriever,
    RetrievalResult
)
from llm_rag_pipeline.index_store import (
    VectorIndexStore,
    ChunkMetadata,
    SearchResult
)
from llm_rag_pipeline.embedding import StubEmbeddingBackend


class TestLengthPenaltyReranker:
    """Tests for LengthPenaltyReranker class."""

    def test_name_property(self):
        """Test that name property returns correct value."""
        reranker = LengthPenaltyReranker()
        assert reranker.name == "length_penalty"

    def test_rerank_empty_results(self):
        """Test reranking empty results."""
        reranker = LengthPenaltyReranker()
        results = reranker.rerank("test query", [])
        assert results == []

    def test_rerank_single_result(self):
        """Test reranking single result."""
        reranker = LengthPenaltyReranker()
        result = SearchResult(
            chunk_id="test:0:10",
            score=0.5,
            content="short text",
            source="test.txt",
            file_path="/tmp/test.txt",
            offset_start=0,
            offset_end=10,
            index=0
        )
        results = reranker.rerank("test query", [result])
        assert len(results) == 1


class TestTfIdfReranker:
    """Tests for TfIdfReranker class."""

    def test_name_property(self):
        """Test that name property returns correct value."""
        reranker = TfIdfReranker()
        assert reranker.name == "tf_idf"

    def test_rerank_empty_results(self):
        """Test reranking empty results."""
        reranker = TfIdfReranker()
        results = reranker.rerank("test query", [])
        assert results == []

    def test_rerank_with_query_terms(self):
        """Test that chunks containing query terms get higher scores."""
        reranker = TfIdfReranker(weight=1.0)

        result_with_term = SearchResult(
            chunk_id="test:0:50",
            score=0.5,
            content="This chunk contains the important keyword apple banana cherry.",
            source="test.txt",
            file_path="/tmp/test.txt",
            offset_start=0,
            offset_end=50,
            index=0
        )

        result_without_term = SearchResult(
            chunk_id="test:50:100",
            score=0.3,
            content="This chunk has no relevant terms about fruit.",
            source="test.txt",
            file_path="/tmp/test.txt",
            offset_start=50,
            offset_end=100,
            index=1
        )

        results = reranker.rerank(
            "apple banana cherry",
            [result_without_term, result_with_term]
        )

        assert len(results) == 2
        assert results[0].chunk_id == "test:0:50"


class TestRerankerSwitchEffect:
    """Tests that reranker toggle actually changes the results.

    This test verifies that enabling/disabling reranking produces
    different orderings or scores.
    """

    def test_rerank_changes_ordering(self):
        """Test that reranking can change the order of results."""
        embedding_backend = StubEmbeddingBackend(dimension=128)

        query = "apple banana"

        chunk1_content = "apple banana cherry date"
        chunk2_content = "different content without query terms"

        chunk1 = ChunkMetadata(
            chunk_id="doc:0:20",
            content=chunk1_content,
            source="doc.txt",
            file_path="/tmp/doc.txt",
            offset_start=0,
            offset_end=20,
            index=0
        )
        chunk2 = ChunkMetadata(
            chunk_id="doc:20:60",
            content=chunk2_content,
            source="doc.txt",
            file_path="/tmp/doc.txt",
            offset_start=20,
            offset_end=60,
            index=1
        )

        vec1 = embedding_backend.embed([chunk1_content])
        vec2 = embedding_backend.embed([chunk2_content])

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorIndexStore(tmpdir, dimension=128)
            store.create("stub")
            store.add_vectors(
                np.vstack([vec1, vec2]),
                [chunk1, chunk2],
                on_conflict="skip"
            )
            store.save()

            store2 = VectorIndexStore(tmpdir, dimension=128)
            store2.load()

            retriever = Retriever(
                index_store=store2,
                embedding_backend=embedding_backend,
                reranker="tf_idf"
            )

            result_no_rerank = retriever.retrieve(
                query,
                top_k=10,
                enable_rerank=False
            )

            result_with_rerank = retriever.retrieve(
                query,
                top_k=10,
                enable_rerank=True
            )

            assert result_no_rerank.was_reranked is False
            assert result_with_rerank.was_reranked is True

            if (len(result_no_rerank.results) >= 2 and
                len(result_with_rerank.results) >= 2):
                scores_no = result_no_rerank.original_scores
                scores_with = result_with_rerank.reranked_scores

                if scores_with is not None and len(scores_with) >= 2:
                    assert scores_no != scores_with or (
                        result_no_rerank.results[0].chunk_id !=
                        result_with_rerank.results[0].chunk_id
                    )


class TestRetriever:
    """Tests for Retriever class."""

    def test_init_with_reranker(self):
        """Test initializing retriever with a reranker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorIndexStore(tmpdir, dimension=128)
            store.create("stub")
            embedding = StubEmbeddingBackend(dimension=128)

            retriever = Retriever(
                index_store=store,
                embedding_backend=embedding,
                reranker="tf_idf"
            )
            assert retriever.reranker_name == "tf_idf"

    def test_init_without_reranker(self):
        """Test initializing retriever without a reranker."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorIndexStore(tmpdir, dimension=128)
            store.create("stub")
            embedding = StubEmbeddingBackend(dimension=128)

            retriever = Retriever(
                index_store=store,
                embedding_backend=embedding,
                reranker=None
            )
            assert retriever.reranker_name is None

    def test_init_with_unknown_reranker(self):
        """Test initializing with unknown reranker raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorIndexStore(tmpdir, dimension=128)
            store.create("stub")
            embedding = StubEmbeddingBackend(dimension=128)

            with pytest.raises(ValueError, match="Unknown reranker"):
                Retriever(
                    index_store=store,
                    embedding_backend=embedding,
                    reranker="unknown_reranker"
                )

    def test_retrieve_empty_index(self):
        """Test retrieving from empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorIndexStore(tmpdir, dimension=128)
            store.create("stub")
            embedding = StubEmbeddingBackend(dimension=128)

            retriever = Retriever(
                index_store=store,
                embedding_backend=embedding,
                reranker=None
            )

            result = retriever.retrieve("test query", top_k=5)
            assert len(result.results) == 0
            assert result.was_reranked is False

    def test_set_reranker(self):
        """Test changing reranker after initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorIndexStore(tmpdir, dimension=128)
            store.create("stub")
            embedding = StubEmbeddingBackend(dimension=128)

            retriever = Retriever(
                index_store=store,
                embedding_backend=embedding,
                reranker=None
            )
            assert retriever.reranker_name is None

            retriever.set_reranker("tf_idf")
            assert retriever.reranker_name == "tf_idf"

            retriever.set_reranker("length_penalty")
            assert retriever.reranker_name == "length_penalty"

            retriever.set_reranker(None)
            assert retriever.reranker_name is None

    def test_set_reranker_invalid(self):
        """Test setting invalid reranker raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorIndexStore(tmpdir, dimension=128)
            store.create("stub")
            embedding = StubEmbeddingBackend(dimension=128)

            retriever = Retriever(
                index_store=store,
                embedding_backend=embedding,
                reranker=None
            )

            with pytest.raises(ValueError, match="Unknown reranker"):
                retriever.set_reranker("invalid")
