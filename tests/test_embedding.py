"""Tests for the embedding module."""

import numpy as np
import pytest

from llm_rag_pipeline.embedding import (
    StubEmbeddingBackend,
    get_embedding_backend,
    get_embedding_backend_from_config
)


class TestStubEmbeddingBackend:
    """Tests for StubEmbeddingBackend class."""

    def test_name_property(self):
        """Test that name property returns 'stub'."""
        backend = StubEmbeddingBackend(dimension=128)
        assert backend.name == "stub"

    def test_dimension_property(self):
        """Test that dimension property returns the correct value."""
        backend = StubEmbeddingBackend(dimension=256)
        assert backend.dimension == 256

    def test_embed_empty_list(self):
        """Test embedding an empty list."""
        backend = StubEmbeddingBackend(dimension=128)
        vectors = backend.embed([])
        assert vectors.shape == (0, 128)

    def test_embed_single_text(self):
        """Test embedding a single text."""
        backend = StubEmbeddingBackend(dimension=64)
        vectors = backend.embed(["Hello world"])
        assert vectors.shape == (1, 64)

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        backend = StubEmbeddingBackend(dimension=128)
        texts = ["Hello", "World", "Test"]
        vectors = backend.embed(texts)
        assert vectors.shape == (3, 128)

    def test_deterministic_embedding(self):
        """Test that the same text produces the same vector."""
        backend = StubEmbeddingBackend(dimension=128)
        text = "This is a test"

        vectors1 = backend.embed([text])
        vectors2 = backend.embed([text])

        np.testing.assert_array_equal(vectors1, vectors2)

    def test_different_texts_different_vectors(self):
        """Test that different texts produce different vectors."""
        backend = StubEmbeddingBackend(dimension=128)
        vectors = backend.embed(["Hello", "Goodbye"])

        assert not np.allclose(vectors[0], vectors[1])

    def test_vectors_are_normalized(self):
        """Test that vectors are normalized to unit length."""
        backend = StubEmbeddingBackend(dimension=128)
        texts = ["Test 1", "Test 2", "Test 3"]
        vectors = backend.embed(texts)

        for vec in vectors:
            norm = np.linalg.norm(vec)
            assert np.isclose(norm, 1.0, atol=1e-5)


class TestGetEmbeddingBackend:
    """Tests for get_embedding_backend function."""

    def test_get_stub_backend(self):
        """Test getting stub backend."""
        backend = get_embedding_backend("stub", dimension=256)
        assert isinstance(backend, StubEmbeddingBackend)
        assert backend.dimension == 256

    def test_get_unknown_backend(self):
        """Test getting an unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            get_embedding_backend("unknown_backend")


class TestGetEmbeddingBackendFromConfig:
    """Tests for get_embedding_backend_from_config function."""

    def test_stub_from_config(self):
        """Test creating stub backend from config."""
        config = {
            "backend": "stub",
            "params": {"dimension": 512}
        }
        backend = get_embedding_backend_from_config(config)
        assert isinstance(backend, StubEmbeddingBackend)
        assert backend.dimension == 512

    def test_default_backend(self):
        """Test that default backend is stub."""
        config = {}
        backend = get_embedding_backend_from_config(config)
        assert isinstance(backend, StubEmbeddingBackend)
