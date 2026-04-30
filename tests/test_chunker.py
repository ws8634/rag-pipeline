"""Tests for the TextChunker module."""

import pytest

from llm_rag_pipeline.chunker import (
    TextChunker,
    Chunk,
    calculate_overlap_between_chunks
)
from llm_rag_pipeline.loader import Document


class TestTextChunker:
    """Tests for TextChunker class."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20

    def test_init_invalid_chunk_size(self):
        """Test initialization with invalid chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0, overlap=10)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=-5, overlap=10)

    def test_init_invalid_overlap(self):
        """Test initialization with invalid overlap."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            TextChunker(chunk_size=100, overlap=-1)
        with pytest.raises(ValueError, match="overlap.*must be less than chunk_size"):
            TextChunker(chunk_size=100, overlap=100)
        with pytest.raises(ValueError, match="overlap.*must be less than chunk_size"):
            TextChunker(chunk_size=100, overlap=150)

    def test_chunk_empty_document(self):
        """Test chunking an empty document."""
        doc = Document(content="", source="test.txt", file_path="/tmp/test.txt")
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_chunk_single_chunk(self):
        """Test chunking a document that fits in one chunk."""
        content = "This is a short document."
        doc = Document(content=content, source="test.txt", file_path="/tmp/test.txt")
        chunker = TextChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].offset_start == 0
        assert chunks[0].offset_end == len(content)

    def test_chunk_multiple_chunks_no_overlap(self):
        """Test chunking with no overlap."""
        content = "abcdefghijklmnopqrstuvwxyz"
        doc = Document(content=content, source="test.txt", file_path="/tmp/test.txt")
        chunker = TextChunker(chunk_size=10, overlap=0)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 3
        assert chunks[0].content == "abcdefghij"
        assert chunks[0].offset_start == 0
        assert chunks[0].offset_end == 10

        assert chunks[1].content == "klmnopqrst"
        assert chunks[1].offset_start == 10
        assert chunks[1].offset_end == 20

        assert chunks[2].content == "uvwxyz"
        assert chunks[2].offset_start == 20
        assert chunks[2].offset_end == 26

    def test_chunk_with_overlap(self):
        """Test chunking with overlap between chunks."""
        content = "abcdefghijklmnopqrstuvwxyz"
        doc = Document(content=content, source="test.txt", file_path="/tmp/test.txt")
        chunker = TextChunker(chunk_size=10, overlap=3)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 4

        assert chunks[0].offset_start == 0
        assert chunks[0].offset_end == 10

        assert chunks[1].offset_start == 7
        assert chunks[1].offset_end == 17

        assert chunks[2].offset_start == 14
        assert chunks[2].offset_end == 24

        assert chunks[3].offset_start == 21
        assert chunks[3].offset_end == 26

    def test_overlap_relationship(self):
        """Test that consecutive chunks have the expected overlap."""
        content = "abcdefghijklmnopqrstuvwxyz0123456789"
        doc = Document(content=content, source="test.txt", file_path="/tmp/test.txt")
        chunk_size = 10
        overlap = 3
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk_document(doc)

        for i in range(len(chunks) - 1):
            actual_overlap = calculate_overlap_between_chunks(chunks[i], chunks[i + 1])
            if chunks[i].file_path == chunks[i + 1].file_path:
                expected_overlap = overlap
                if chunks[i + 1].offset_end - chunks[i + 1].offset_start < chunk_size:
                    expected_overlap = min(
                        overlap,
                        chunks[i].offset_end - chunks[i + 1].offset_start
                    )
                    expected_overlap = max(0, expected_overlap)
                assert actual_overlap == expected_overlap, (
                    f"Chunk {i} and {i+1} should have overlap of {expected_overlap}, "
                    f"got {actual_overlap}"
                )

    def test_utf8_characters(self):
        """Test chunking with multi-byte UTF-8 characters.

        In Python, strings are Unicode, so we're testing that we don't
        split in the middle of a character by using the string's character
        indices rather than byte indices.
        """
        content = "这是一段中文测试文本。"
        doc = Document(content=content, source="test.txt", file_path="/tmp/test.txt")
        chunker = TextChunker(chunk_size=5, overlap=2)
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk.content == content[chunk.offset_start:chunk.offset_end]
            for char in chunk.content:
                assert len(char.encode('utf-8')) > 0

    def test_utf8_mixed_content(self):
        """Test chunking with mixed ASCII and multi-byte characters."""
        content = "Hello 世界! This is a test with 中文和English混合."
        doc = Document(content=content, source="test.txt", file_path="/tmp/test.txt")
        chunker = TextChunker(chunk_size=10, overlap=3)
        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            chunk_content = content[chunk.offset_start:chunk.offset_end]
            assert chunk.content == chunk_content
            try:
                chunk.content.encode('utf-8').decode('utf-8')
            except UnicodeDecodeError:
                pytest.fail("Chunk contains invalid UTF-8")

    def test_validate_utf8_boundary(self):
        """Test UTF-8 boundary validation."""
        chunker = TextChunker(chunk_size=100, overlap=20)
        content = "Hello World"

        assert chunker.validate_utf8_boundary(content, 0) is True
        assert chunker.validate_utf8_boundary(content, 5) is True
        assert chunker.validate_utf8_boundary(content, len(content)) is True

        assert chunker.validate_utf8_boundary(content, -1) is False
        assert chunker.validate_utf8_boundary(content, len(content) + 1) is False

    def test_chunk_id_format(self):
        """Test that chunk IDs are correctly formatted."""
        content = "abcdefghijklmnopqrst"
        doc = Document(content=content, source="mydoc.txt", file_path="/tmp/mydoc.txt")
        chunker = TextChunker(chunk_size=10, overlap=0)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 2
        assert chunks[0].chunk_id == "mydoc.txt:0:10"
        assert chunks[1].chunk_id == "mydoc.txt:10:20"

    def test_chunk_indexes(self):
        """Test that chunk indexes are sequential."""
        content = "a" * 100
        doc = Document(content=content, source="test.txt", file_path="/tmp/test.txt")
        chunker = TextChunker(chunk_size=30, overlap=5)
        chunks = chunker.chunk_document(doc)

        expected_indexes = list(range(len(chunks)))
        actual_indexes = [chunk.index for chunk in chunks]
        assert actual_indexes == expected_indexes


class TestCalculateOverlapBetweenChunks:
    """Tests for calculate_overlap_between_chunks function."""

    def test_no_overlap_different_files(self):
        """Test chunks from different files have no overlap."""
        chunk1 = Chunk(
            content="abc", chunk_id="file1:0:3",
            source="file1.txt", file_path="/tmp/file1.txt",
            offset_start=0, offset_end=3, index=0
        )
        chunk2 = Chunk(
            content="abc", chunk_id="file2:0:3",
            source="file2.txt", file_path="/tmp/file2.txt",
            offset_start=0, offset_end=3, index=0
        )
        assert calculate_overlap_between_chunks(chunk1, chunk2) == 0

    def test_no_overlap_same_file(self):
        """Test chunks from same file with no overlap."""
        chunk1 = Chunk(
            content="abc", chunk_id="file1:0:3",
            source="file1.txt", file_path="/tmp/file1.txt",
            offset_start=0, offset_end=3, index=0
        )
        chunk2 = Chunk(
            content="def", chunk_id="file1:3:6",
            source="file1.txt", file_path="/tmp/file1.txt",
            offset_start=3, offset_end=6, index=1
        )
        assert calculate_overlap_between_chunks(chunk1, chunk2) == 0

    def test_with_overlap(self):
        """Test chunks with actual overlap."""
        chunk1 = Chunk(
            content="abcdef", chunk_id="file1:0:6",
            source="file1.txt", file_path="/tmp/file1.txt",
            offset_start=0, offset_end=6, index=0
        )
        chunk2 = Chunk(
            content="cdefgh", chunk_id="file1:2:8",
            source="file1.txt", file_path="/tmp/file1.txt",
            offset_start=2, offset_end=8, index=1
        )
        assert calculate_overlap_between_chunks(chunk1, chunk2) == 4

    def test_full_overlap(self):
        """Test when one chunk is entirely within another."""
        chunk1 = Chunk(
            content="abcdefgh", chunk_id="file1:0:8",
            source="file1.txt", file_path="/tmp/file1.txt",
            offset_start=0, offset_end=8, index=0
        )
        chunk2 = Chunk(
            content="cdef", chunk_id="file1:2:6",
            source="file1.txt", file_path="/tmp/file1.txt",
            offset_start=2, offset_end=6, index=1
        )
        assert calculate_overlap_between_chunks(chunk1, chunk2) == 4
