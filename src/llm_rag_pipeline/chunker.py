"""Text chunking module with UTF-8 friendly handling."""

from typing import List, Tuple
from dataclasses import dataclass
from .loader import Document


@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    content: str
    chunk_id: str
    source: str
    file_path: str
    offset_start: int
    offset_end: int
    index: int


class TextChunker:
    """UTF-8 friendly text chunker with overlap support.

    This chunker ensures that:
    1. Chunks never split multi-byte UTF-8 characters
    2. Overlap between chunks is preserved correctly
    3. Chunk boundaries are calculated based on character positions
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """Initialize the text chunker.

        Args:
            chunk_size: Maximum number of characters per chunk.
            overlap: Number of overlapping characters between consecutive chunks.

        Raises:
            ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document into multiple chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of Chunk objects.
        """
        content = document.content
        chunks = []

        if not content:
            return chunks

        effective_step = self.chunk_size - self.overlap
        if effective_step <= 0:
            effective_step = 1

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            chunk_content = content[start:end]

            chunk_id = f"{document.source}:{start}:{end}"

            chunk = Chunk(
                content=chunk_content,
                chunk_id=chunk_id,
                source=document.source,
                file_path=document.file_path,
                offset_start=start,
                offset_end=end,
                index=chunk_index
            )
            chunks.append(chunk)

            start += effective_step
            chunk_index += 1

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk multiple documents.

        Args:
            documents: List of Document objects.

        Returns:
            Flat list of Chunk objects from all documents.
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def validate_utf8_boundary(self, text: str, position: int) -> bool:
        """Check if a position is at a valid UTF-8 character boundary.

        Note: In Python, strings are Unicode, so this is primarily a sanity check.
        The actual UTF-8 safety comes from operating on character indices, not byte indices.

        Args:
            text: The text string.
            position: Position to check.

        Returns:
            True if position is a valid boundary.
        """
        if position < 0 or position > len(text):
            return False
        return True


def calculate_overlap_between_chunks(chunk1: Chunk, chunk2: Chunk) -> int:
    """Calculate the number of overlapping characters between two consecutive chunks.

    This is a utility function for testing purposes.

    Args:
        chunk1: First chunk.
        chunk2: Second chunk (should be consecutive).

    Returns:
        Number of overlapping characters.
    """
    if chunk1.file_path != chunk2.file_path:
        return 0

    overlap_start = max(chunk1.offset_start, chunk2.offset_start)
    overlap_end = min(chunk1.offset_end, chunk2.offset_end)

    return max(0, overlap_end - overlap_start)
