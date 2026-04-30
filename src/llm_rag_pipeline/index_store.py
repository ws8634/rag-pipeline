"""Vector index and storage module."""

import json
import os
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


@dataclass
class SearchResult:
    """Result of a vector search."""
    chunk_id: str
    score: float
    content: str
    source: str
    file_path: str
    offset_start: int
    offset_end: int
    index: int


@dataclass
class ChunkMetadata:
    """Metadata for a stored chunk."""
    chunk_id: str
    content: str
    source: str
    file_path: str
    offset_start: int
    offset_end: int
    index: int


@dataclass
class IndexMetadata:
    """Metadata for the entire index."""
    version: str = "1.0"
    dimension: int = 128
    embedding_backend: str = "stub"
    created_at: str = ""
    updated_at: str = ""
    num_vectors: int = 0


class VectorIndexStore:
    """Vector index store using FAISS with metadata persistence.

    This store handles:
    1. Vector indexing using FAISS
    2. Chunk metadata storage
    3. Index metadata (version, dimension, backend)
    4. Idempotent updates and conflict resolution
    """

    INDEX_FILE = "index.faiss"
    METADATA_FILE = "metadata.json"
    CHUNKS_FILE = "chunks.json"

    def __init__(self, index_dir: str, dimension: int = 128):
        """Initialize the vector index store.

        Args:
            index_dir: Directory to store index files.
            dimension: Dimension of vectors to store.
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu not installed. Install with: pip install faiss-cpu>=1.7.0"
            )

        self.index_dir = Path(index_dir)
        self.dimension = dimension

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._index: Optional[faiss.IndexFlatL2] = None
        self._chunk_metadata: Dict[str, ChunkMetadata] = {}
        self._index_metadata: IndexMetadata = IndexMetadata(dimension=dimension)
        self._id_map: Dict[int, str] = {}
        self._next_id: int = 0

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self._index is None or len(self._chunk_metadata) == 0

    @property
    def num_vectors(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._chunk_metadata)

    def _get_index_path(self) -> Path:
        return self.index_dir / self.INDEX_FILE

    def _get_metadata_path(self) -> Path:
        return self.index_dir / self.METADATA_FILE

    def _get_chunks_path(self) -> Path:
        return self.index_dir / self.CHUNKS_FILE

    def create(self, embedding_backend: str = "stub"):
        """Create a new empty index.

        Args:
            embedding_backend: Name of the embedding backend used.
        """
        import datetime
        now = datetime.datetime.now().isoformat()

        self._index = faiss.IndexFlatL2(self.dimension)
        self._chunk_metadata = {}
        self._id_map = {}
        self._next_id = 0

        self._index_metadata = IndexMetadata(
            dimension=self.dimension,
            embedding_backend=embedding_backend,
            created_at=now,
            updated_at=now,
            num_vectors=0
        )

    def load(self) -> bool:
        """Load index from disk.

        Returns:
            True if successfully loaded, False if no index exists.
        """
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        chunks_path = self._get_chunks_path()

        if not index_path.exists() or not metadata_path.exists() or not chunks_path.exists():
            return False

        try:
            self._index = faiss.read_index(str(index_path))
            self.dimension = self._index.d

            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta_dict = json.load(f)
            self._index_metadata = IndexMetadata(**meta_dict)

            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_dict = json.load(f)

            self._chunk_metadata = {}
            self._id_map = {}
            max_id = 0

            for internal_id_str, chunk_dict in chunks_dict.items():
                internal_id = int(internal_id_str)
                metadata = ChunkMetadata(**chunk_dict)
                self._chunk_metadata[metadata.chunk_id] = metadata
                self._id_map[internal_id] = metadata.chunk_id
                if internal_id > max_id:
                    max_id = internal_id

            self._next_id = max_id + 1

            return True
        except Exception:
            return False

    def save(self):
        """Save index to disk."""
        import datetime

        if self._index is None:
            self.create()

        self._index_metadata.updated_at = datetime.datetime.now().isoformat()
        self._index_metadata.num_vectors = len(self._chunk_metadata)

        faiss.write_index(self._index, str(self._get_index_path()))

        with open(self._get_metadata_path(), 'w', encoding='utf-8') as f:
            json.dump(asdict(self._index_metadata), f, indent=2)

        chunks_dict = {}
        for internal_id, chunk_id in self._id_map.items():
            metadata = self._chunk_metadata[chunk_id]
            chunks_dict[str(internal_id)] = asdict(metadata)

        with open(self._get_chunks_path(), 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, indent=2, ensure_ascii=False)

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadatas: List[ChunkMetadata],
        on_conflict: str = "replace"
    ) -> int:
        """Add vectors and metadata to the index.

        Args:
            vectors: Numpy array of shape (n, dimension)
            metadatas: List of metadata for each vector
            on_conflict: Strategy for existing chunk_ids:
                - "replace": Replace existing vectors
                - "skip": Skip existing vectors
                - "fail": Raise ValueError if any chunk exists

        Returns:
            Number of vectors actually added.

        Raises:
            ValueError: If on_conflict is "fail" and a duplicate exists.
        """
        if self._index is None:
            self.create()

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch. Expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        if len(vectors) != len(metadatas):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) must match "
                f"number of metadatas ({len(metadatas)})"
            )

        existing_ids = set(self._chunk_metadata.keys())
        to_add: List[Tuple[np.ndarray, ChunkMetadata]] = []
        to_replace: List[Tuple[np.ndarray, ChunkMetadata]] = []

        for i, metadata in enumerate(metadatas):
            if metadata.chunk_id in existing_ids:
                if on_conflict == "fail":
                    raise ValueError(
                        f"Chunk ID already exists: {metadata.chunk_id}. "
                        f"Use on_conflict='replace' or 'skip' to handle duplicates."
                    )
                elif on_conflict == "replace":
                    to_replace.append((vectors[i], metadata))
                elif on_conflict == "skip":
                    continue
            else:
                to_add.append((vectors[i], metadata))

        added_count = 0

        for vector, metadata in to_add:
            internal_id = self._next_id
            self._next_id += 1

            vector_2d = vector.reshape(1, -1)
            self._index.add(vector_2d)

            self._chunk_metadata[metadata.chunk_id] = metadata
            self._id_map[internal_id] = metadata.chunk_id
            added_count += 1

        if to_replace:
            raise NotImplementedError(
                "FAISS IndexFlatL2 does not support in-place updates. "
                "Please recreate the index or use on_conflict='skip'."
            )

        return added_count

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector of shape (dimension,) or (1, dimension)
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects, sorted by score (lower is better for L2).
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        actual_k = min(top_k, self._index.ntotal)
        if actual_k == 0:
            return []

        distances, indices = self._index.search(query_vector, actual_k)

        results = []
        for i in range(actual_k):
            internal_id = int(indices[0][i])
            distance = float(distances[0][i])

            if internal_id in self._id_map:
                chunk_id = self._id_map[internal_id]
                metadata = self._chunk_metadata[chunk_id]

                result = SearchResult(
                    chunk_id=metadata.chunk_id,
                    score=distance,
                    content=metadata.content,
                    source=metadata.source,
                    file_path=metadata.file_path,
                    offset_start=metadata.offset_start,
                    offset_end=metadata.offset_end,
                    index=metadata.index
                )
                results.append(result)

        return results

    def get_index_metadata(self) -> IndexMetadata:
        """Get the index metadata."""
        return self._index_metadata

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkMetadata]:
        """Get chunk metadata by chunk ID."""
        return self._chunk_metadata.get(chunk_id)

    def clear(self):
        """Clear all data from the index."""
        self._index = None
        self._chunk_metadata = {}
        self._id_map = {}
        self._next_id = 0

        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        chunks_path = self._get_chunks_path()

        for path in [index_path, metadata_path, chunks_path]:
            if path.exists():
                path.unlink()

    def check_compatibility(
        self,
        dimension: int,
        embedding_backend: str
    ) -> Tuple[bool, str]:
        """Check if new data is compatible with existing index.

        Args:
            dimension: Vector dimension
            embedding_backend: Embedding backend name

        Returns:
            Tuple of (is_compatible, error_message)
        """
        if self._index is None and len(self._chunk_metadata) == 0:
            return True, ""

        if self.dimension != dimension:
            return False, (
                f"Dimension mismatch. Existing index has dimension {self.dimension}, "
                f"new data has dimension {dimension}. "
                f"Please clear the index or use a different index directory."
            )

        if self._index_metadata.embedding_backend != embedding_backend:
            return False, (
                f"Embedding backend mismatch. Existing index uses "
                f"{self._index_metadata.embedding_backend}, new data uses {embedding_backend}. "
                f"Please clear the index or use a different index directory."
            )

        return True, ""
