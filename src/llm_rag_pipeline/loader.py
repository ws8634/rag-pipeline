"""Document loading and parsing module."""

import os
from pathlib import Path
from typing import List, Optional, Generator, Tuple
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a loaded document with metadata."""
    content: str
    source: str
    file_path: str
    offset_start: int = 0
    offset_end: int = 0


class DocumentLoader:
    """Loads text and markdown files from a directory recursively."""

    EXTENSIONS = {'.txt', '.md'}

    def __init__(self, data_dir: str):
        """Initialize the document loader.

        Args:
            data_dir: Directory to load documents from.
        """
        self.data_dir = Path(data_dir)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate if the directory exists and is accessible.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.data_dir.exists():
            return False, f"Directory does not exist: {self.data_dir}"
        if not self.data_dir.is_dir():
            return False, f"Path is not a directory: {self.data_dir}"
        return True, None

    def load_documents(self) -> List[Document]:
        """Load all documents from the directory.

        Returns:
            List of Document objects.

        Raises:
            FileNotFoundError: If directory doesn't exist.
            NotADirectoryError: If path is not a directory.
        """
        valid, error = self.validate()
        if not valid:
            if "does not exist" in error:
                raise FileNotFoundError(error)
            else:
                raise NotADirectoryError(error)

        documents = []
        for file_path in self._find_files():
            doc = self._load_file(file_path)
            if doc:
                documents.append(doc)

        return documents

    def _find_files(self) -> Generator[Path, None, None]:
        """Find all files with valid extensions recursively."""
        for root, dirs, files in os.walk(self.data_dir):
            for file_name in files:
                file_path = Path(root) / file_name
                if file_path.suffix.lower() in self.EXTENSIONS:
                    yield file_path

    def _load_file(self, file_path: Path) -> Optional[Document]:
        """Load a single file.

        Returns:
            Document object if successful, None if failed to read.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return Document(
                content=content,
                source=file_path.name,
                file_path=str(file_path.absolute()),
                offset_start=0,
                offset_end=len(content)
            )
        except (UnicodeDecodeError, IOError, PermissionError):
            return None
