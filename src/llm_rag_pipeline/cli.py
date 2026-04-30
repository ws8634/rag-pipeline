"""Command Line Interface for LLM RAG Pipeline."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click

from .loader import DocumentLoader
from .chunker import TextChunker
from .embedding import get_embedding_backend, StubEmbeddingBackend
from .index_store import VectorIndexStore, ChunkMetadata
from .retriever import Retriever, RetrievalResult
from .generator import get_generation_backend, GenerationResult


DEFAULT_INDEX_DIR = ".rag_index"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 100
DEFAULT_TOP_K = 5
DEFAULT_EMBED_DIM = 128


def _get_index_dir(index_dir: Optional[str] = None) -> str:
    """Get the index directory, using environment variable or default."""
    if index_dir:
        return index_dir
    env_index = os.getenv("RAG_INDEX_DIR")
    if env_index:
        return env_index
    return DEFAULT_INDEX_DIR


def _error_exit(message: str, code: int = 1):
    """Print error to stderr and exit with non-zero code."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(code)


@click.group()
@click.version_option(version="0.1.0")
def main():
    """LLM RAG Pipeline - A reference RAG implementation for testing."""
    pass


@main.command()
@click.argument('data_dir', type=click.Path(exists=False))
@click.option('--index-dir', '-i', type=click.Path(), 
              help='Directory to store the vector index (default: .rag_index)')
@click.option('--chunk-size', '-s', type=int, default=DEFAULT_CHUNK_SIZE,
              help=f'Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})')
@click.option('--overlap', '-o', type=int, default=DEFAULT_OVERLAP,
              help=f'Overlap between chunks in characters (default: {DEFAULT_OVERLAP})')
@click.option('--embed-backend', '-e', type=click.Choice(['stub', 'openai']),
              default='stub', help='Embedding backend to use (default: stub)')
@click.option('--embed-dim', type=int, default=DEFAULT_EMBED_DIM,
              help=f'Embedding dimension for stub backend (default: {DEFAULT_EMBED_DIM})')
@click.option('--on-conflict', type=click.Choice(['skip', 'fail']), default='skip',
              help='Strategy when duplicate chunk IDs exist (default: skip)')
@click.option('--clear', is_flag=True, help='Clear existing index before ingesting')
def ingest(
    data_dir: str,
    index_dir: Optional[str],
    chunk_size: int,
    overlap: int,
    embed_backend: str,
    embed_dim: int,
    on_conflict: str,
    clear: bool
):
    """Ingest documents from a directory into the vector index.

    DATA_DIR: Directory containing .txt and .md files to ingest.
    """
    index_path = _get_index_dir(index_dir)

    if clear:
        click.echo(f"Clearing existing index at: {index_path}")
        tmp_store = VectorIndexStore(index_path, embed_dim)
        tmp_store.clear()

    loader = DocumentLoader(data_dir)

    valid, error_msg = loader.validate()
    if not valid:
        _error_exit(error_msg, 1)

    try:
        documents = loader.load_documents()
    except FileNotFoundError as e:
        _error_exit(str(e), 1)
    except NotADirectoryError as e:
        _error_exit(str(e), 1)

    if not documents:
        _error_exit(
            f"No readable documents found in {data_dir}. "
            "Ensure the directory contains .txt or .md files that can be read.",
            2
        )

    click.echo(f"Loaded {len(documents)} document(s)")

    try:
        chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    except ValueError as e:
        _error_exit(str(e), 1)

    all_chunks = chunker.chunk_documents(documents)

    if not all_chunks:
        _error_exit(
            "No valid chunks created. All documents may be empty.",
            3
        )

    click.echo(f"Created {len(all_chunks)} chunk(s) from documents")

    try:
        if embed_backend == "stub":
            embedding_backend = get_embedding_backend("stub", dimension=embed_dim)
        else:
            embedding_backend = get_embedding_backend(embed_backend)
    except (ValueError, ImportError) as e:
        _error_exit(f"Failed to initialize embedding backend: {e}", 1)

    store = VectorIndexStore(index_path, embedding_backend.dimension)

    exists = store.load()

    compatible, compat_error = store.check_compatibility(
        embedding_backend.dimension,
        embed_backend
    )
    if not compatible:
        _error_exit(compat_error, 4)

    if not exists:
        store.create(embed_backend)

    click.echo(f"Embedding chunks using {embedding_backend.name} backend...")

    chunk_texts = [chunk.content for chunk in all_chunks]
    vectors = embedding_backend.embed(chunk_texts)

    metadatas = [
        ChunkMetadata(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            source=chunk.source,
            file_path=chunk.file_path,
            offset_start=chunk.offset_start,
            offset_end=chunk.offset_end,
            index=chunk.index
        )
        for chunk in all_chunks
    ]

    try:
        added = store.add_vectors(vectors, metadatas, on_conflict=on_conflict)
    except ValueError as e:
        _error_exit(str(e), 5)
    except NotImplementedError as e:
        _error_exit(str(e), 6)

    store.save()

    click.echo(f"Successfully added {added} chunk(s) to index")
    click.echo(f"Index stored at: {index_path}")


@main.command()
@click.argument('query_text', required=False, type=str)
@click.option('--index-dir', '-i', type=click.Path(),
              help='Directory containing the vector index (default: .rag_index)')
@click.option('--top-k', '-k', type=int, default=DEFAULT_TOP_K,
              help=f'Number of top results to retrieve (default: {DEFAULT_TOP_K})')
@click.option('--embed-backend', '-e', type=click.Choice(['stub', 'openai']),
              default='stub', help='Embedding backend to use (default: stub)')
@click.option('--embed-dim', type=int, default=DEFAULT_EMBED_DIM,
              help=f'Embedding dimension for stub backend (default: {DEFAULT_EMBED_DIM})')
@click.option('--gen-backend', '-g', type=click.Choice(['stub', 'openai']),
              default='stub', help='Generation backend to use (default: stub)')
@click.option('--rerank/--no-rerank', default=True,
              help='Enable/disable reranking (default: enabled)')
@click.option('--reranker', type=click.Choice(['tf_idf', 'length_penalty']),
              default='tf_idf', help='Reranker to use (default: tf_idf)')
@click.option('--max-context-chars', type=int, default=8000,
              help='Maximum characters for context (default: 8000)')
def query(
    query_text: Optional[str],
    index_dir: Optional[str],
    top_k: int,
    embed_backend: str,
    embed_dim: int,
    gen_backend: str,
    rerank: bool,
    reranker: str,
    max_context_chars: int
):
    """Query the RAG system and get answers with citations.

    QUERY_TEXT: The question to ask. If not provided, reads from stdin.
    """
    index_path = _get_index_dir(index_dir)

    if query_text is None:
        query_text = sys.stdin.read().strip()

    if not query_text:
        _error_exit("No query text provided. Pass as argument or pipe to stdin.", 1)

    if query_text.strip().startswith('{'):
        try:
            data = json.loads(query_text)
            if "query" in data:
                query_text = data["query"]
            else:
                _error_exit("JSON input must contain a 'query' field.", 2)
        except json.JSONDecodeError as e:
            _error_exit(f"Invalid JSON input: {e}", 2)

    store = VectorIndexStore(index_path, embed_dim)
    exists = store.load()

    if not exists or store.is_empty:
        _error_exit(
            f"Index not found or empty at {index_path}. "
            "Run 'rag ingest' first to populate the index.",
            3
        )

    try:
        if embed_backend == "stub":
            embedding_backend = get_embedding_backend("stub", dimension=store.dimension)
        else:
            embedding_backend = get_embedding_backend(embed_backend)
    except (ValueError, ImportError) as e:
        _error_exit(f"Failed to initialize embedding backend: {e}", 4)

    index_meta = store.get_index_metadata()
    if index_meta.embedding_backend != embed_backend:
        _error_exit(
            f"Embedding backend mismatch. Index uses {index_meta.embedding_backend}, "
            f"but query uses {embed_backend}. Please use the same backend.",
            5
        )

    try:
        retriever = Retriever(
            index_store=store,
            embedding_backend=embedding_backend,
            reranker=reranker if rerank else None
        )
    except ValueError as e:
        _error_exit(f"Failed to initialize retriever: {e}", 6)

    start_time = time.time()

    retrieval_result = retriever.retrieve(
        query_text,
        top_k=top_k,
        enable_rerank=rerank
    )

    try:
        generation_backend = get_generation_backend(gen_backend)
    except (ValueError, ImportError) as e:
        _error_exit(f"Failed to initialize generation backend: {e}", 7)

    generation_result = generation_backend.generate(
        query_text,
        retrieval_result,
        max_context_chars=max_context_chars
    )

    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)

    output = {
        "answer": generation_result.answer,
        "cites": generation_result.cites,
        "latency_ms": latency_ms,
        "embed_backend": embed_backend,
        "gen_backend": gen_backend,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
