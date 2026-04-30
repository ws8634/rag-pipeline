"""Tests for the CLI module."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from llm_rag_pipeline.cli import main


class TestIngestCommand:
    """Tests for the ingest CLI command."""

    def test_ingest_nonexistent_directory(self):
        """Test ingest with non-existent directory exits with non-zero code."""
        runner = CliRunner()
        result = runner.invoke(main, ["ingest", "/nonexistent/directory"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "does not exist" in str(result.output).lower()

    def test_ingest_empty_directory(self):
        """Test ingest with empty directory exits with non-zero code."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["ingest", tmpdir])

            assert result.exit_code != 0
            assert "no readable" in result.output.lower() or "no readable" in str(result.output).lower()

    def test_ingest_directory_with_only_empty_files(self):
        """Test ingest with directory containing only empty files."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            empty_file = data_dir / "empty.txt"
            empty_file.write_text("", encoding="utf-8")

            result = runner.invoke(main, ["ingest", str(data_dir)])

            assert result.exit_code != 0

    def test_ingest_valid_documents(self):
        """Test ingest with valid documents."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("This is a test document with some content.", encoding="utf-8")

            md_file = data_dir / "doc.md"
            md_file.write_text("# Markdown Document\n\nThis is markdown content.", encoding="utf-8")

            result = runner.invoke(
                main,
                ["ingest", "--index-dir", str(Path(tmpdir) / "index"), str(data_dir)]
            )

            assert result.exit_code == 0
            assert "loaded" in result.output.lower()
            assert "chunk" in result.output.lower()

            index_dir = Path(tmpdir) / "index"
            assert index_dir.exists()
            assert (index_dir / "index.faiss").exists()
            assert (index_dir / "metadata.json").exists()
            assert (index_dir / "chunks.json").exists()

    def test_ingest_invalid_chunk_size(self):
        """Test ingest with invalid chunk size."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("Test content", encoding="utf-8")

            result = runner.invoke(
                main,
                ["ingest", "--chunk-size", "0", str(data_dir)]
            )

            assert result.exit_code != 0

    def test_ingest_invalid_overlap(self):
        """Test ingest with invalid overlap."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("Test content", encoding="utf-8")

            result = runner.invoke(
                main,
                ["ingest", "--overlap", "1000", "--chunk-size", "100", str(data_dir)]
            )

            assert result.exit_code != 0

    def test_ingest_recursive_directories(self):
        """Test that ingest reads files recursively from subdirectories."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            file1 = data_dir / "root.txt"
            file1.write_text("Root level file", encoding="utf-8")

            sub_dir = data_dir / "subdir"
            sub_dir.mkdir()
            file2 = sub_dir / "nested.md"
            file2.write_text("Nested markdown", encoding="utf-8")

            deep_dir = sub_dir / "deep"
            deep_dir.mkdir()
            file3 = deep_dir / "deep.txt"
            file3.write_text("Deeply nested file", encoding="utf-8")

            result = runner.invoke(
                main,
                ["ingest", "--index-dir", str(Path(tmpdir) / "index"), str(data_dir)]
            )

            assert result.exit_code == 0


class TestQueryCommand:
    """Tests for the query CLI command."""

    def test_query_empty_index(self):
        """Test query on empty index exits with non-zero code."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / "index"
            index_dir.mkdir()

            result = runner.invoke(
                main,
                ["query", "--index-dir", str(index_dir), "test question"]
            )

            assert result.exit_code != 0
            assert "not found" in result.output.lower() or "empty" in result.output.lower()

    def test_query_nonexistent_index(self):
        """Test query on non-existent index exits with non-zero code."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["query", "--index-dir", "/nonexistent/index", "test question"]
        )

        assert result.exit_code != 0

    def test_query_without_text(self):
        """Test query without providing text exits with non-zero code."""
        runner = CliRunner()
        result = runner.invoke(main, ["query"])

        assert result.exit_code != 0

    def test_query_with_valid_index(self):
        """Test query with a valid populated index."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text(
                "RAG stands for Retrieval-Augmented Generation. "
                "It is a technique that combines retrieval and generation in LLMs.",
                encoding="utf-8"
            )

            index_dir = Path(tmpdir) / "index"

            ingest_result = runner.invoke(
                main,
                ["ingest", "--index-dir", str(index_dir), str(data_dir)]
            )
            assert ingest_result.exit_code == 0

            query_result = runner.invoke(
                main,
                ["query", "--index-dir", str(index_dir), "What does RAG stand for?"]
            )

            assert query_result.exit_code == 0

            try:
                output_json = json.loads(query_result.output)
                assert "answer" in output_json
                assert "cites" in output_json
                assert "latency_ms" in output_json
                assert "embed_backend" in output_json
                assert "gen_backend" in output_json
                assert output_json["embed_backend"] == "stub"
                assert output_json["gen_backend"] == "stub"
            except json.JSONDecodeError:
                pytest.fail("Query output is not valid JSON")

    def test_query_json_input(self):
        """Test query with JSON input containing 'query' field."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("Test content for query", encoding="utf-8")

            index_dir = Path(tmpdir) / "index"

            ingest_result = runner.invoke(
                main,
                ["ingest", "--index-dir", str(index_dir), str(data_dir)]
            )
            assert ingest_result.exit_code == 0

            json_input = json.dumps({"query": "What is this about?"})
            query_result = runner.invoke(
                main,
                ["query", "--index-dir", str(index_dir), json_input]
            )

            assert query_result.exit_code == 0

    def test_query_invalid_json_input(self):
        """Test query with invalid JSON input."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("Test content", encoding="utf-8")

            index_dir = Path(tmpdir) / "index"

            ingest_result = runner.invoke(
                main,
                ["ingest", "--index-dir", str(index_dir), str(data_dir)]
            )
            assert ingest_result.exit_code == 0

            invalid_json = '{"query": incomplete'
            query_result = runner.invoke(
                main,
                ["query", "--index-dir", str(index_dir), invalid_json]
            )

            assert query_result.exit_code != 0

    def test_query_with_rerank_disabled(self):
        """Test query with rerank disabled."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("Test content about apples and bananas.", encoding="utf-8")

            index_dir = Path(tmpdir) / "index"

            ingest_result = runner.invoke(
                main,
                ["ingest", "--index-dir", str(index_dir), str(data_dir)]
            )
            assert ingest_result.exit_code == 0

            query_result = runner.invoke(
                main,
                [
                    "query",
                    "--index-dir", str(index_dir),
                    "--no-rerank",
                    "What fruits are mentioned?"
                ]
            )

            assert query_result.exit_code == 0

    def test_query_from_stdin(self):
        """Test query reading from stdin."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            test_file = data_dir / "test.txt"
            test_file.write_text("Test content", encoding="utf-8")

            index_dir = Path(tmpdir) / "index"

            ingest_result = runner.invoke(
                main,
                ["ingest", "--index-dir", str(index_dir), str(data_dir)]
            )
            assert ingest_result.exit_code == 0

            query_result = runner.invoke(
                main,
                ["query", "--index-dir", str(index_dir)],
                input="What is in the documents?"
            )

            assert query_result.exit_code == 0


class TestFullPipeline:
    """Tests for the full RAG pipeline."""

    def test_ingest_and_query_flow(self):
        """Test the complete flow: ingest -> query."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            index_dir = Path(tmpdir) / "index"

            doc1 = data_dir / "doc1.txt"
            doc1.write_text(
                "The quick brown fox jumps over the lazy dog. "
                "This is a classic pangram used for testing.",
                encoding="utf-8"
            )

            doc2 = data_dir / "sub/doc2.md"
            doc2.parent.mkdir()
            doc2.write_text(
                "# Pandas\n\n"
                "Pandas are large, black and white bears native to central China. "
                "They primarily eat bamboo shoots and leaves.",
                encoding="utf-8"
            )

            ingest_result = runner.invoke(
                main,
                [
                    "ingest",
                    "--index-dir", str(index_dir),
                    "--chunk-size", "100",
                    "--overlap", "20",
                    str(data_dir)
                ]
            )
            assert ingest_result.exit_code == 0

            query_result = runner.invoke(
                main,
                [
                    "query",
                    "--index-dir", str(index_dir),
                    "--top-k", "5",
                    "What do pandas eat?"
                ]
            )

            assert query_result.exit_code == 0

            output = json.loads(query_result.output)
            assert output["answer"] is not None
            assert isinstance(output["cites"], list)
            assert isinstance(output["latency_ms"], int)
            assert output["embed_backend"] == "stub"
            assert output["gen_backend"] == "stub"

    def test_clear_flag_ingest(self):
        """Test that --clear flag removes existing index before ingest."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()
            index_dir = Path(tmpdir) / "index"

            doc1 = data_dir / "old.txt"
            doc1.write_text("Old document content", encoding="utf-8")

            result1 = runner.invoke(
                main,
                ["ingest", "--index-dir", str(index_dir), str(data_dir)]
            )
            assert result1.exit_code == 0

            doc1.unlink()
            doc2 = data_dir / "new.txt"
            doc2.write_text("New document content", encoding="utf-8")

            result2 = runner.invoke(
                main,
                ["ingest", "--index-dir", str(index_dir), "--clear", str(data_dir)]
            )
            assert result2.exit_code == 0
            assert "clearing" in result2.output.lower()
