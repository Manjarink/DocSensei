"""
tests/test_utils.py – Unit tests for DocSensei utility modules.

Covers loaders, splitter, helpers, and prompts to ensure core
logic works correctly without requiring API keys or a live LLM.
"""

import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# Loader Tests
# ─────────────────────────────────────────────

class TestFileValidation:
    """Tests for utils/loaders.py validate_file function."""

    def test_nonexistent_file(self, tmp_path):
        from utils.loaders import validate_file
        ok, msg = validate_file(str(tmp_path / "ghost.pdf"))
        assert not ok
        assert "does not exist" in msg

    def test_unsupported_extension(self, tmp_path):
        from utils.loaders import validate_file
        f = tmp_path / "document.txt"
        f.write_text("hello")
        ok, msg = validate_file(str(f))
        assert not ok
        assert "Unsupported" in msg

    def test_empty_file(self, tmp_path):
        from utils.loaders import validate_file
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        ok, msg = validate_file(str(f))
        assert not ok
        assert "empty" in msg

    def test_oversized_file(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "MAX_FILE_SIZE_BYTES", 10)
        from utils.loaders import validate_file
        f = tmp_path / "big.pdf"
        f.write_bytes(b"x" * 100)
        ok, msg = validate_file(str(f))
        assert not ok
        assert "too large" in msg

    def test_valid_pdf_file(self, tmp_path):
        from utils.loaders import validate_file
        f = tmp_path / "valid.pdf"
        f.write_bytes(b"fake pdf content")
        ok, msg = validate_file(str(f))
        assert ok
        assert msg == ""


class TestFileHash:
    """Tests for compute_file_hash."""

    def test_same_content_same_hash(self, tmp_path):
        from utils.loaders import compute_file_hash
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"same content")
        f2.write_bytes(b"same content")
        assert compute_file_hash(str(f1)) == compute_file_hash(str(f2))

    def test_different_content_different_hash(self, tmp_path):
        from utils.loaders import compute_file_hash
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_file_hash(str(f1)) != compute_file_hash(str(f2))


# ─────────────────────────────────────────────
# Splitter Tests
# ─────────────────────────────────────────────

class TestSplitter:
    """Tests for utils/splitter.py."""

    def _make_doc(self, text: str, source: str = "test.pdf", page: int = 1) -> Document:
        return Document(page_content=text, metadata={"source": source, "page": page})

    def test_split_basic(self):
        from utils.splitter import split_documents
        docs = [self._make_doc("Hello world " * 500)]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_split_preserves_metadata(self):
        from utils.splitter import split_documents
        docs = [self._make_doc("Test content " * 300, source="my.pdf", page=3)]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
        assert all(c.metadata["source"] == "my.pdf" for c in chunks)
        assert all(c.metadata["page"] == 3 for c in chunks)

    def test_split_adds_chunk_index(self):
        from utils.splitter import split_documents
        docs = [self._make_doc("Hello " * 600)]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=10)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(1, len(chunks) + 1))

    def test_split_empty_raises(self):
        from utils.splitter import split_documents
        with pytest.raises(ValueError, match="No documents"):
            split_documents([])

    def test_split_stats(self):
        from utils.splitter import split_documents, get_split_stats
        docs = [
            self._make_doc("A " * 400, source="doc1.pdf"),
            self._make_doc("B " * 400, source="doc2.pdf"),
        ]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=10)
        stats = get_split_stats(chunks)
        assert "doc1.pdf" in stats
        assert "doc2.pdf" in stats


# ─────────────────────────────────────────────
# Helper Tests
# ─────────────────────────────────────────────

class TestHelpers:
    """Tests for utils/helpers.py."""

    def test_format_file_size_bytes(self):
        from utils.helpers import format_file_size
        assert format_file_size(500) == "500 B"

    def test_format_file_size_kb(self):
        from utils.helpers import format_file_size
        assert "KB" in format_file_size(2048)

    def test_format_file_size_mb(self):
        from utils.helpers import format_file_size
        assert "MB" in format_file_size(5 * 1024 * 1024)

    def test_truncate_text_short(self):
        from utils.helpers import truncate_text
        assert truncate_text("hello", 200) == "hello"

    def test_truncate_text_long(self):
        from utils.helpers import truncate_text
        long = "x" * 300
        result = truncate_text(long, 100)
        assert result.endswith("…")
        assert len(result) <= 104  # 100 + ellipsis

    def test_format_sources_basic(self):
        from utils.helpers import format_sources
        docs = [
            Document(page_content="", metadata={"source": "a.pdf", "page": 5}),
            Document(page_content="", metadata={"source": "b.docx", "page": "N/A"}),
        ]
        result = format_sources(docs)
        assert "a.pdf" in result
        assert "Page 5" in result
        assert "b.docx" in result

    def test_format_sources_deduplication(self):
        from utils.helpers import format_sources
        docs = [
            Document(page_content="", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="", metadata={"source": "a.pdf", "page": 1}),
        ]
        result = format_sources(docs)
        assert result.count("a.pdf") == 1

    def test_export_chat_to_json(self):
        from utils.helpers import export_chat_to_json
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        output = export_chat_to_json(history)
        parsed = json.loads(output)
        assert len(parsed) == 2

    def test_search_chat_history(self):
        from utils.helpers import search_chat_history
        history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me about Java."},
        ]
        results = search_chat_history(history, "python")
        assert len(results) == 2

    def test_search_chat_history_no_match(self):
        from utils.helpers import search_chat_history
        history = [{"role": "user", "content": "Hello world"}]
        results = search_chat_history(history, "xyz123")
        assert results == []

    def test_chat_history_save_load(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "BASE_DIR", tmp_path)
        monkeypatch.setattr(config, "CHAT_HISTORY_FILE", "test_chat.json")

        from utils import helpers
        # Force reload path
        history = [{"role": "user", "content": "Test message"}]
        helpers.save_chat_history(history)
        loaded = helpers.load_chat_history()
        assert loaded == history


# ─────────────────────────────────────────────
# Prompt Tests
# ─────────────────────────────────────────────

class TestPrompts:
    """Tests for utils/prompts.py."""

    def test_system_prompt_not_empty(self):
        from utils.prompts import SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 100

    def test_rag_prompt_has_context(self):
        from utils.prompts import SYSTEM_PROMPT
        assert "{context}" in SYSTEM_PROMPT

    def test_contextualise_prompt_exists(self):
        from utils.prompts import CONTEXTUALISE_Q_PROMPT
        assert CONTEXTUALISE_Q_PROMPT is not None

    def test_rag_prompt_chain(self):
        from utils.prompts import RAG_PROMPT
        assert RAG_PROMPT is not None

    def test_summary_template_has_placeholders(self):
        from utils.prompts import SUMMARY_PROMPT_TEMPLATE
        assert "{doc_name}" in SUMMARY_PROMPT_TEMPLATE
        assert "{content}" in SUMMARY_PROMPT_TEMPLATE
