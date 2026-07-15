"""
utils/helpers.py – General-purpose helper utilities for DocSensei.

Contains file management, chat history persistence, export, and misc helpers.
"""

import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fpdf import FPDF

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# File Management Helpers
# ─────────────────────────────────────────────

def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to the uploads directory.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Absolute path string of the saved file.

    Raises:
        IOError: If the file cannot be written.
    """
    dest_path = config.UPLOADS_DIR / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info("Saved uploaded file: %s", dest_path)
    return str(dest_path)


def save_uploaded_files(uploaded_files: list) -> tuple[list[str], list[str]]:
    """
    Save multiple uploaded files, returning paths and any errors.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple of (saved_paths, error_messages).
    """
    saved_paths: list[str] = []
    errors: list[str] = []

    for uf in uploaded_files:
        try:
            path = save_uploaded_file(uf)
            saved_paths.append(path)
        except IOError as exc:
            errors.append(str(exc))

    return saved_paths, errors


def cleanup_uploads() -> None:
    """Remove all files in the uploads directory."""
    for file_path in config.UPLOADS_DIR.iterdir():
        if file_path.is_file() and file_path.name != ".gitkeep":
            file_path.unlink()
            logger.info("Deleted upload: %s", file_path)


def format_file_size(size_bytes: int) -> str:
    """
    Convert a byte count to a human-readable string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Human-readable string (e.g., '2.3 MB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.1f} GB"


# ─────────────────────────────────────────────
# Chat History Persistence
# ─────────────────────────────────────────────

def _chat_history_path() -> Path:
    return config.BASE_DIR / config.CHAT_HISTORY_FILE


def save_chat_history(history: list[dict[str, Any]]) -> None:
    """
    Persist chat history to a JSON file.

    Args:
        history: List of message dicts with 'role' and 'content' keys.
    """
    try:
        with open(_chat_history_path(), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.error("Failed to save chat history: %s", exc)


def load_chat_history() -> list[dict[str, Any]]:
    """
    Load persisted chat history from disk.

    Returns:
        List of message dicts, or empty list if file doesn't exist.
    """
    path = _chat_history_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to load chat history: %s", exc)
        return []


def clear_chat_history() -> None:
    """Delete the persisted chat history file."""
    path = _chat_history_path()
    if path.exists():
        path.unlink()
        logger.info("Chat history cleared.")


def export_chat_to_json(history: list[dict[str, Any]]) -> str:
    """
    Serialise chat history to a JSON string for download.

    Args:
        history: List of message dicts.

    Returns:
        Pretty-printed JSON string.
    """
    return json.dumps(history, ensure_ascii=False, indent=2)


def export_chat_to_pdf(history: list[dict[str, Any]]) -> bytes:
    """
    Export chat history to a PDF file and return the bytes.

    Args:
        history: List of message dicts with 'role' and 'content' keys.

    Returns:
        PDF content as bytes.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "DocSensei - Chat History Export", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(6)

    for msg in history:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "")

        # Role header
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(40, 40, 60) if role == "Assistant" else pdf.set_fill_color(60, 60, 80)
        pdf.cell(0, 8, f"  {role}", ln=True, fill=True)

        # Content
        pdf.set_font("Helvetica", "", 10)
        # Strip markdown syntax for clean PDF output
        clean = re.sub(r'\*+', '', content)
        clean = re.sub(r'#+\s', '', clean)
        pdf.multi_cell(0, 6, clean)
        pdf.ln(3)

    return pdf.output()


# ─────────────────────────────────────────────
# Source Formatting
# ─────────────────────────────────────────────

def format_sources(source_docs) -> str:
    """
    Format retrieved source documents into a citation string.

    Args:
        source_docs: List of LangChain Document objects.

    Returns:
        Formatted markdown string listing unique sources.
    """
    seen: set[str] = set()
    lines: list[str] = []

    for doc in source_docs:
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page", "N/A")
        key = f"{source}::{page}"
        if key not in seen:
            seen.add(key)
            if page != "N/A":
                lines.append(f"• **{source}** (Page {page})")
            else:
                lines.append(f"• **{source}**")

    return "\n".join(lines) if lines else "• No sources available"


def search_chat_history(
    history: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    """
    Search through chat history for messages containing a query string.

    Args:
        history: List of message dicts.
        query: Search term (case-insensitive).

    Returns:
        Filtered list of matching message dicts.
    """
    query_lower = query.lower()
    return [
        msg for msg in history
        if query_lower in msg.get("content", "").lower()
    ]


def get_timestamp() -> str:
    """Return current timestamp as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_chars: int = 200) -> str:
    """
    Truncate text to a maximum character count, appending '…'.

    Args:
        text: Input string.
        max_chars: Maximum characters to keep.

    Returns:
        Truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
