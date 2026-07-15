"""
utils/loaders.py – Document loaders for PDF and DOCX files.

Handles loading, validation, and text extraction from uploaded documents.
"""

import hashlib
import logging
import warnings
from pathlib import Path
from typing import Optional

# langchain-community emits a sunset DeprecationWarning at import time;
# suppress it since PyPDFLoader / Docx2txtLoader still work correctly.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# File Validation
# ─────────────────────────────────────────────

def validate_file(file_path: str) -> tuple[bool, str]:
    """
    Validate an uploaded file for type, size, and readability.

    Args:
        file_path: Absolute path to the file.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    path = Path(file_path)

    # Check existence
    if not path.exists():
        return False, f"File does not exist: {path.name}"

    # Check extension
    if path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{path.suffix}'. "
            f"Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )

    # Check file size
    size = path.stat().st_size
    if size == 0:
        return False, f"File '{path.name}' is empty."
    if size > config.MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        return False, (
            f"File '{path.name}' is too large ({mb:.1f} MB). "
            f"Maximum allowed: {config.MAX_FILE_SIZE_MB} MB."
        )

    return True, ""


def compute_file_hash(file_path: str) -> str:
    """
    Compute an MD5 hash of a file for duplicate detection.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest string.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def load_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF file and return a list of Document objects (one per page).

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects with page_content and metadata.

    Raises:
        ValueError: If the PDF cannot be loaded or is corrupted.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Enrich metadata
        file_name = Path(file_path).name
        for doc in documents:
            doc.metadata["source"] = file_name
            doc.metadata["file_path"] = file_path
            # PyPDFLoader sets page as 0-indexed; convert to 1-indexed
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1

        logger.info("Loaded PDF '%s': %d pages", file_name, len(documents))
        return documents

    except Exception as exc:
        raise ValueError(f"Failed to load PDF '{Path(file_path).name}': {exc}") from exc


def load_docx(file_path: str) -> list[Document]:
    """
    Load a DOCX file and return a list of Document objects.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        List containing a single Document with all text content.

    Raises:
        ValueError: If the DOCX cannot be loaded or is corrupted.
    """
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

        file_name = Path(file_path).name
        for doc in documents:
            doc.metadata["source"] = file_name
            doc.metadata["file_path"] = file_path
            doc.metadata["page"] = "N/A"

        logger.info("Loaded DOCX '%s'", file_name)
        return documents

    except Exception as exc:
        raise ValueError(
            f"Failed to load DOCX '{Path(file_path).name}': {exc}"
        ) from exc


def load_document(file_path: str) -> list[Document]:
    """
    Dispatch to the appropriate loader based on file extension.

    Args:
        file_path: Path to a PDF or DOCX file.

    Returns:
        List of Document objects.

    Raises:
        ValueError: For unsupported extensions or load failures.
    """
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)
    elif extension == ".docx":
        return load_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported: {', '.join(config.ALLOWED_EXTENSIONS)}"
        )


def load_all_documents(
    file_paths: list[str],
    progress_callback: Optional[callable] = None,
) -> tuple[list[Document], list[str]]:
    """
    Load multiple documents, skipping duplicates and invalid files.

    Args:
        file_paths: List of file paths to load.
        progress_callback: Optional callable(current, total, file_name) for progress.

    Returns:
        Tuple of (all_documents, error_messages).
    """
    all_documents: list[Document] = []
    errors: list[str] = []
    seen_hashes: set[str] = set()

    for idx, file_path in enumerate(file_paths):
        file_name = Path(file_path).name

        if progress_callback:
            progress_callback(idx, len(file_paths), file_name)

        # Validate
        is_valid, err_msg = validate_file(file_path)
        if not is_valid:
            errors.append(err_msg)
            logger.warning("Skipping invalid file: %s", err_msg)
            continue

        # Duplicate check
        file_hash = compute_file_hash(file_path)
        if file_hash in seen_hashes:
            errors.append(f"Duplicate file skipped: '{file_name}'")
            logger.warning("Duplicate file skipped: %s", file_name)
            continue
        seen_hashes.add(file_hash)

        # Load
        try:
            docs = load_document(file_path)
            all_documents.extend(docs)
        except ValueError as exc:
            errors.append(str(exc))
            logger.error("Error loading %s: %s", file_name, exc)

    if progress_callback:
        progress_callback(len(file_paths), len(file_paths), "Done")

    return all_documents, errors
