"""
utils/splitter.py – Text splitting utilities for DocSensei.

Splits loaded documents into optimal chunks for embedding and retrieval.
"""

import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)


def build_splitter(
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    separators: list[str] = config.SEPARATORS,
) -> RecursiveCharacterTextSplitter:
    """
    Create and return a configured RecursiveCharacterTextSplitter.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        separators: Priority-ordered list of split separators.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks while preserving metadata.

    Each chunk retains the source, page, and file_path from the parent document,
    plus a chunk_index field for ordering.

    Args:
        documents: List of loaded Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunked Document objects.

    Raises:
        ValueError: If the documents list is empty.
    """
    if not documents:
        raise ValueError("No documents provided for splitting.")

    # Filter out empty or whitespace-only documents (e.g., scanned PDFs without OCR)
    valid_documents = [d for d in documents if d.page_content and d.page_content.strip()]
    if not valid_documents:
        raise ValueError(
            "No readable text could be extracted from the uploaded document(s). "
            "Please check if the file is a scanned image-only PDF, empty, or password-protected."
        )

    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(valid_documents)

    if not chunks:
        raise ValueError(
            "Splitting produced zero chunks. The extracted document text may be too short."
        )

    # Add chunk index metadata per source
    source_chunk_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        source_chunk_counters[source] = source_chunk_counters.get(source, 0) + 1
        chunk.metadata["chunk_index"] = source_chunk_counters[source]

    logger.info(
        "Split %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def get_split_stats(chunks: list[Document]) -> dict[str, int]:
    """
    Return per-source chunk counts for diagnostic purposes.

    Args:
        chunks: List of chunked Document objects.

    Returns:
        Dictionary mapping source file name to chunk count.
    """
    stats: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        stats[source] = stats.get(source, 0) + 1
    return stats
