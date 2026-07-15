"""
utils/vector_db.py – ChromaDB vector store management for DocSensei.

Handles creating, loading, updating, and querying the ChromaDB vector store.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config
from utils.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class VectorDatabase:
    """
    Manages the ChromaDB vector store for DocSensei.

    Provides methods to add documents, query for similar chunks,
    and manage the lifecycle of the vector store.
    """

    def __init__(self) -> None:
        """Initialise the VectorDatabase with lazy loading."""
        self._store: Optional[Chroma] = None
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Lazily initialise and return the embedding model."""
        if self._embeddings is None:
            self._embeddings = get_embedding_model()
        return self._embeddings

    @property
    def store(self) -> Chroma:
        """Lazily load or create the ChromaDB store."""
        if self._store is None:
            self._store = self._load_or_create_store()
        return self._store

    def _load_or_create_store(self) -> Chroma:
        """
        Load an existing ChromaDB store or create a new one.

        Returns:
            Chroma vector store instance.
        """
        logger.info(
            "Loading ChromaDB from '%s', collection='%s'",
            config.CHROMA_PERSIST_DIR,
            config.CHROMA_COLLECTION_NAME,
        )
        return Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )

    def add_documents(self, chunks: list[Document]) -> None:
        """
        Embed and add document chunks to the vector store.

        Args:
            chunks: List of Document chunks to add.

        Raises:
            ValueError: If chunks list is empty.
            RuntimeError: If ChromaDB write fails.
        """
        if not chunks:
            raise ValueError("No chunks provided to add to vector database.")

        try:
            self.store.add_documents(chunks)
            logger.info("Added %d chunks to ChromaDB.", len(chunks))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to add documents to ChromaDB: {exc}"
            ) from exc

    def similarity_search(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
    ) -> list[Document]:
        """
        Perform a similarity search against the vector store.

        Args:
            query: User's natural language question.
            k: Number of top results to return.

        Returns:
            List of the most similar Document chunks.

        Raises:
            RuntimeError: If the query fails.
        """
        try:
            results = self.store.similarity_search(query, k=k)
            logger.info(
                "Similarity search for '%s' returned %d results.", query[:60], len(results)
            )
            return results
        except Exception as exc:
            raise RuntimeError(f"Vector similarity search failed: {exc}") from exc

    def similarity_search_with_score(
        self,
        query: str,
        k: int = config.RETRIEVAL_TOP_K,
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search and return documents with relevance scores.

        Args:
            query: Natural language query.
            k: Number of results.

        Returns:
            List of (Document, score) tuples.
        """
        try:
            return self.store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as exc:
            raise RuntimeError(
                f"Scored similarity search failed: {exc}"
            ) from exc

    def get_retriever(self, k: int = config.RETRIEVAL_TOP_K):
        """
        Return a LangChain retriever interface for the vector store.

        Args:
            k: Number of documents to retrieve.

        Returns:
            VectorStoreRetriever instance.
        """
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def get_document_count(self) -> int:
        """
        Return the number of documents currently in the vector store.

        Returns:
            Integer count of stored vectors.
        """
        try:
            collection = self.store._collection  # type: ignore[attr-defined]
            return collection.count()
        except Exception:
            return 0

    def list_sources(self) -> list[str]:
        """
        Return unique source file names stored in the vector database.

        Returns:
            Sorted list of unique source document names.
        """
        try:
            collection = self.store._collection  # type: ignore[attr-defined]
            results = collection.get(include=["metadatas"])
            sources = set()
            for meta in results.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(sources)
        except Exception as exc:
            logger.warning("Could not list sources: %s", exc)
            return []

    def clear(self) -> None:
        """
        Delete all documents from the vector store and reset state.

        This removes the persisted ChromaDB data from disk.
        """
        try:
            persist_path = Path(config.CHROMA_PERSIST_DIR)
            if persist_path.exists():
                shutil.rmtree(persist_path)
                persist_path.mkdir(exist_ok=True)
                logger.info("ChromaDB cleared at '%s'.", persist_path)

            self._store = None  # Reset the cached store
        except Exception as exc:
            raise RuntimeError(f"Failed to clear vector database: {exc}") from exc

    def is_empty(self) -> bool:
        """Check whether the vector store contains any documents."""
        return self.get_document_count() == 0


# ─────────────────────────────────────────────
# Singleton instance
# ─────────────────────────────────────────────
_vector_db: Optional[VectorDatabase] = None


def get_vector_db() -> VectorDatabase:
    """
    Return the global VectorDatabase singleton.

    Returns:
        VectorDatabase instance.
    """
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDatabase()
    return _vector_db
