"""
utils/embeddings.py – Embedding model setup for DocSensei.

Initialises the Google Generative AI embedding model used to convert
text chunks into vector representations.
"""

import logging
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """
    Return a cached instance of the Google Generative AI embedding model.

    The model is created once and reused across the application to avoid
    redundant initialisation overhead.

    Returns:
        GoogleGenerativeAIEmbeddings instance.

    Raises:
        ValueError: If GOOGLE_API_KEY is not configured.
    """
    if not config.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Please add it to your .env file or Streamlit secrets."
        )

    logger.info("Initialising embedding model: %s", config.EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
