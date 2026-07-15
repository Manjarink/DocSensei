"""
DocSensei Configuration Module.

Centralizes all application settings and constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# Base Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ─────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────
LLM_MODEL: str = "gemini-2.5-flash"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 8192

# ─────────────────────────────────────────────
# Embedding Configuration
# ─────────────────────────────────────────────
EMBEDDING_MODEL: str = "models/text-embedding-004"

# ─────────────────────────────────────────────
# Text Splitting Configuration
# ─────────────────────────────────────────────
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]

# ─────────────────────────────────────────────
# Vector Database Configuration
# ─────────────────────────────────────────────
CHROMA_COLLECTION_NAME: str = "docsensei_collection"
CHROMA_PERSIST_DIR: str = str(VECTORSTORE_DIR)

# ─────────────────────────────────────────────
# Retrieval Configuration
# ─────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 6
SIMILARITY_THRESHOLD: float = 0.3

# ─────────────────────────────────────────────
# File Upload Configuration
# ─────────────────────────────────────────────
MAX_FILE_SIZE_MB: int = 50
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: list[str] = [".pdf", ".docx"]

# ─────────────────────────────────────────────
# UI Configuration
# ─────────────────────────────────────────────
APP_TITLE: str = "DocSensei"
APP_SUBTITLE: str = "AI-Powered PDF & DOCX Question Answering"
APP_ICON: str = "📄"
PAGE_LAYOUT: str = "wide"

# ─────────────────────────────────────────────
# Chat Configuration
# ─────────────────────────────────────────────
MAX_CHAT_HISTORY: int = 50
CHAT_HISTORY_FILE: str = "chat_history.json"
