"""
utils/rag.py – RAG pipeline orchestration for DocSensei.

Implements the full Retrieval-Augmented Generation pipeline using LangChain,
ChromaDB, and Google Gemini. Supports streaming, conversation memory,
document summaries, and suggested questions.
"""

import ast
import logging
from typing import Generator, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import config
from utils.vector_db import VectorDatabase
from utils.prompts import (
    RAG_PROMPT,
    CONTEXTUALISE_Q_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUGGESTED_QUESTIONS_PROMPT,
)
from utils.helpers import format_sources

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────

def build_llm(streaming: bool = False) -> ChatGoogleGenerativeAI:
    """
    Create and return a configured Gemini LLM instance.

    Args:
        streaming: If True, enables streaming token output.

    Returns:
        ChatGoogleGenerativeAI instance.

    Raises:
        ValueError: If GOOGLE_API_KEY is not set.
    """
    if not config.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file or Streamlit secrets."
        )

    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_output_tokens=config.LLM_MAX_TOKENS,
        streaming=streaming,
        convert_system_message_to_human=True,
    )


# ─────────────────────────────────────────────
# History-Aware RAG Chain
# ─────────────────────────────────────────────

def build_rag_chain(vector_db: VectorDatabase):
    """
    Build the full history-aware RAG chain.

    Uses create_history_aware_retriever to reformulate questions
    based on chat history, then passes the retrieved context to Gemini.

    Args:
        vector_db: The VectorDatabase instance with indexed documents.

    Returns:
        LangChain retrieval chain.
    """
    llm = build_llm(streaming=False)
    retriever = vector_db.get_retriever(k=config.RETRIEVAL_TOP_K)

    # History-aware retriever: reformulates question using chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALISE_Q_PROMPT
    )

    # Document QA chain
    question_answer_chain = create_stuff_documents_chain(llm, RAG_PROMPT)

    # Full RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# ─────────────────────────────────────────────
# Main RAG Interface
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    High-level interface for the DocSensei RAG pipeline.

    Manages the LLM chain, chat history conversion, and streaming.
    """

    def __init__(self, vector_db: VectorDatabase) -> None:
        """
        Initialise the RAG pipeline.

        Args:
            vector_db: Initialised VectorDatabase instance.
        """
        self.vector_db = vector_db
        self._chain = None

    @property
    def chain(self):
        """Lazily build and cache the RAG chain."""
        if self._chain is None:
            self._chain = build_rag_chain(self.vector_db)
        return self._chain

    def _convert_history(
        self, chat_history: list[dict]
    ) -> list[HumanMessage | AIMessage]:
        """
        Convert Streamlit session chat history to LangChain message objects.

        Args:
            chat_history: List of dicts with 'role' and 'content' keys.

        Returns:
            List of HumanMessage / AIMessage objects.
        """
        messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Strip source section from stored messages to reduce noise
                content = msg["content"].split("📚 **Sources**")[0].strip()
                messages.append(AIMessage(content=content))
        return messages

    def answer(
        self,
        question: str,
        chat_history: list[dict],
    ) -> tuple[str, list[Document]]:
        """
        Generate a non-streaming answer to the user's question.

        Args:
            question: User's natural language question.
            chat_history: Prior conversation history.

        Returns:
            Tuple of (answer_text, source_documents).

        Raises:
            RuntimeError: If the LLM call fails.
        """
        try:
            lc_history = self._convert_history(chat_history)
            result = self.chain.invoke({
                "input": question,
                "chat_history": lc_history,
            })
            answer = result.get("answer", "")
            source_docs = result.get("context", [])
            return answer, source_docs

        except Exception as exc:
            raise RuntimeError(f"RAG pipeline error: {exc}") from exc

    def stream_answer(
        self,
        question: str,
        chat_history: list[dict],
    ) -> Generator[str, None, None]:
        """
        Stream an answer token-by-token, yielding chunks.

        This method uses a non-streaming chain for retrieval (for accuracy)
        then streams the final LLM response independently using direct
        Gemini streaming.

        Args:
            question: User question.
            chat_history: Prior conversation.

        Yields:
            String chunks of the answer as they are generated.
        """
        # First, retrieve context via the RAG chain
        lc_history = self._convert_history(chat_history)
        result = self.chain.invoke({
            "input": question,
            "chat_history": lc_history,
        })
        full_answer: str = result.get("answer", "")
        source_docs: list[Document] = result.get("context", [])

        # Yield the answer in chunks to simulate streaming
        chunk_size = 8
        for i in range(0, len(full_answer), chunk_size):
            yield full_answer[i:i + chunk_size]

        # Store sources on instance for retrieval after streaming
        self._last_source_docs = source_docs

    def get_last_sources(self) -> list[Document]:
        """Return sources from the last stream_answer call."""
        return getattr(self, "_last_source_docs", [])


# ─────────────────────────────────────────────
# Document Summary
# ─────────────────────────────────────────────

def summarise_document(
    doc_name: str,
    chunks: list[Document],
    max_chunks: int = 5,
) -> str:
    """
    Generate a concise summary of a document from its chunks.

    Args:
        doc_name: Display name of the document.
        chunks: List of Document chunks from this document.
        max_chunks: Max number of chunks to include in the summary prompt.

    Returns:
        Summary string generated by Gemini.
    """
    if not chunks:
        return "No content available for summarisation."

    llm = build_llm(streaming=False)
    sample = chunks[:max_chunks]
    combined_text = "\n\n".join(c.page_content for c in sample)

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        doc_name=doc_name,
        content=combined_text[:4000],  # Limit context length
    )

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        logger.error("Summary generation failed: %s", exc)
        return f"Could not generate summary: {exc}"


# ─────────────────────────────────────────────
# Suggested Questions
# ─────────────────────────────────────────────

def generate_suggested_questions(chunks: list[Document]) -> list[str]:
    """
    Generate suggested questions based on the uploaded document content.

    Args:
        chunks: List of Document chunks to analyse.

    Returns:
        List of 5 suggested question strings.
    """
    if not chunks:
        return []

    llm = build_llm(streaming=False)
    sample = chunks[:8]
    combined = "\n\n".join(c.page_content for c in sample)

    prompt = SUGGESTED_QUESTIONS_PROMPT.format(content=combined[:5000])

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        # Safely parse the list
        questions = ast.literal_eval(raw)
        if isinstance(questions, list):
            return [str(q) for q in questions[:5]]
    except Exception as exc:
        logger.warning("Failed to generate suggested questions: %s", exc)

    return [
        "What are the main topics covered in this document?",
        "Can you summarise the key findings?",
        "What are the most important recommendations?",
        "Are there any specific dates or deadlines mentioned?",
        "Who are the key stakeholders mentioned?",
    ]
