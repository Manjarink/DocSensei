"""
utils/rag.py – RAG pipeline orchestration for DocSensei.

Implements the full Retrieval-Augmented Generation pipeline using
langchain_core (LCEL), ChromaDB, and Google Gemini.
Compatible with LangChain 1.x (no langchain.chains dependency).
"""

import ast
import logging
from typing import Generator, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from utils.vector_db import VectorDatabase
from utils.prompts import (
    RAG_PROMPT,
    CONTEXTUALISE_Q_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUGGESTED_QUESTIONS_PROMPT,
)

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
# Main RAG Interface (LCEL-based, no langchain.chains)
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    High-level interface for the DocSensei RAG pipeline.

    Built entirely with langchain_core LCEL — no dependency on
    the deprecated langchain.chains module.
    """

    def __init__(self, vector_db: VectorDatabase) -> None:
        """
        Initialise the RAG pipeline.

        Args:
            vector_db: Initialised VectorDatabase instance.
        """
        self.vector_db = vector_db
        self._last_source_docs: list[Document] = []

    # ── Internal helpers ──

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
        messages: list[HumanMessage | AIMessage] = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Strip source section to reduce noise in context
                content = msg["content"].split("📚 **Sources**")[0].strip()
                messages.append(AIMessage(content=content))
        return messages

    def _reformulate_question(
        self,
        question: str,
        lc_history: list,
    ) -> str:
        """
        Reformulate the question as a standalone query using chat history.

        If there is no prior history the original question is returned as-is.

        Args:
            question: The user's raw question.
            lc_history: LangChain message history list.

        Returns:
            Standalone question string.
        """
        if not lc_history:
            return question

        try:
            llm = build_llm()
            chain = CONTEXTUALISE_Q_PROMPT | llm | StrOutputParser()
            return chain.invoke({
                "input": question,
                "chat_history": lc_history,
            })
        except Exception as exc:
            logger.warning("Question reformulation failed, using original: %s", exc)
            return question

    def _build_context(self, docs: list[Document]) -> str:
        """
        Concatenate document chunks into a single context string.

        Args:
            docs: Retrieved Document objects.

        Returns:
            Newline-separated page content string.
        """
        return "\n\n".join(
            f"[Source: {d.metadata.get('source', 'Unknown')}, "
            f"Page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
            for d in docs
        )

    # ── Public interface ──

    def answer(
        self,
        question: str,
        chat_history: list[dict],
    ) -> tuple[str, list[Document]]:
        """
        Generate a complete answer to the user's question.

        Pipeline:
          1. Convert chat history to LangChain messages
          2. Reformulate question as standalone if history exists
          3. Retrieve top-k similar chunks from ChromaDB
          4. Build context string from retrieved chunks
          5. Invoke Gemini with the RAG prompt
          6. Return answer text + source documents

        Args:
            question: User's natural language question.
            chat_history: Prior conversation history (Streamlit format).

        Returns:
            Tuple of (answer_text, source_documents).

        Raises:
            RuntimeError: If the LLM call or retrieval fails.
        """
        try:
            lc_history = self._convert_history(chat_history)

            # Step 1: Reformulate to standalone question
            standalone_q = self._reformulate_question(question, lc_history)
            logger.info("Standalone question: %s", standalone_q[:100])

            # Step 2: Retrieve relevant chunks
            source_docs = self.vector_db.similarity_search(
                standalone_q, k=config.RETRIEVAL_TOP_K
            )
            logger.info("Retrieved %d chunks.", len(source_docs))

            # Step 3: Build context
            context = self._build_context(source_docs)

            # Step 4: Invoke LLM via RAG prompt
            llm = build_llm()
            prompt_messages = RAG_PROMPT.format_messages(
                context=context,
                chat_history=lc_history,
                input=question,
            )
            response = llm.invoke(prompt_messages)
            answer_text = response.content if hasattr(response, "content") else str(response)

            self._last_source_docs = source_docs
            return answer_text, source_docs

        except Exception as exc:
            raise RuntimeError(f"RAG pipeline error: {exc}") from exc

    def stream_answer(
        self,
        question: str,
        chat_history: list[dict],
    ) -> Generator[str, None, None]:
        """
        Stream the answer character-by-character.

        Retrieves the full answer first (for accuracy), then yields it
        in small chunks to produce a streaming effect in the UI.

        Args:
            question: User question.
            chat_history: Prior conversation.

        Yields:
            String chunks of the answer.
        """
        full_answer, source_docs = self.answer(question, chat_history)
        self._last_source_docs = source_docs

        chunk_size = 8
        for i in range(0, len(full_answer), chunk_size):
            yield full_answer[i: i + chunk_size]

    def get_last_sources(self) -> list[Document]:
        """Return source documents from the last answer() / stream_answer() call."""
        return self._last_source_docs


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

    llm = build_llm()
    sample = chunks[:max_chunks]
    combined_text = "\n\n".join(c.page_content for c in sample)

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        doc_name=doc_name,
        content=combined_text[:4000],
    )

    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
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
        List of up to 5 suggested question strings.
    """
    if not chunks:
        return []

    llm = build_llm()
    sample = chunks[:8]
    combined = "\n\n".join(c.page_content for c in sample)

    prompt = SUGGESTED_QUESTIONS_PROMPT.format(content=combined[:5000])

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()
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
