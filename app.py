"""
app.py – DocSensei main Streamlit application.

Entry point for the DocSensei AI PDF & DOCX Question Answering System.
Orchestrates the sidebar, document processing pipeline, chat interface,
and all bonus features.
"""

import logging
import time
from pathlib import Path

import streamlit as st

import config
from utils.loaders import load_all_documents
from utils.splitter import split_documents, get_split_stats
from utils.vector_db import get_vector_db
from utils.rag import RAGPipeline, summarise_document, generate_suggested_questions
from utils.helpers import (
    save_uploaded_files,
    format_sources,
    export_chat_to_json,
    export_chat_to_pdf,
    search_chat_history,
    clear_chat_history,
    get_timestamp,
    cleanup_uploads,
)
from utils import ui

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"{config.APP_TITLE} – AI Document Q&A",
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/docsensei",
        "Report a bug": "https://github.com/your-org/docsensei/issues",
        "About": "DocSensei – AI-Powered Document Q&A powered by Gemini.",
    },
)


# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────
def init_session_state() -> None:
    """Initialise all required Streamlit session state variables."""
    defaults: dict = {
        "chat_history": [],          # List[{role, content, sources, timestamp}]
        "documents_processed": False,
        "processed_sources": [],     # File names successfully indexed
        "all_chunks": [],            # All document chunks (for summaries / questions)
        "suggested_questions": [],
        "rag_pipeline": None,
        "processing_error": None,
        "search_query": "",
        "theme": "dark",
        "chunk_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────
# API Key Validation
# ─────────────────────────────────────────────
def validate_api_key() -> bool:
    """
    Check for GOOGLE_API_KEY in config or Streamlit secrets.

    Returns:
        True if a key is available, False otherwise.
    """
    # Check .env / environment
    if config.GOOGLE_API_KEY:
        return True
    # Check Streamlit secrets (for Community Cloud deployment)
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
        if key:
            config.GOOGLE_API_KEY = key
            return True
    except Exception:
        pass
    return False


# ─────────────────────────────────────────────
# Document Processing Pipeline
# ─────────────────────────────────────────────
def process_documents(uploaded_files: list) -> None:
    """
    Run the full document ingestion pipeline:
      1. Save uploaded files to disk
      2. Load & validate documents
      3. Split into chunks
      4. Embed and store in ChromaDB

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
    """
    if not uploaded_files:
        ui.render_alert("Please upload at least one file before processing.", "warning")
        return

    st.session_state.processing_error = None

    with st.status("⚙️ Processing documents…", expanded=True) as status:
        # Step 1 – Save
        st.write("💾 Saving uploaded files…")
        saved_paths, save_errors = save_uploaded_files(uploaded_files)

        if save_errors:
            for err in save_errors:
                st.warning(f"⚠️ {err}")

        if not saved_paths:
            status.update(label="❌ No files saved.", state="error")
            return

        # Step 2 – Load
        st.write("📖 Loading and extracting text…")
        progress = st.progress(0, text="Loading…")

        def update_progress(current: int, total: int, name: str) -> None:
            pct = int((current / max(total, 1)) * 100)
            progress.progress(pct, text=f"Loading: {name}")

        docs, load_errors = load_all_documents(saved_paths, progress_callback=update_progress)

        if load_errors:
            for err in load_errors:
                st.warning(f"⚠️ {err}")

        if not docs:
            status.update(label="❌ No documents could be loaded.", state="error")
            return

        st.write(f"✅ Loaded **{len(docs)}** document pages/sections.")

        # Step 3 – Split
        st.write("✂️ Splitting text into chunks…")
        try:
            chunks = split_documents(docs)
            stats = get_split_stats(chunks)
            st.write(f"✅ Created **{len(chunks)}** text chunks.")
        except ValueError as exc:
            status.update(label=f"❌ Splitting failed: {exc}", state="error")
            return

        # Step 4 – Embed & Store
        st.write("🧠 Generating embeddings and storing in ChromaDB…")
        try:
            vdb = get_vector_db()
            vdb.add_documents(chunks)
            st.write(f"✅ Stored **{len(chunks)}** vectors in ChromaDB.")
        except (RuntimeError, ValueError) as exc:
            status.update(label=f"❌ Vector DB error: {exc}", state="error")
            st.session_state.processing_error = str(exc)
            return

        # Update session state
        st.session_state.documents_processed = True
        st.session_state.all_chunks = chunks
        st.session_state.chunk_count = len(chunks)
        st.session_state.processed_sources = list(stats.keys())

        # Build RAG pipeline
        st.session_state.rag_pipeline = RAGPipeline(vdb)

        # Generate suggested questions in background
        st.write("💡 Generating suggested questions…")
        st.session_state.suggested_questions = generate_suggested_questions(chunks)

        status.update(
            label=f"✅ {len(saved_paths)} document(s) processed successfully!",
            state="complete",
        )

    st.balloons()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar() -> None:
    """Render the complete sidebar with logo, upload, and controls."""
    with st.sidebar:
        ui.render_sidebar_logo()

        # ── API Key Status ──
        if not validate_api_key():
            st.error(
                "🔑 **GOOGLE_API_KEY** is missing.\n\n"
                "Add it to `.env` or Streamlit secrets to continue."
            )

        # ── File Upload ──
        st.markdown("### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            label="PDF / DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help=f"Max {config.MAX_FILE_SIZE_MB} MB per file. Supported: PDF, DOCX",
            key="file_uploader",
            label_visibility="collapsed",
        )

        st.markdown(
            f"<div style='font-size:0.75rem; color:#5a6070; text-align:center; margin-top:0.3rem;'>"
            f"Max {config.MAX_FILE_SIZE_MB} MB · PDF & DOCX only</div>",
            unsafe_allow_html=True,
        )

        if uploaded_files:
            st.markdown(
                f"<div style='color:#4caf7d; font-size:0.82rem; margin:0.4rem 0;'>"
                f"📎 {len(uploaded_files)} file(s) selected</div>",
                unsafe_allow_html=True,
            )
            for f in uploaded_files:
                size_kb = len(f.getbuffer()) / 1024
                st.markdown(
                    f"<div style='font-size:0.78rem; color:#9ea3c0; padding:2px 0;'>"
                    f"• {f.name} ({size_kb:.1f} KB)</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # ── Process Button ──
        if st.button("⚡ Process Documents", type="primary", use_container_width=True):
            if not validate_api_key():
                st.error("Set your GOOGLE_API_KEY first.")
            elif uploaded_files:
                process_documents(uploaded_files)
            else:
                st.warning("Upload files first.")

        ui.render_divider()

        # ── Status ──
        st.markdown("### 📊 Status")
        vdb = get_vector_db()
        chunk_count = st.session_state.chunk_count
        ui.render_status_badge(st.session_state.documents_processed, chunk_count)

        if st.session_state.processed_sources:
            st.markdown("")
            st.markdown(
                f"<div style='font-size:0.8rem; color:#9ea3c0;'>"
                f"<strong>Indexed Documents:</strong></div>",
                unsafe_allow_html=True,
            )
            for src in st.session_state.processed_sources:
                icon = "📄" if src.endswith(".pdf") else "📝"
                st.markdown(
                    f"<div style='font-size:0.78rem; color:#6c63ff; padding:2px 0;'>"
                    f"{icon} {src}</div>",
                    unsafe_allow_html=True,
                )

        ui.render_divider()

        # ── Chat Controls ──
        st.markdown("### 💬 Chat Controls")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                clear_chat_history()
                st.rerun()

        with col2:
            if st.button("🗄️ Clear DB", use_container_width=True):
                try:
                    vdb.clear()
                    st.session_state.documents_processed = False
                    st.session_state.processed_sources = []
                    st.session_state.all_chunks = []
                    st.session_state.chunk_count = 0
                    st.session_state.rag_pipeline = None
                    st.session_state.suggested_questions = []
                    cleanup_uploads()
                    st.success("Vector DB cleared!")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(f"Failed to clear: {exc}")

        ui.render_divider()

        # ── Export Controls ──
        st.markdown("### 📥 Export Chat")

        if st.session_state.chat_history:
            json_data = export_chat_to_json(st.session_state.chat_history)
            st.download_button(
                label="⬇️ Download as JSON",
                data=json_data,
                file_name=f"docsensei_chat_{get_timestamp().replace(':', '-').replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )

            pdf_bytes = export_chat_to_pdf(st.session_state.chat_history)
            st.download_button(
                label="⬇️ Download as PDF",
                data=bytes(pdf_bytes),
                file_name=f"docsensei_chat_{get_timestamp().replace(':', '-').replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.markdown(
                "<div style='font-size:0.78rem; color:#5a6070;'>No chat history to export.</div>",
                unsafe_allow_html=True,
            )

        ui.render_divider()

        # ── Search Chat History ──
        st.markdown("### 🔍 Search History")
        search_q = st.text_input(
            "Search in chat…",
            placeholder="Type to search…",
            label_visibility="collapsed",
            key="search_input",
        )
        if search_q:
            matches = search_chat_history(st.session_state.chat_history, search_q)
            st.markdown(
                f"<div style='font-size:0.78rem; color:#9ea3c0;'>"
                f"Found {len(matches)} match(es)</div>",
                unsafe_allow_html=True,
            )
            for m in matches[:5]:
                role_icon = "🧑" if m["role"] == "user" else "🤖"
                st.markdown(
                    f"<div style='font-size:0.75rem; color:#9ea3c0; "
                    f"background:#1a1e36; border-radius:6px; padding:0.4rem 0.6rem; margin:0.2rem 0;'>"
                    f"{role_icon} {m['content'][:80]}…</div>",
                    unsafe_allow_html=True,
                )

        # ── Footer ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center; font-size:0.68rem; color:#3a3f55;'>"
            "DocSensei v1.0 · Built with ❤️ using Gemini & LangChain"
            "</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# Chat History Renderer
# ─────────────────────────────────────────────
def render_chat_history() -> None:
    """Render all messages in chat history using custom bubble components."""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            ui.render_user_message(msg["content"])
        else:
            sources_str = msg.get("sources", "")
            # Split the stored combined answer to separate sources if present
            content = msg["content"]
            ui.render_assistant_message(content, sources_str)


# ─────────────────────────────────────────────
# Document Summaries Tab
# ─────────────────────────────────────────────
def render_summaries_tab() -> None:
    """Render document summaries for all processed documents."""
    if not st.session_state.documents_processed:
        ui.render_empty_state()
        return

    chunks = st.session_state.all_chunks
    sources = st.session_state.processed_sources

    for source in sources:
        source_chunks = [c for c in chunks if c.metadata.get("source") == source]
        with st.expander(f"📄 {source} — {len(source_chunks)} chunks", expanded=False):
            with st.spinner(f"Generating summary for {source}…"):
                summary = summarise_document(source, source_chunks)
            st.markdown(summary)


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main() -> None:
    """Main application entry point."""
    init_session_state()
    ui.inject_css()

    # ── Sidebar ──
    render_sidebar()

    # ── Main Content Tabs ──
    tab_chat, tab_summaries, tab_about = st.tabs([
        "💬 Chat",
        "📄 Document Summaries",
        "ℹ️ About",
    ])

    # ─── CHAT TAB ───
    with tab_chat:
        ui.render_hero_header()
        ui.render_divider()

        # Stats row
        if st.session_state.documents_processed:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                ui.render_stat_card(len(st.session_state.processed_sources), "Documents")
            with c2:
                ui.render_stat_card(st.session_state.chunk_count, "Chunks")
            with c3:
                ui.render_stat_card(len(st.session_state.chat_history) // 2, "Exchanges")
            with c4:
                ui.render_stat_card("Gemini 2.5", "LLM")
            st.markdown("")

        # Suggested questions
        if (
            st.session_state.documents_processed
            and st.session_state.suggested_questions
            and not st.session_state.chat_history
        ):
            clicked = ui.render_suggested_questions(st.session_state.suggested_questions)
            if clicked:
                st.session_state["_pending_question"] = clicked
                st.rerun()

        # Empty state
        if not st.session_state.documents_processed:
            ui.render_empty_state()
            ui.render_welcome_features()

        # Chat history
        if st.session_state.chat_history:
            render_chat_history()

        ui.render_divider()

        # ── Chat Input ──
        question = st.chat_input(
            placeholder="Ask anything about your documents…",
            disabled=not st.session_state.documents_processed,
        )

        # Handle pending question from suggested questions click
        if "_pending_question" in st.session_state:
            question = st.session_state.pop("_pending_question")

        if question:
            if not validate_api_key():
                ui.render_alert("Set your GOOGLE_API_KEY to use the chat.", "error")
            elif not st.session_state.documents_processed:
                ui.render_alert("Process documents first before asking questions.", "warning")
            elif st.session_state.rag_pipeline is None:
                ui.render_alert("RAG pipeline not initialised. Please reprocess documents.", "error")
            else:
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question,
                    "timestamp": get_timestamp(),
                })
                ui.render_user_message(question)

                # Generate answer with streaming effect
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("""
                <div class="msg-wrapper assistant">
                    <div class="avatar assistant">🤖</div>
                    <div class="thinking-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                try:
                    pipeline: RAGPipeline = st.session_state.rag_pipeline

                    # Get answer
                    answer_text, source_docs = pipeline.answer(
                        question=question,
                        chat_history=st.session_state.chat_history[:-1],  # exclude current
                    )

                    thinking_placeholder.empty()

                    # Format sources
                    sources_str = format_sources(source_docs)

                    # Stream the answer character by character in the bubble
                    answer_placeholder = st.empty()
                    displayed = ""
                    for chunk in answer_text:
                        displayed += chunk
                        answer_placeholder.markdown(f"""
                        <div class="msg-wrapper assistant">
                            <div class="avatar assistant">🤖</div>
                            <div class="bubble assistant">{displayed}▌</div>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.005)

                    # Final render with sources
                    answer_placeholder.empty()
                    ui.render_assistant_message(answer_text, sources_str)

                    # Store in history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_str,
                        "timestamp": get_timestamp(),
                    })

                    # Trim history
                    if len(st.session_state.chat_history) > config.MAX_CHAT_HISTORY:
                        st.session_state.chat_history = st.session_state.chat_history[
                            -config.MAX_CHAT_HISTORY:
                        ]

                except RuntimeError as exc:
                    thinking_placeholder.empty()
                    ui.render_alert(f"Error generating answer: {exc}", "error")
                    logger.error("RAG error: %s", exc)

    # ─── SUMMARIES TAB ───
    with tab_summaries:
        st.markdown("## 📄 Document Summaries")
        st.markdown(
            "<div style='color:#9ea3c0; font-size:0.88rem;'>"
            "AI-generated summaries of your uploaded documents.</div>",
            unsafe_allow_html=True,
        )
        ui.render_divider()
        render_summaries_tab()

    # ─── ABOUT TAB ───
    with tab_about:
        st.markdown("## ℹ️ About DocSensei")
        ui.render_divider()
        st.markdown("""
**DocSensei** is an AI-powered document question-answering system built with:

| Component | Technology |
|-----------|-----------|
| 🤖 LLM | Google Gemini 2.5 Flash |
| 🔍 Embeddings | Google Generative AI Embeddings |
| 🗄️ Vector DB | ChromaDB (persistent) |
| 🔗 Framework | LangChain |
| 🎨 Frontend | Streamlit |

### RAG Pipeline
```
Upload Files → Load → Extract Text → Split Chunks
→ Generate Embeddings → Store in ChromaDB
→ User Question → Similarity Search → Retrieve Top-K
→ Gemini LLM → Answer + Citations
```

### Features
- ✅ Multi-document PDF & DOCX support
- ✅ Semantic similarity search
- ✅ Conversation memory
- ✅ Page-level citations
- ✅ Document summaries
- ✅ Suggested questions
- ✅ Chat export (JSON & PDF)
- ✅ Chat history search
- ✅ Streaming responses
- ✅ Persistent ChromaDB

---
*Built with ❤️ using Python, LangChain, and Google Gemini.*
        """)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
