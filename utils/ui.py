"""
utils/ui.py – Streamlit UI components and custom CSS for DocSensei.

Contains all reusable UI components, styling, and rendering helpers
to keep app.py clean and declarative.
"""

import streamlit as st


# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0d0f1a;
    --bg-secondary: #13162b;
    --bg-card: #1a1e36;
    --bg-sidebar: #10132280;
    --accent-primary: #6c63ff;
    --accent-secondary: #7c73ff;
    --accent-glow: rgba(108, 99, 255, 0.35);
    --text-primary: #e8eaf6;
    --text-secondary: #9ea3c0;
    --text-muted: #5a6070;
    --border-color: rgba(108, 99, 255, 0.25);
    --user-bubble: linear-gradient(135deg, #6c63ff, #8b80ff);
    --ai-bubble: #1a1e36;
    --success: #4caf7d;
    --error: #f44336;
    --warning: #ff9800;
    --radius: 16px;
    --radius-sm: 8px;
}

/* ── Global Reset ── */
.stApp {
    background-color: var(--bg-primary);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* ── Hide default Streamlit header/footer ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-primary); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f1a 0%, #13162b 100%);
    border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary);
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    border-radius: var(--radius-sm);
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.5rem 1rem;
    transition: all 0.25s ease;
    border: 1px solid var(--border-color);
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: white;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px var(--accent-glow);
    background: linear-gradient(135deg, #7c73ff, #9b91ff);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 2px dashed var(--border-color);
    border-radius: var(--radius);
    padding: 1rem;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-primary);
}

/* ── Chat Container ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 1rem 0;
    max-height: 65vh;
    overflow-y: auto;
    scrollbar-width: thin;
}

/* ── Message Bubbles ── */
.msg-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
    animation: fadeSlideIn 0.3s ease forwards;
}
.msg-wrapper.user { flex-direction: row-reverse; }
.msg-wrapper.assistant { flex-direction: row; }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.avatar.user { background: var(--user-bubble); }
.avatar.assistant { background: var(--bg-card); border: 1px solid var(--border-color); }

.bubble {
    max-width: 78%;
    padding: 0.9rem 1.2rem;
    border-radius: var(--radius);
    font-size: 0.92rem;
    line-height: 1.6;
    word-wrap: break-word;
}
.bubble.user {
    background: var(--user-bubble);
    color: white;
    border-bottom-right-radius: 4px;
}
.bubble.assistant {
    background: var(--ai-bubble);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    border-bottom-left-radius: 4px;
}

/* ── Sources section ── */
.sources-block {
    margin-top: 0.75rem;
    padding: 0.75rem 1rem;
    background: rgba(108, 99, 255, 0.08);
    border-left: 3px solid var(--accent-primary);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 0.82rem;
    color: var(--text-secondary);
}

/* ── Hero Header ── */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
    letter-spacing: -1px;
}
.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 0.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(108, 99, 255, 0.15);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    color: var(--accent-secondary);
    letter-spacing: 0.5px;
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}
.status-badge.ready { background: rgba(76, 175, 125, 0.15); color: #4caf7d; border: 1px solid rgba(76, 175, 125, 0.3); }
.status-badge.empty { background: rgba(255, 152, 0, 0.12); color: #ff9800; border: 1px solid rgba(255, 152, 0, 0.3); }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-color), transparent);
    margin: 1rem 0;
}

/* ── Stat Card ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1rem;
    text-align: center;
    transition: transform 0.2s ease;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-number { font-size: 1.4rem; font-weight: 700; color: var(--accent-primary); }
.stat-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Thinking indicator ── */
.thinking-dots {
    display: inline-flex;
    gap: 4px;
    padding: 0.9rem 1.2rem;
    background: var(--ai-bubble);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    border-bottom-left-radius: 4px;
}
.thinking-dots span {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-primary);
    animation: bounce 1.2s infinite ease-in-out;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Input box ── */
.stTextInput > div > div > input,
.stChatInput > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.stChatInput > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-primary), #a78bfa) !important;
    border-radius: 4px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0.5rem 1rem;
}
.sidebar-logo .logo-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.4rem;
    filter: drop-shadow(0 0 12px var(--accent-primary));
}
.sidebar-logo .logo-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6c63ff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-logo .logo-tagline {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Suggested question chip ── */
.q-chip {
    display: inline-block;
    background: rgba(108, 99, 255, 0.1);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
    margin: 0.25rem;
}
.q-chip:hover {
    background: rgba(108, 99, 255, 0.25);
    color: var(--text-primary);
    border-color: var(--accent-primary);
}

/* ── Alert boxes ── */
.alert {
    padding: 0.8rem 1rem;
    border-radius: var(--radius-sm);
    font-size: 0.88rem;
    margin-bottom: 0.5rem;
}
.alert-success { background: rgba(76, 175, 125, 0.12); border-left: 3px solid #4caf7d; color: #4caf7d; }
.alert-error   { background: rgba(244, 67, 54, 0.12); border-left: 3px solid #f44336; color: #f44336; }
.alert-warning { background: rgba(255, 152, 0, 0.12); border-left: 3px solid #ff9800; color: #ff9800; }
.alert-info    { background: rgba(108, 99, 255, 0.1); border-left: 3px solid var(--accent-primary); color: var(--accent-secondary); }

/* ── Metrics override ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1rem;
}
</style>
"""


# ─────────────────────────────────────────────
# Component Functions
# ─────────────────────────────────────────────

def inject_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero_header() -> None:
    """Render the main area hero header with title and subtitle."""
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">📄 DocSensei</div>
        <div class="hero-subtitle">AI-Powered PDF & DOCX Question Answering</div>
        <div class="hero-badge">⚡ Powered by Google Gemini 2.5 Flash + RAG</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_logo() -> None:
    """Render the DocSensei logo in the sidebar."""
    st.markdown("""
    <div class="sidebar-logo">
        <span class="logo-icon">📄</span>
        <div class="logo-name">DocSensei</div>
        <div class="logo-tagline">AI Document Intelligence</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)


def render_status_badge(is_ready: bool, doc_count: int = 0) -> None:
    """
    Render a coloured status badge showing vector DB state.

    Args:
        is_ready: True if documents have been processed.
        doc_count: Number of indexed document chunks.
    """
    if is_ready:
        st.markdown(
            f'<div class="status-badge ready">✅ Ready · {doc_count} chunks indexed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge empty">⚠️ No documents loaded</div>',
            unsafe_allow_html=True,
        )


def render_user_message(content: str) -> None:
    """Render a user chat bubble."""
    st.markdown(f"""
    <div class="msg-wrapper user">
        <div class="avatar user">🧑</div>
        <div class="bubble user">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_assistant_message(content: str, sources: str = "") -> None:
    """
    Render an assistant chat bubble with optional sources.

    Args:
        content: The main answer text (markdown supported).
        sources: Optional formatted sources string.
    """
    sources_html = ""
    if sources:
        sources_html = f'<div class="sources-block">📚 <strong>Sources</strong><br>{sources}</div>'

    st.markdown(f"""
    <div class="msg-wrapper assistant">
        <div class="avatar assistant">🤖</div>
        <div class="bubble assistant">
            {content}
            {sources_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_thinking_indicator() -> None:
    """Render an animated thinking/typing indicator."""
    st.markdown("""
    <div class="msg-wrapper assistant">
        <div class="avatar assistant">🤖</div>
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_divider() -> None:
    """Render a decorative horizontal divider."""
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def render_stat_card(number: str | int, label: str) -> None:
    """
    Render a small statistics card.

    Args:
        number: The primary stat value to display.
        label: Label describing the stat.
    """
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{number}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_alert(message: str, kind: str = "info") -> None:
    """
    Render a styled alert box.

    Args:
        message: Alert message text.
        kind: One of 'info', 'success', 'warning', 'error'.
    """
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    icon = icons.get(kind, "ℹ️")
    st.markdown(
        f'<div class="alert alert-{kind}">{icon} {message}</div>',
        unsafe_allow_html=True,
    )


def render_suggested_questions(questions: list[str]) -> str | None:
    """
    Render clickable suggested question chips.

    Args:
        questions: List of question strings.

    Returns:
        The clicked question string, or None.
    """
    if not questions:
        return None

    st.markdown("**💡 Suggested Questions:**")
    cols = st.columns(len(questions))
    for i, (col, q) in enumerate(zip(cols, questions)):
        with col:
            if st.button(q, key=f"sq_{i}_{hash(q)}", use_container_width=True):
                return q
    return None


def render_empty_state() -> None:
    """Render the empty state when no documents are loaded."""
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color: #5a6070;">
        <div style="font-size: 3.5rem; margin-bottom: 1rem; filter: grayscale(0.3);">📁</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #9ea3c0; margin-bottom: 0.5rem;">
            No Documents Processed Yet
        </div>
        <div style="font-size: 0.88rem; line-height: 1.6;">
            Upload your PDF or DOCX files in the sidebar,<br>
            then click <strong>Process Documents</strong> to get started.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_welcome_features() -> None:
    """Render a feature overview when the app first loads."""
    st.markdown("""
    <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:1rem; margin-top:1.5rem;">
        <div class="stat-card">
            <div style="font-size:1.8rem;">📄</div>
            <div style="font-weight:600; color:#e8eaf6; margin:0.4rem 0 0.2rem;">Multi-Format</div>
            <div style="font-size:0.78rem; color:#9ea3c0;">PDF & DOCX support</div>
        </div>
        <div class="stat-card">
            <div style="font-size:1.8rem;">🔍</div>
            <div style="font-weight:600; color:#e8eaf6; margin:0.4rem 0 0.2rem;">Semantic Search</div>
            <div style="font-size:0.78rem; color:#9ea3c0;">ChromaDB vector retrieval</div>
        </div>
        <div class="stat-card">
            <div style="font-size:1.8rem;">📚</div>
            <div style="font-weight:600; color:#e8eaf6; margin:0.4rem 0 0.2rem;">Cited Answers</div>
            <div style="font-size:0.78rem; color:#9ea3c0;">Page-level citations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
