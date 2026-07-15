# DocSensei – AI PDF & DOCX Question Answering System

<div align="center">

![DocSensei](https://img.shields.io/badge/DocSensei-AI%20Document%20Q%26A-6c63ff?style=for-the-badge&logo=google&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-ff6b6b?style=for-the-badge)

**DocSensei** is a production-ready, AI-powered document question-answering system.  
Upload your PDF and DOCX files, ask questions in natural language, and get cited answers — powered by Google Gemini 2.5 Flash and a full RAG pipeline.

[🚀 Live Demo](#deployment) · [📖 Docs](#installation) · [🐛 Issues](https://github.com/your-org/docsensei/issues)

</div>

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📄 **Multi-format Upload** | Upload multiple PDFs and DOCX files simultaneously |
| 🔍 **Semantic Search** | ChromaDB vector similarity search for accurate retrieval |
| 🤖 **Gemini 2.5 Flash** | State-of-the-art LLM for high-quality answers |
| 📚 **Page Citations** | Every answer cites document name and page number |
| 💬 **Chat Memory** | Conversation-aware multi-turn Q&A |
| ⚡ **Streaming** | Real-time streaming response display |
| 📝 **Summaries** | AI-generated summaries for each uploaded document |
| 💡 **Suggested Questions** | Auto-generated questions from document content |
| 🔎 **Search History** | Search through past conversation messages |
| 📥 **Export Chat** | Download chat history as JSON or PDF |
| 🎨 **Dark UI** | Premium dark-mode interface with animations |
| 🛡️ **Secure** | API keys via `.env`, file validation, size limits |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DocSensei Architecture                │
└─────────────────────────────────────────────────────────┘

 USER
  │ Upload PDF/DOCX
  ▼
┌──────────┐    ┌───────────┐    ┌──────────────┐
│ Loaders  │───▶│  Splitter │───▶│  Embeddings  │
│ (PDF,    │    │(Recursive │    │(Google Gen AI│
│  DOCX)   │    │  chunks)  │    │  Embeddings) │
└──────────┘    └───────────┘    └──────┬───────┘
                                        │
                                        ▼
                                 ┌────────────┐
                                 │  ChromaDB  │
                                 │ (Vectors)  │
                                 └──────┬─────┘
                                        │
 USER asks question                     │
  │                                     │
  ▼                                     ▼
┌────────────────┐    Retrieve    ┌───────────────┐
│  History-Aware │───────────────▶│ Similarity    │
│  Retriever     │   Top-K docs   │ Search        │
└───────┬────────┘                └───────────────┘
        │
        ▼
┌───────────────────┐
│  Gemini 2.5 Flash │
│    (RAG Chain)    │
└────────┬──────────┘
         │
         ▼
  Answer + Citations → USER
```

---

## 📁 Project Structure

```
DocSensei/
│
├── app.py                 # Main Streamlit application
├── config.py              # Centralised configuration
├── requirements.txt       # Python dependencies
├── README.md
├── .gitignore
├── .env.example
│
├── uploads/               # Temporarily stored user uploads
├── vectorstore/           # Persistent ChromaDB data
├── assets/                # Static assets (logos, images)
├── screenshots/           # App screenshots
│
├── utils/
│   ├── __init__.py
│   ├── loaders.py         # PDF & DOCX document loaders
│   ├── splitter.py        # Text chunking
│   ├── embeddings.py      # Google Generative AI embeddings
│   ├── vector_db.py       # ChromaDB wrapper
│   ├── rag.py             # RAG pipeline orchestration
│   ├── prompts.py         # Prompt templates
│   ├── helpers.py         # File management, export, utilities
│   └── ui.py              # Streamlit UI components & CSS
│
└── tests/
    ├── __init__.py
    └── test_utils.py      # Unit tests
```

---

## 🔧 Installation

### Prerequisites
- Python 3.12+
- A [Google AI API key](https://aistudio.google.com/apikey)

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/docsensei.git
cd docsensei
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Google AI Studio API key for Gemini & embeddings |

---

## 🚀 Running Locally

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📋 Usage

1. **Upload Documents** — Use the sidebar to upload one or more PDF/DOCX files (max 50 MB each)
2. **Process** — Click ⚡ **Process Documents** to extract text, generate embeddings, and store in ChromaDB
3. **Ask Questions** — Type your question in the chat input
4. **Review Answers** — Each answer includes source citations with document name and page number
5. **Export** — Download your chat history as JSON or PDF

---

## ☁️ Deployment

### Streamlit Community Cloud

1. Push your project to a GitHub repository
2. Log in to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App** and connect your GitHub repo
4. Set `app.py` as the main file
5. Add your `GOOGLE_API_KEY` in **Secrets** (Settings → Secrets):
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   ```
6. Deploy!

> **Note**: The `vectorstore/` directory will be ephemeral on Streamlit Cloud. For production, use a hosted vector database like Pinecone or Weaviate.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📸 Screenshots

> Add your screenshots to the `screenshots/` directory and update these links.

| Chat Interface | Document Summaries |
|---------------|-------------------|
| *(screenshot)* | *(screenshot)* |

---

## 🔮 Future Improvements

- [ ] 🔗 Support for more file types (TXT, XLSX, PPTX, HTML)
- [ ] 🌐 Multi-language document support
- [ ] 🔉 Voice input and speech output
- [ ] 📊 Document comparison across multiple files
- [ ] 🗂️ Named conversation sessions
- [ ] 🔒 User authentication
- [ ] ☁️ Cloud vector DB integration (Pinecone / Weaviate)
- [ ] 📈 Usage analytics dashboard
- [ ] 🔄 Automatic re-indexing on document changes
- [ ] 🧩 REST API backend (FastAPI)

---

## 📦 Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.12+ |
| Frontend | Streamlit | 1.35+ |
| LLM | Google Gemini 2.5 Flash | Latest |
| Embeddings | Google Generative AI | `embedding-001` |
| Framework | LangChain | 0.2+ |
| Vector DB | ChromaDB | 0.5+ |
| PDF Loader | PyPDFLoader | 4.2+ |
| DOCX Loader | Docx2txtLoader | 0.8 |
| Text Splitter | RecursiveCharacterTextSplitter | — |
| PDF Export | fpdf2 | 2.7+ |
| Config | python-dotenv | 1.0+ |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using Python, LangChain, and Google Gemini.

</div>
