# EU News Analyst

## Professional Intelligence Briefing System for European Union Affairs

A production-grade **Retrieval-Augmented Generation (RAG)** platform that scrapes, indexes, and analyses EU Commission news in real-time. Ask natural language questions and get structured intelligence briefings backed by real EU news articles.

**Stack:** Python · Streamlit · FAISS · Sentence-Transformers · Groq LLaMA 3.3 70B

> **Last updated:** June 29, 2026 — Migrated from Google Gemini to Groq + local embeddings

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Data Pipeline](#data-pipeline)
8. [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
9. [Query Modes](#query-modes)
10. [Voice Assistant](#voice-assistant)
11. [Tests](#tests)
12. [Troubleshooting](#troubleshooting)
13. [Changelog](#changelog)

---

## Overview

The EU News Analyst aggregates, indexes, and analyses news from the EU Commission website. Users interact through a professional chat interface that delivers intelligence-style briefings with source citations, analyst commentary, and optional voice output.

### Primary Capabilities

- Automated HTML scraping from `commission.europa.eu` with date-based pagination
- Semantic search via FAISS vector database with local `all-MiniLM-L6-v2` embeddings
- Natural language analysis powered by **Groq LLaMA 3.3 70B** (free API)
- Two response modes: daily overview briefing or deep-dive topic analysis
- Voice input via Google Speech Recognition + voice output via Microsoft Edge TTS
- Real-time chat with conversation history persistence

---

## System Architecture

```
EU Commission Website
        │
        ▼  scrape_eu_news.py
eu_news_data.json
        │
        ▼  build_index.py
┌─────────────────────────────────────┐
│  all-MiniLM-L6-v2 (local, CPU)      │  ← No API key needed for embeddings
│  384-dim vectors                    │
│  FAISS IndexFlatL2                  │
└─────────────────────────────────────┘
        │
        ▼  app.py (Streamlit)
┌─────────────────────────────────────┐
│  User Query                         │
│     → embed (sentence-transformers) │
│     → FAISS search (top-10 chunks)  │
│     → keyword match today's news    │
│     → build prompt with context     │
│     → Groq API (LLaMA 3.3 70B)      │
│     → structured response + sources │
└─────────────────────────────────────┘
        │
        ▼
  Browser UI at http://localhost:8501
```

---

## Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Web framework | Streamlit 1.61+ | Chat UI, sidebar, voice controls |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | Local, CPU, 384-dim, free |
| Vector search | `faiss-cpu` IndexFlatL2 | L2 distance flat index |
| LLM | Groq API — `llama-3.3-70b-versatile` | Free tier: 1,500 req/day |
| Scraping | `requests` + `BeautifulSoup4` | HTML scraping, no RSS |
| Voice input | `SpeechRecognition` + Google STT | Browser mic recording |
| Voice output | `edge-tts` (Microsoft) | Multiple EN voices |
| Runtime | Python 3.13, `gpu_env` virtualenv | CPU-only recommended |

---

## Project Structure

```
EU_News_Analyst/
├── app.py                      # Streamlit UI + full RAG engine
├── build_index.py              # Embedding + FAISS index builder
├── scrape_eu_news.py           # EU Commission paginated scraper
├── eu_news_data.json           # Scraped articles (JSON)
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Test-only dependencies
├── tests/                      # Pytest unit tests
├── LICENSE                     # MIT license
├── RAG_GEMINI.ipynb            # Legacy Gemini-era notebook — not part of the runtime
├── .gitignore                  # Excludes secrets, env, binaries
├── .streamlit/
│   ├── secrets.toml            # API keys — NOT in git
│   └── config.toml             # Streamlit theme
└── gpu_env/                    # Python virtual environment

# Generated files (gitignored — rebuild with build_index.py)
# news_index.faiss
# items.json
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### 2. Install dependencies

```powershell
python -m venv gpu_env
gpu_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` holds the runtime import closure only: `streamlit`, `sentence-transformers` + `torch`, `faiss-cpu`, `numpy`, `requests`, `beautifulsoup4`, `SpeechRecognition`, `edge-tts`, `tqdm`. The pre-migration Google/LangChain stack (`google-generativeai`, `google-api-core`, `langchain*`) and unused extras (`pandas`, `feedparser`, `lxml`, `selenium`, `PyMuPDF`, `tabula-py`, `pydub`, `python-dotenv`, Jupyter/IPython) are no longer listed — nothing in the codebase imports them.

> **Streamlit Cloud:** install the CPU-only `torch` wheel. The default Linux PyPI `torch` wheel is the ~2.5 GB CUDA build and exceeds Cloud's resource limits. `torch` is deliberately left unpinned so each platform can resolve its own build.

### 3. Configure API key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

### 4. Scrape news

```powershell
python scrape_eu_news.py
```

### 5. Build vector index

```powershell
python build_index.py
```

### 6. Launch the app

```powershell
# Windows (use python -m, not the .exe launcher)
& "gpu_env\Scripts\python.exe" -m streamlit run app.py --server.port 8501
```

Open **http://localhost:8501**

---

## Configuration

### Scraper — `scrape_eu_news.py`

```python
import os

CUTOFF_DATE = datetime(2026, 6, 1)   # Scrape from this date onwards
MAX_PAGES   = 50                      # Safety limit on pagination
OUTPUT_FILE = os.getenv("EU_NEWS_OUTPUT_FILE", "eu_news_data.json")
```

### Index builder — `build_index.py`

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, CPU-friendly
EMBEDDING_DIM   = 384
CHUNK_SIZE      = 1000                 # Characters per chunk
CHUNK_OVERLAP   = 200                  # Overlap between chunks
```

### App — `app.py`

Reads `GROQ_API_KEY` from `.streamlit/secrets.toml` (priority) or `GROQ_API_KEY` environment variable.

---

## Data Pipeline

```
Step 1 — Scrape          scrape_eu_news.py
  └─ Visits commission.europa.eu page by page
  └─ Extracts title, date, full content, URL
  └─ Stops at CUTOFF_DATE
  └─ Output: eu_news_data.json

Step 2 — Index           build_index.py
  └─ Loads eu_news_data.json
  └─ Splits articles into 1000-char chunks (200 overlap)
  └─ Embeds each chunk: sentence-transformers → 384-dim vector
  └─ Builds FAISS IndexFlatL2
  └─ Output: news_index.faiss, items.json

Step 3 — Serve           app.py (runs continuously)
  └─ Loads index + embeddings at startup (cached)
  └─ Embeds user queries at runtime
  └─ Searches FAISS for nearest chunks
  └─ Calls Groq API with context + prompt
  └─ Returns structured briefing
```

**Recommended refresh schedule:** Run steps 1 + 2 daily (or weekly).

---

## How the RAG Pipeline Works

```
User: "Tell me about Ukraine support today"
          │
          ▼
1.  detect_query_type() → "detailed"
          │
          ▼
2.  embed_model.encode(query) → [384-dim vector]
          │
          ▼
3.  faiss.index.search(vector, k=10) → top-10 nearest chunks
          │
          ▼
4.  keyword overlap match → filter to today's Ukraine articles
          │
          ▼
5.  Build prompt:
    "You are a senior EU policy analyst.
     USER QUESTION: [query]
     RELEVANT ARTICLES: [full article text]
     Provide: Key Facts / What's Happening /
     Why It Matters / Stakeholders / What to Watch"
          │
          ▼
6.  groq_generate(prompt) → Groq REST API → LLaMA 3.3 70B
          │
          ▼
7.  Structured response + sources displayed in Streamlit
    + optional Edge TTS voice playback
```

---

## Query Modes

The app auto-detects which mode to use based on the query:

| Mode | Triggers | Output |
|------|----------|--------|
| **Overview** | "today's news", "headlines", "briefing", "what's happening" | Top 5 stories + quick analysis + prompts for follow-up |
| **Detailed** | Specific topic, "why", "how", "explain", "tell me about X" | Key Facts · What's Happening · Why It Matters · Stakeholders · What to Watch |

---

## Voice Assistant

**Voice Input:**
1. Click the microphone button in the sidebar
2. Speak your query
3. The app transcribes via Google Speech Recognition and submits automatically

**Voice Output:**
1. Enable "Voice Responses" toggle in the sidebar
2. Select a voice (British/US, Male/Female)
3. Responses are read aloud via Microsoft Edge TTS after each answer

---

## Tests

The unit tests cover the scraper, the chunking/index pipeline, and the RAG logic in `app.py`.
Heavy dependencies (streamlit, sentence-transformers, faiss, edge-tts, SpeechRecognition) are
stubbed, so no API key, model download, or network access is needed.

```bash
pip install -r requirements-dev.txt
pytest

# with a coverage report
pytest --cov=app --cov=build_index --cov=scrape_eu_news --cov-report=term-missing
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Briefing header shows "briefing from [older date]" + an `ARCHIVE` badge | Nothing in the archive is dated today, so the app fell back to the most recent date it has | Expected on weekends/holidays. To refresh: re-run `scrape_eu_news.py` + `build_index.py`, then **restart** the Streamlit server |
| "No news data available for [date]" | *No* article in the archive has a parseable date — not merely "nothing published today". Usually an empty/failed scrape or dates written in a non-English locale | Check the scraper output and the `WARNING` the app logs on startup, then re-run `scrape_eu_news.py` + `build_index.py` |
| `FATAL: kernel is for sm80, built for sm37` | Old GPU incompatible with PyTorch | Already fixed — `CUDA_VISIBLE_DEVICES=-1` forces CPU mode |
| `Fatal error in launcher` when running streamlit | `.exe` launcher points to wrong Python path | Use `python -m streamlit run app.py` instead |
| `Port 8501 is already in use` | Previous Streamlit process still running | Run: `Get-NetTCPConnection -LocalPort 8501 \| % { Stop-Process -Id $_.OwningProcess -Force }` |
| `403 API key reported as leaked` | Key was committed to public repo | Generate a new key at console.groq.com |
| Slow embedding on first run | Model downloading (~90MB) | One-time download, cached afterwards |

---

## Changelog

### Unreleased
- **Changed:** the briefing no longer dead-ends when nothing is dated today. It falls back to the most recent date present in the archive (compared as dates, not strings), and the header and sidebar say so — `briefing from [date]` with an `ARCHIVE` badge and a `Latest` counter instead of `LIVE`/`Today`.
- **Changed:** "No news data available" is now shown only when *no* article has a parseable date; that case also logs a warning naming the expected date format.
- **Fixed:** `requirements.txt` matches the code again (adds `sentence-transformers`, drops the pre-migration Google/LangChain stack) and pulls the CPU-only `torch` wheel so Streamlit Cloud deploys.
- **Fixed:** `build_index.py` and `scrape_eu_news.py` no longer crash on Windows consoles (cp1252 could not encode their emoji output).

### v2.0.0 — June 29, 2026
- **Migrated LLM:** Google Gemini → Groq LLaMA 3.3 70B (free, faster)
- **Migrated embeddings:** Google `text-embedding-004` API → local `all-MiniLM-L6-v2` (no API key needed)
- **Removed:** all Google SDK dependencies — `google-generativeai`, `google-api-core` (no Google GenAI package is installed or imported any more)
- **Added:** `sentence-transformers` (with `torch`) for local CPU embeddings
- **Fixed:** GPU compatibility — forced CPU-only mode for older hardware
- **Updated:** News data refreshed to June 2026 (50 articles, 171 vectors)
- **Fixed:** API key security — removed hardcoded keys from source code

### v1.0.0 — February 2, 2026
- Initial release with Google Gemini + FAISS
- 483 articles (Jan–Feb 2026), 1772 vectors
- Voice input/output via SpeechRecognition + Edge TTS

---

## License

MIT License — see LICENSE file for details.
