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
11. [Troubleshooting](#troubleshooting)
12. [Changelog](#changelog)

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
| Web framework | Streamlit 1.52+ | Chat UI, sidebar, voice controls |
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
├── .gitignore                  # Excludes secrets, env, binaries
├── .streamlit/
│   ├── secrets.toml            # API keys — NOT in git
│   └── config.toml             # Streamlit theme
└── gpu_env/                    # Python virtual environment

# Generated files (gitignored — rebuild with build_index.py)
# news_index.faiss
# items_with_embeddings.pkl
# items_metadata.pkl
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

### 3. Configure API key

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
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
CUTOFF_DATE = datetime(2026, 6, 1)   # Scrape from this date onwards
MAX_PAGES   = 50                      # Safety limit on pagination
OUTPUT_FILE = r"D:\...\eu_news_data.json"
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
  └─ Output: news_index.faiss, items_with_embeddings.pkl, items_metadata.pkl

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

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "No news data available for [date]" | Scraped data is old or app cache is stale | Re-run `scrape_eu_news.py` + `build_index.py`, then **restart** the Streamlit server |
| `FATAL: kernel is for sm80, built for sm37` | Old GPU incompatible with PyTorch | Already fixed — `CUDA_VISIBLE_DEVICES=-1` forces CPU mode |
| `Fatal error in launcher` when running streamlit | `.exe` launcher points to wrong Python path | Use `python -m streamlit run app.py` instead |
| `Port 8501 is already in use` | Previous Streamlit process still running | Run: `Get-NetTCPConnection -LocalPort 8501 \| % { Stop-Process -Id $_.OwningProcess -Force }` |
| `403 API key reported as leaked` | Key was committed to public repo | Generate a new key at console.groq.com |
| Slow embedding on first run | Model downloading (~90MB) | One-time download, cached afterwards |

---

## Changelog

### v2.0.0 — June 29, 2026
- **Migrated LLM:** Google Gemini → Groq LLaMA 3.3 70B (free, faster)
- **Migrated embeddings:** Google `text-embedding-004` API → local `all-MiniLM-L6-v2` (no API key needed)
- **Removed:** `google-generativeai` (deprecated SDK) dependency
- **Added:** `sentence-transformers`, `google-genai` packages
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

8. [Usage Guide](#usage-guide)
9. [Data Pipeline](#data-pipeline)
10. [Voice Assistant](#voice-assistant)
11. [API Reference](#api-reference)
12. [Performance Specifications](#performance-specifications)
13. [Troubleshooting](#troubleshooting)
14. [Development Roadmap](#development-roadmap)
15. [Contributing](#contributing)
16. [License](#license)

---

## Overview

The EU News Analyst is an enterprise-level intelligence platform that aggregates, indexes, and analyzes news from authoritative European Union sources. The system provides users with detailed analytical briefings on EU policy, economic developments, institutional activities, and geopolitical events.

### Primary Capabilities

- Automated news aggregation from official EU institutions and major European media outlets
- Semantic search powered by FAISS vector database with Google text-embedding-004 embeddings
- Natural language query processing with Google Gemini 2.0 Flash and 1.5 Flash models
- Professional intelligence-style analytical responses with executive summaries
- Separate analyst assessment section providing strategic insights and forward-looking analysis
- Voice input via microphone recording with Google Speech Recognition
- Voice output via Microsoft Edge Text-to-Speech with multiple voice options
- Real-time chat interface with conversation history persistence

---

## System Architecture

The platform follows a modular architecture with clear separation of concerns:

```
                                    USER INTERFACE
                                         |
                           +-------------+-------------+
                           |                           |
                      Text Input              Voice Input (Microphone)
                           |                           |
                           +-------------+-------------+
                                         |
                                         v
                              +-------------------+
                              |   QUERY ROUTER    |
                              | (Streamlit App)   |
                              +-------------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
          +------------------+                     +--------------------+
          |  FAISS VECTOR    |                     |   NEWS DATABASE    |
          |    SEARCH        |                     |  (JSON + Pickle)   |
          +------------------+                     +--------------------+
                    |                                         |
                    +--------------------+--------------------+
                                         |
                                         v
                              +-------------------+
                              |  CONTEXT BUILDER  |
                              |  (Relevance       |
                              |   Ranking)        |
                              +-------------------+
                                         |
                                         v
                              +-------------------+
                              |  GOOGLE GEMINI    |
                              |  LLM ANALYSIS     |
                              +-------------------+
                                         |
                    +--------------------+--------------------+
                    |                    |                    |
                    v                    v                    v
          +----------------+   +------------------+   +----------------+
          |   EXECUTIVE    |   |     ANALYST      |   |    SOURCES     |
          |   SUMMARY      |   |   ASSESSMENT     |   |   CITATION     |
          +----------------+   +------------------+   +----------------+
                                         |
                                         v
                              +-------------------+
                              |   EDGE TTS        |
                              | (Voice Output)    |
                              +-------------------+
                                         |
                                         v
                                  USER RESPONSE
```

### Data Flow Description

1. User submits a query via text input or voice recording
2. Voice input is transcribed using Google Speech Recognition API
3. Query is processed to generate embeddings via Google text-embedding-004
4. FAISS performs approximate nearest neighbor search across indexed news articles
5. Relevant articles are retrieved and formatted as context
6. Google Gemini generates structured analytical response
7. A separate analyst assessment provides strategic insights
8. Sources are cited with links to original articles
9. Optional voice synthesis via Edge TTS delivers audio response

---

## Core Features

### Intelligence Analysis Engine

The system functions as a virtual senior intelligence analyst, providing structured briefings that include:

**Executive Summary**
- Concise overview of key developments relevant to the query
- Prioritized information hierarchy for rapid comprehension
- Clear articulation of main themes and trends

**Detailed Analysis**
- In-depth examination of policy developments and their significance
- Stakeholder mapping with positions and motivations
- Economic, social, and political impact assessment
- Cross-referencing of information across multiple sources

**Contextual Background**
- Historical context necessary for understanding current events
- Institutional framework explanations
- Relevant precedents and analogous situations

**Forward Outlook**
- Anticipated next steps and likely developments
- Key dates and upcoming decision points
- Scenarios and potential outcomes

### Analyst Assessment

A dedicated section provides the analyst perspective:

- Pattern recognition across news reporting
- Identification of underlying dynamics and power structures
- Recommendations for continued monitoring
- Acknowledgment of intelligence gaps and unanswered questions

### Real-Time Voice Interaction

The voice assistant provides hands-free operation:

- Microphone recording directly in the browser interface
- Real-time speech-to-text transcription
- Multiple voice options for text-to-speech output:
  - British Female (Sonia)
  - British Male (Ryan)
  - US Female (Jenny)
  - US Male (Guy)
- Audio playback integrated into chat responses

### Professional User Interface

The interface follows enterprise design principles:

- Clean gradient header with EU-institutional color palette
- Status indicators for system health and database connectivity
- Statistics dashboard showing article counts and index size
- Collapsible source citations with direct links
- Conversation history with distinct user and assistant styling
- Responsive layout optimized for desktop viewing

---

## Technology Stack

### Core Framework

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Web Framework | Streamlit | 1.40.0+ | Interactive web application |
| Language | Python | 3.13+ | Primary development language |

### Artificial Intelligence

| Component | Technology | Purpose |
|-----------|------------|---------|
| Large Language Model | Google Gemini 2.0 Flash | Primary analysis generation |
| Fallback LLM | Google Gemini 1.5 Flash | Backup model for reliability |
| Embeddings | Google text-embedding-004 | Semantic vector generation |
| Vector Database | FAISS (CPU) | Approximate nearest neighbor search |

### Voice Processing

| Component | Technology | Purpose |
|-----------|------------|---------|
| Speech Recognition | Google Speech Recognition | Voice input transcription |
| Text-to-Speech | Microsoft Edge TTS | Voice output synthesis |
| Audio Processing | PyDub | Audio format handling |

### Data Acquisition

| Component | Technology | Purpose |
|-----------|------------|---------|
| HTTP Client | Requests | Web page fetching |
| HTML Parsing | BeautifulSoup4 | Content extraction |
| XML Parsing | BeautifulSoup4 (xml) | RSS feed parsing |
| Browser Automation | Selenium | Dynamic content scraping |

### Data Processing

| Component | Technology | Purpose |
|-----------|------------|---------|
| Text Splitting | LangChain | Document chunking |
| Numerical Computing | NumPy | Vector operations |
| Data Structures | Pandas | Tabular data handling |
| Serialization | Pickle | Index persistence |

---

## Project Structure

```
EU_News_Analyst/
|
|-- app.py                      Main Streamlit application (667 lines)
|                               - User interface components
|                               - Query processing pipeline
|                               - Voice input/output handling
|                               - Gemini API integration
|
|-- scrape_eu_news.py           News aggregation engine (271 lines)
|                               - RSS feed parsing
|                               - Direct HTML scraping
|                               - Article deduplication
|                               - JSON database generation
|
|-- RAG_GEMINI.ipynb            Jupyter notebook for development
|                               - Experimentation environment
|                               - Pipeline prototyping
|                               - Embedding generation
|
|-- requirements.txt            Python package dependencies
|                               - Production dependencies
|                               - Development tools
|
|-- eu_news_data.json           News article database
|                               - Article metadata (title, date, source)
|                               - Full article content
|                               - Source links
|
|-- news_index.faiss            FAISS vector index
|                               - Pre-computed embeddings
|                               - Optimized for similarity search
|
|-- items_with_embeddings.pkl   Pickled embedding data
|                               - Text chunks with metadata
|                               - Embedding vectors
|
|-- gpu_env/                    Python virtual environment
|   |-- Scripts/                Windows executables
|   |-- Lib/site-packages/      Installed packages
|
|-- README.md                   Project documentation
```

---

## Installation Guide

### System Requirements

- Operating System: Windows 10/11, macOS 10.15+, or Linux
- Python: Version 3.13 or higher
- Memory: Minimum 8GB RAM recommended
- Storage: 2GB available disk space
- Network: Internet connection for API access

### Step-by-Step Installation

**1. Clone the Repository**

```bash
git clone https://github.com/yourusername/EU_News_Analyst.git
cd EU_News_Analyst
```

**2. Create and Activate Virtual Environment**

Windows (PowerShell):
```powershell
python -m venv gpu_env
gpu_env\Scripts\Activate.ps1
```

Windows (Command Prompt):
```cmd
python -m venv gpu_env
gpu_env\Scripts\activate.bat
```

macOS/Linux:
```bash
python -m venv gpu_env
source gpu_env/bin/activate
```

**3. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Obtain Google API Key**

1. Navigate to Google AI Studio: https://aistudio.google.com/
2. Sign in with your Google account
3. Create a new API key
4. Copy the generated key

**5. Configure Environment Variable**

Windows (PowerShell):
```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

Windows (Persistent):
```powershell
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "your_api_key_here", "User")
```

macOS/Linux:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

**6. Generate News Index (First Run Only)**

If the FAISS index does not exist:
```bash
python scrape_eu_news.py
```

**7. Launch Application**

```bash
streamlit run app.py --server.port 8501
```

**8. Access the Interface**

Open your web browser and navigate to:
```
http://localhost:8501
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| GOOGLE_API_KEY | Yes | Google Generative AI API key for Gemini and embeddings |

### Application Settings

The following parameters can be modified in app.py:

```python
# Date Configuration
CURRENT_DATE = datetime.now()
DATE_STR = CURRENT_DATE.strftime("%A, %d %B %Y")

# Page Configuration
st.set_page_config(
    page_title="EU Intelligence Briefing",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Scraper Configuration

Modify scrape_eu_news.py for data acquisition settings:

```python
# Output Configuration
OUTPUT_FILE = "eu_news_data.json"
TARGET_TOTAL_ARTICLES = 500

# Date Range
TODAY = datetime.now()
START_DATE = TODAY - timedelta(days=60)  # 60-day rolling window
```

### News Sources

The scraper aggregates from the following sources:

| Source | Type | Coverage |
|--------|------|----------|
| European Commission | RSS Feed | Official press releases |
| European Parliament | RSS Feed | Parliamentary news |
| EU Council | Direct Scrape | Council decisions |
| Euronews EU | RSS Feed | Pan-European news |
| Politico EU | RSS Feed | Policy analysis |
| EU Observer | RSS Feed | Investigative journalism |
| DW EU News | RSS Feed | German perspective |
| Reuters EU | RSS Feed | Wire service coverage |

---

## Usage Guide

### Text Queries

1. Enter your question in the chat input field at the bottom of the interface
2. Press Enter or click the send button
3. Wait for the system to process your query (typically 3-8 seconds)
4. Review the structured response including:
   - Main analysis with executive summary
   - Analyst assessment section
   - Expandable sources panel

### Voice Queries

**Method 1: Sidebar Microphone**
1. Navigate to the "Voice Input" section in the sidebar
2. Click the microphone button to start recording
3. Speak your question clearly
4. Wait for transcription to appear
5. Click "Submit Voice Query" to process

**Method 2: Main Input Area**
1. Locate the microphone icon next to the chat input
2. Click to record your question
3. The system will automatically transcribe and submit

### Voice Responses

1. Enable "Voice Responses" toggle in the sidebar
2. Select your preferred voice from the dropdown
3. Submitted queries will generate audio playback
4. Use the audio player controls in the response to listen

### Example Queries

```
What are the latest EU regulations on artificial intelligence?

Summarize today's major developments in EU-UK relations.

What is the European Central Bank's current monetary policy stance?

Explain the recent changes to EU agricultural subsidies.

What are EU member states' positions on migration policy reform?
```

---

## Data Pipeline

### News Acquisition Process

```
1. RSS Feed Parsing
   |-- Fetch RSS XML from configured sources
   |-- Parse using BeautifulSoup XML parser
   |-- Extract: title, link, publication date, description
   |-- Filter by date range (60-day window)

2. Direct HTML Scraping
   |-- Fetch article pages via HTTP requests
   |-- Parse HTML content with BeautifulSoup
   |-- Extract full article text
   |-- Handle pagination and dynamic content

3. Content Processing
   |-- Deduplicate articles by title similarity
   |-- Normalize date formats
   |-- Clean HTML entities and formatting
   |-- Validate content completeness

4. Database Generation
   |-- Serialize to JSON format
   |-- Store in eu_news_data.json
   |-- Maintain source attribution
```

### Index Generation Process

```
1. Text Chunking
   |-- Load articles from JSON database
   |-- Split into semantic chunks (LangChain)
   |-- Preserve metadata with each chunk
   |-- Target chunk size: 500-1000 tokens

2. Embedding Generation
   |-- Process chunks through text-embedding-004
   |-- Generate 768-dimensional vectors
   |-- Batch processing for efficiency

3. FAISS Index Construction
   |-- Build flat L2 index
   |-- Store index in news_index.faiss
   |-- Serialize chunks in items_with_embeddings.pkl
```

### Query Processing Flow

```
1. Query Embedding
   |-- Encode user query via text-embedding-004
   |-- Generate query vector (768 dimensions)

2. Vector Search
   |-- Perform approximate nearest neighbor search
   |-- Retrieve top-k relevant chunks (k=10)
   |-- Calculate similarity scores

3. Context Assembly
   |-- Merge retrieved chunks
   |-- Add today's full articles
   |-- Format for LLM consumption
   |-- Include source metadata

4. LLM Generation
   |-- Submit context + query to Gemini
   |-- Generate structured analysis
   |-- Generate separate analyst assessment
   |-- Return formatted response
```

### Two-Tier RAG Response System

The system implements an intelligent two-tier response mechanism that adapts to the user's query intent:

**Tier 1: Overview Mode**

Triggered by queries asking for general news briefings, headlines, or summaries:
- Keywords detected: "news", "headlines", "briefing", "today", "what's happening", "summary"
- Response format: Bullet-point headlines with brief analysis
- Purpose: Quick situational awareness without overwhelming detail

Example queries for Overview Mode:
```
What's in the news today?
Give me the EU headlines
Brief me on today's developments
What's happening in the European Union?
```

**Tier 2: Detailed Analysis Mode**

Triggered by specific questions about topics, policies, or events:
- Activated when query asks about a particular subject
- Response format: Comprehensive analysis with key facts, stakeholders, implications
- Purpose: Deep-dive investigation when user wants full context

Example queries for Detailed Mode:
```
Tell me more about the AI Act implementation
What are the details on the ECB interest rate decision?
Explain the EU agricultural policy changes
```

**Query Detection Logic**

The `detect_query_type()` function analyzes queries to route them appropriately:

```python
def detect_query_type(query: str) -> str:
    """
    Detect if query is asking for overview/headlines or detailed analysis.
    
    Returns:
        'overview' - for general news/headline queries
        'detailed' - for specific topic deep-dives
    """
    query_lower = query.lower()
    overview_keywords = [
        'news', 'headlines', 'briefing', 'today', 
        "what's happening", 'summary', 'brief me',
        'update', 'latest', 'what happened'
    ]
    return 'overview' if any(kw in query_lower for kw in overview_keywords) else 'detailed'
```

This two-tier approach optimizes the user experience by:
1. Reducing response latency for simple briefing requests
2. Preserving analytical depth for investigative queries
3. Encouraging natural conversational flow (overview → follow-up → deep-dive)

---

## Voice Assistant

### Speech Recognition Details

The system uses Google Speech Recognition for transcription:

- Audio format: WAV (converted automatically)
- Language: English (primary)
- Processing: Server-side via Google API
- Latency: 1-3 seconds typical

### Text-to-Speech Details

Microsoft Edge TTS provides voice synthesis:

| Voice ID | Name | Accent | Gender |
|----------|------|--------|--------|
| en-GB-SoniaNeural | Sonia | British | Female |
| en-GB-RyanNeural | Ryan | British | Male |
| en-US-JennyNeural | Jenny | American | Female |
| en-US-GuyNeural | Guy | American | Male |

### Audio Processing

- Content cleaning: URLs, markdown, special characters removed
- Length limit: 2000 characters per synthesis
- Format: MP3 streaming
- Playback: Native HTML5 audio player

---

## API Reference

### Core Functions

**load_data()**
```python
@st.cache_resource
def load_data():
    """
    Load FAISS index and embeddings data.
    
    Returns:
        tuple: (index, items, news_data, stats, status)
            - index: FAISS index object
            - items: List of text chunks with metadata
            - news_data: Raw news articles from JSON
            - stats: Dictionary with article and chunk counts
            - status: String indicating system status
    """
```

**get_embedding(text)**
```python
def get_embedding(text: str) -> list:
    """
    Generate embedding vector for input text.
    
    Args:
        text: Input string to embed
        
    Returns:
        list: 768-dimensional embedding vector
    """
```

**analyze_query(query, news_data, index, items)**
```python
def analyze_query(query: str, news_data: list, index, items: list) -> dict:
    """
    Generate comprehensive intelligence analysis.
    
    Args:
        query: User question or topic
        news_data: Full article database
        index: FAISS index for search
        items: Indexed chunks with metadata
        
    Returns:
        dict: {
            'analysis': Main analytical response,
            'thoughts': Analyst assessment section,
            'sources': List of source citations
        }
    """
```

**text_to_speech(text, voice)**
```python
async def text_to_speech(text: str, voice: str) -> BytesIO:
    """
    Convert text to speech audio.
    
    Args:
        text: Text content to synthesize
        voice: Voice ID (e.g., 'en-GB-SoniaNeural')
        
    Returns:
        BytesIO: Audio buffer containing MP3 data
    """
```

**speech_to_text(audio_bytes)**
```python
def speech_to_text(audio_bytes: bytes) -> str:
    """
    Transcribe audio to text.
    
    Args:
        audio_bytes: Raw audio data
        
    Returns:
        str: Transcribed text or None on failure
    """
```

---

## Performance Specifications

### Latency Benchmarks

| Operation | Typical Duration | Maximum Duration |
|-----------|-----------------|------------------|
| FAISS Search | 50-100ms | 200ms |
| Embedding Generation | 200-500ms | 1s |
| LLM Analysis | 2-5s | 15s |
| Voice Synthesis | 1-3s | 5s |
| Speech Recognition | 1-3s | 5s |
| End-to-End Query | 4-10s | 25s |

### Resource Utilization

| Resource | Idle | Active Query |
|----------|------|--------------|
| Memory | 500MB | 1-2GB |
| CPU | 5% | 30-60% |
| Network | Minimal | 100KB-1MB |

### Database Metrics

| Metric | Value |
|--------|-------|
| Articles | 400-500 |
| Text Chunks | 2000-3000 |
| Vector Dimensions | 768 |
| Index Size | 10-50MB |
| JSON Database | 5-15MB |

---

## Troubleshooting

### Common Issues and Solutions

**Issue: GOOGLE_API_KEY not found**

Symptom: Application displays "CRITICAL: GOOGLE_API_KEY environment variable is missing"

Solution:
```powershell
# Windows PowerShell
$env:GOOGLE_API_KEY="your_api_key_here"
streamlit run app.py --server.port 8501
```

**Issue: FAISS index not found**

Symptom: Application fails to load with FileNotFoundError for news_index.faiss

Solution:
```bash
python scrape_eu_news.py
```

**Issue: No news for today's date**

Symptom: Responses indicate "No news available for today"

Solution:
```bash
# Re-run scraper to fetch latest articles
python scrape_eu_news.py
```

**Issue: Voice input not working**

Symptom: Microphone button does not respond or transcription fails

Solutions:
1. Check browser microphone permissions
2. Ensure HTTPS or localhost connection
3. Install audio dependencies:
```bash
pip install pyaudio pydub
```

**Issue: API rate limit exceeded**

Symptom: Responses fail with quota error messages

Solutions:
1. Reduce query frequency
2. Check API quota in Google AI Studio
3. Consider upgrading API plan

**Issue: Port already in use**

Symptom: Streamlit fails to start with address already in use error

Solution:
```powershell
# Find and kill process on port 8501
Get-Process -Name python | Stop-Process -Force
streamlit run app.py --server.port 8502
```

---

## Development Roadmap

### Version 2.0 (Planned)

- Real-time news streaming with WebSocket integration
- Named Entity Recognition for automated entity tracking
- Multi-language query support (French, German, Spanish)
- User authentication and personalized preferences

### Version 2.1 (Planned)

- Multi-modal analysis incorporating images and charts
- Automated daily briefing email generation
- Export functionality for PDF and Excel reports
- Sentiment analysis with trend visualization

### Version 2.2 (Planned)

- Graph database integration for relationship mapping
- Custom embedding fine-tuning on EU domain
- API endpoints for programmatic access
- Webhook notifications for breaking news

### Version 3.0 (Future)

- Custom fine-tuned language models
- On-premise deployment option
- Enterprise SSO integration
- Audit logging and compliance features

---

## Contributing

Contributions to the EU News Analyst project are welcome. Areas of particular interest include:

- Additional news source integrations
- Language model optimization
- User interface enhancements
- Documentation improvements
- Performance optimizations
- Security hardening

### Contribution Process

1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Submit a pull request with detailed description
5. Address review feedback

---

## License

This project is provided as open-source software. Users must comply with:

- Google Generative AI Terms of Service
- News source copyright and attribution requirements
- Applicable data protection regulations

---

## Contact and Support

For issues, feature requests, or questions:

- Create an issue in the GitHub repository
- Review existing documentation and troubleshooting guides
- Check closed issues for previously resolved problems

---

**EU News Analyst - Professional Intelligence Briefing System**

*Version 2.0*

*Last Updated: February 2, 2026*
