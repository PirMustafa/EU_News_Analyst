"""
EU Daily Intelligence Briefing
A professional news analysis system with voice interaction
"""

import streamlit as st
import os
from datetime import datetime
import tempfile
import asyncio
import numpy as np

import common

# --- CONFIGURATION ---
CURRENT_DATE = datetime.now()
DATE_STR = CURRENT_DATE.strftime(common.DISPLAY_DATE_FORMAT)

st.set_page_config(
    page_title=f"EU Intelligence Briefing - {CURRENT_DATE.strftime('%d %b %Y')}",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1a365d 0%, #2d4a7c 50%, #1e3a5f 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .header-title {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .header-subtitle {
        color: #a0c4ff;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .status-badge {
        display: inline-block;
        background: #22c55e;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 1rem;
    }
    
    /* Chat message styling */
    .user-message {
        background: #f0f4f8;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1a365d;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .message-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    /* Analyst thoughts section */
    .analyst-thoughts {
        background: #fefce8;
        border: 1px solid #fef08a;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .analyst-thoughts-title {
        font-weight: 600;
        color: #854d0e;
        margin-bottom: 0.5rem;
    }
    
    /* Sources styling */
    .sources-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .source-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }
    
    .source-item:last-child {
        border-bottom: none;
    }
    
    /* Sidebar styling */
    .sidebar-stat {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #1a365d;
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
    }
    
    /* Voice button styling */
    .voice-active {
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
        100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- LAZY IMPORTS ---
@st.cache_resource
def load_dependencies():
    """Load heavy dependencies once."""
    import faiss
    import pickle
    embed_model = common.load_embedding_model()
    return embed_model, faiss, pickle

# --- API SETUP ---
# Read Groq API key from secrets or environment
groq_api_key = None

# 1. Try Streamlit secrets first
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY")
except:
    pass

# 2. Try environment variable
if not groq_api_key:
    groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error("CRITICAL: No GROQ_API_KEY found. Please set it in .streamlit/secrets.toml")
    st.stop()

embed_model, faiss, pickle = load_dependencies()

# --- DATA LOADING ---
@st.cache_resource
def load_data():
    """Load FAISS index and embeddings data."""
    try:
        index = faiss.read_index(common.INDEX_FILE)
        with open(common.EMBEDDINGS_FILE, "rb") as f:
            items_with_embeddings = pickle.load(f)
        
        items = []
        for item in items_with_embeddings:
            items.append({
                "text": item["text"],
                "meta": item["metadata"]
            })
        
        # Load raw news data
        news_data = common.load_news_data()
        
        unique_articles = len(set(item["metadata"]["title"] for item in items_with_embeddings))
        return index, items, news_data, {"articles": unique_articles, "chunks": len(items)}, "Online"
    except Exception as e:
        return None, [], [], {}, f"Error: {str(e)}"

# --- EMBEDDING FUNCTION ---
def get_embedding(text):
    """Generate embedding for text using local sentence-transformers model."""
    try:
        return common.embed_text(embed_model, text)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def groq_generate(prompt, model="llama-3.3-70b-versatile"):
    """Call Groq API for text generation."""
    import requests
    headers = {
        "Authorization": "Bearer " + groq_api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.7
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

async def text_to_speech(text, voice="en-GB-SoniaNeural"):
    """Convert text to speech using Edge TTS."""
    try:
        import edge_tts
        from io import BytesIO
        
        # Clean text for speech
        clean_text = text.split("**Sources:**")[0].split("Sources:")[0]
        clean_text = clean_text.replace("**", "").replace("*", "").replace("#", "").replace("-", "")
        import re
        clean_text = re.sub(r'http\S+', '', clean_text)
        clean_text = re.sub(r'\[.*?\]', '', clean_text)
        # Remove numbered lists formatting
        clean_text = re.sub(r'\d+\.\s+', '', clean_text)
        clean_text = clean_text[:2000]  # Increased limit for more content
        
        communicate = edge_tts.Communicate(clean_text, voice)
        
        audio_buffer = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        return None

def speech_to_text(audio_bytes):
    """Convert speech to text."""
    try:
        import speech_recognition as sr
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.unlink(temp_path)
        return text
    except Exception as e:
        return None

# --- QUERY TYPE DETECTION ---
def detect_query_type(query, conversation_history):
    """Detect if user wants overview/headlines or detailed analysis."""
    query_lower = query.lower().strip()
    
    # Overview/headlines patterns
    overview_patterns = [
        "what's the news", "whats the news", "what is the news",
        "today's news", "todays news", "news today",
        "headlines", "what's happening", "whats happening",
        "give me the news", "show me the news", "latest news",
        "briefing", "summary", "overview", "what's new", "whats new",
        "any news", "news update", "daily briefing", "morning briefing"
    ]
    
    # Check if this is an overview request
    for pattern in overview_patterns:
        if pattern in query_lower:
            return "overview"
    
    # If there's conversation history with headlines, and user asks about specific topic
    if conversation_history and len(conversation_history) > 0:
        # Check if user is asking about something mentioned in previous headlines
        return "detailed"
    
    # Default to overview for short/vague queries, detailed for specific ones
    if len(query.split()) <= 5 and not any(word in query_lower for word in ['why', 'how', 'explain', 'details', 'more about', 'tell me about']):
        return "overview"
    
    return "detailed"

# --- INTELLIGENCE ANALYST FUNCTION ---
def analyze_query(query, news_data, index, items, query_type="auto", conversation_history=None):
    """Generate intelligence analysis - either overview or detailed."""
    
    # Auto-detect query type if not specified
    if query_type == "auto":
        query_type = detect_query_type(query, conversation_history or [])
    
    # Get today's news
    todays_news = [a for a in news_data if a.get('date', '').strip() == DATE_STR]
    
    if not todays_news:
        return {
            "analysis": f"No news data available for {DATE_STR}. The archive may need to be updated by running the scraper.",
            "thoughts": "",
            "sources": [],
            "query_type": query_type
        }
    
    sources = []
    for article in todays_news[:10]:
        sources.append({
            'title': article.get('title', 'Untitled'),
            'source': article.get('source', 'Unknown'),
            'date': article.get('date', DATE_STR),
            'link': article.get('link', ''),
            'content': article.get('content', '')[:300]
        })
    
    if query_type == "overview":
        # --- OVERVIEW MODE: Headlines + Brief Summary ---
        headlines_list = "\n".join([
            f"{i+1}. **{article.get('title', 'Untitled')}** ({article.get('source', 'Unknown')})"
            for i, article in enumerate(todays_news[:10])
        ])
        
        # Create brief context for analysis
        brief_context = "\n".join([
            f"- {article.get('title', '')}: {article.get('content', '')[:200]}..."
            for article in todays_news[:8]
        ])
        
        overview_prompt = f"""You are an EU news briefing assistant. Provide a concise daily briefing.

TODAY'S DATE: {DATE_STR}
TOTAL ARTICLES: {len(todays_news)}

TODAY'S HEADLINES:
{headlines_list}

BRIEF CONTENT:
{brief_context}

Provide a response in this EXACT format:

## Today's EU News Briefing ({DATE_STR})

**{len(todays_news)} articles published today**

### Top Stories

[List the 3-5 most important stories with ONE sentence each explaining why they matter]

### Quick Analysis

[2-3 sentences on the main themes or trends across today's news]

### Want to Know More?

Ask me about any specific story for detailed analysis. For example:
- "Tell me more about [topic]"
- "What are the details on [headline]?"

Keep it brief and scannable. No deep analysis yet - save that for when they ask."""

        try:
            analysis = groq_generate(overview_prompt)
            return {
                "analysis": analysis,
                "thoughts": "",
                "sources": sources,
                "query_type": "overview"
            }
        except Exception as e:
            return {
                "analysis": f"Error generating overview: {str(e)}",
                "thoughts": "",
                "sources": sources,
                "query_type": "overview"
            }
    
    else:
        # --- DETAILED MODE: Deep dive on specific topic ---
        
        # Search FAISS for relevant content
        query_emb = get_embedding(query)
        relevant_chunks = []
        if query_emb and index:
            query_array = np.array([query_emb], dtype=np.float32)
            distances, indices = index.search(query_array, 10)
            for idx in indices[0]:
                if idx < len(items):
                    relevant_chunks.append(items[idx])
        
        # Find articles matching the query topic
        matching_articles = []
        query_words = set(query.lower().split())
        for article in todays_news:
            title_words = set(article.get('title', '').lower().split())
            content_words = set(article.get('content', '')[:500].lower().split())
            # Check for word overlap
            if len(query_words & title_words) >= 1 or len(query_words & content_words) >= 2:
                matching_articles.append(article)
        
        # If no direct matches, use FAISS results
        if not matching_articles and relevant_chunks:
            # Get articles from chunks
            chunk_titles = [c['meta'].get('title') for c in relevant_chunks[:5]]
            matching_articles = [a for a in todays_news if a.get('title') in chunk_titles]
        
        # If still no matches, use top articles
        if not matching_articles:
            matching_articles = todays_news[:5]
        
        # Build detailed context
        context_parts = []
        for article in matching_articles[:5]:
            context_parts.append(f"""
ARTICLE: {article.get('title', 'Untitled')}
SOURCE: {article.get('source', 'Unknown')}
DATE: {article.get('date', DATE_STR)}
FULL CONTENT: {article.get('content', '')}
---""")
        
        full_context = "\n".join(context_parts)
        
        detailed_prompt = f"""You are a senior EU policy analyst providing detailed analysis.

TODAY'S DATE: {DATE_STR}

USER QUESTION: {query}

RELEVANT ARTICLES:
{full_context}

Provide a detailed analysis with:

## Detailed Analysis

### Key Facts
[Bullet points of the most important facts from the articles]

### What's Happening
[Explain the situation in detail - who, what, when, where]

### Why It Matters
[Explain the significance and implications]

### Stakeholders & Positions
[Key players and their stances]

### What to Watch
[Future developments to monitor]

Be thorough but focused on what the user asked. Use information from the articles.
If the articles don't contain enough information, say so."""

        thoughts_prompt = f"""Based on these articles about "{query}":

{full_context}

In 2-3 sentences, what's the key insight an analyst should note?"""

        try:
            analysis = groq_generate(detailed_prompt)
            thoughts = groq_generate(thoughts_prompt)
            return {
                "analysis": analysis,
                "thoughts": thoughts,
                "sources": sources[:5],
                "query_type": "detailed"
            }
        except Exception as e:
            return {
                "analysis": f"Analysis generation failed: {str(e)}",
                "thoughts": "",
                "sources": sources[:5],
                "query_type": "detailed"
            }

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False
if "processing" not in st.session_state:
    st.session_state.processing = False

# --- LOAD DATA ---
index, items, news_data, stats, status = load_data()
todays_count = len([a for a in news_data if a.get('date') == DATE_STR])

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### System Status")
    
    # Status indicator
    if status == "Online":
        st.success(f"System: {status}")
    else:
        st.error(f"System: {status}")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### Intelligence Database")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="stat-number">{todays_count}</div>
            <div class="stat-label">Today</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="stat-number">{stats.get('articles', 0)}</div>
            <div class="stat-label">Archive</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Voice toggle
    st.markdown("### Voice Assistant")
    voice_enabled = st.toggle("Enable Voice Responses", value=st.session_state.voice_enabled)
    st.session_state.voice_enabled = voice_enabled
    
    if voice_enabled:
        st.caption("Responses will be read aloud")
        # Voice selection
        voice_option = st.selectbox(
            "Voice",
            ["British Female (Sonia)", "British Male (Ryan)", "US Female (Jenny)", "US Male (Guy)"],
            key="voice_select",
            label_visibility="collapsed"
        )
        voice_map = {
            "British Female (Sonia)": "en-GB-SoniaNeural",
            "British Male (Ryan)": "en-GB-RyanNeural",
            "US Female (Jenny)": "en-US-JennyNeural",
            "US Male (Guy)": "en-US-GuyNeural"
        }
        st.session_state.selected_voice = voice_map.get(voice_option, "en-GB-SoniaNeural")
    
    st.markdown("---")
    
    # Audio input section
    st.markdown("### Voice Input")
    st.caption("Record or upload your question")
    
    # Real-time microphone input
    audio_value = st.audio_input("Click to record", key="mic_input")
    
    if audio_value:
        with st.spinner("Transcribing..."):
            transcribed = speech_to_text(audio_value.getvalue())
            if transcribed:
                st.session_state.transcribed_text = transcribed
                st.success(f"Heard: {transcribed}")
    
    # Show transcribed text and submit button
    if st.session_state.get('transcribed_text'):
        if st.button("Submit Voice Query", use_container_width=True, type="primary"):
            st.session_state.voice_query = st.session_state.transcribed_text
            st.session_state.transcribed_text = None
            st.rerun()
        if st.button("Clear Recording", use_container_width=True):
            st.session_state.transcribed_text = None
            st.rerun()
    
    st.markdown("---")
    
    # File upload as alternative
    with st.expander("Or upload audio file"):
        audio_file = st.file_uploader("Upload audio", type=['wav', 'mp3', 'm4a'], key="audio_upload", label_visibility="collapsed")
        if audio_file:
            with st.spinner("Transcribing..."):
                transcribed = speech_to_text(audio_file.getvalue())
                if transcribed:
                    st.success(f"Transcribed: {transcribed}")
                    if st.button("Use this query", key="use_file_query"):
                        st.session_state.voice_query = transcribed
                        st.rerun()
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN HEADER ---
st.markdown(f"""
<div class="header-container">
    <h1 class="header-title">EU DAILY INTELLIGENCE BRIEFING</h1>
    <p class="header-subtitle">
        {DATE_STR}
        <span class="status-badge">LIVE</span>
    </p>
</div>
""", unsafe_allow_html=True)

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        with st.chat_message("user"):
            st.write(msg['content'])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg['content'])
            
            # Display analyst thoughts if available
            if msg.get('thoughts'):
                with st.container():
                    st.markdown("---")
                    st.markdown("**Analyst Assessment**")
                    st.markdown(msg['thoughts'])
            
            # Display sources if available
            if msg.get('sources'):
                with st.expander("View Sources"):
                    for src in msg['sources']:
                        st.markdown(f"**{src['title']}**")
                        st.caption(f"{src['source']} | {src['date']}")
                        if src.get('link'):
                            st.markdown(f"[Read original]({src['link']})")
                        st.markdown("---")
            
            # Play audio if available
            if msg.get('audio'):
                st.audio(msg['audio'], format='audio/mp3')

# --- INPUT AREA ---
# Check for voice query from sidebar
voice_query = st.session_state.get('voice_query', None)
if voice_query:
    del st.session_state.voice_query
    user_input = voice_query
else:
    # Main input area with voice option
    col_input, col_voice = st.columns([5, 1])
    with col_input:
        user_input = st.chat_input("Ask about EU news, policy, or current events...")
    with col_voice:
        st.markdown("<br>", unsafe_allow_html=True)
        main_audio = st.audio_input("Voice", key="main_mic", label_visibility="collapsed")
        if main_audio and not st.session_state.get('main_audio_processed'):
            with st.spinner("Listening..."):
                transcribed = speech_to_text(main_audio.getvalue())
                if transcribed:
                    st.session_state.voice_query = transcribed
                    st.session_state.main_audio_processed = True
                    st.rerun()
        else:
            st.session_state.main_audio_processed = False

# --- PROCESS QUERY ---
if user_input and index is not None:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show processing status
    with st.status("Processing your query...", expanded=True) as status:
        st.write("Analyzing your question...")
        
        # Generate analysis with conversation history for context
        result = analyze_query(
            user_input, 
            news_data, 
            index, 
            items,
            query_type="auto",
            conversation_history=st.session_state.messages
        )
        
        query_type = result.get('query_type', 'detailed')
        if query_type == 'overview':
            st.write("Generating headlines briefing...")
        else:
            st.write("Generating detailed analysis...")
        
        # Generate voice if enabled
        audio_data = None
        if st.session_state.voice_enabled:
            st.write("Generating voice response...")
            try:
                selected_voice = st.session_state.get('selected_voice', 'en-GB-SoniaNeural')
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                audio_data = loop.run_until_complete(text_to_speech(result['analysis'], selected_voice))
                loop.close()
            except Exception as e:
                st.write(f"Voice generation skipped: {e}")
        
        status.update(label="Complete", state="complete", expanded=False)
    
    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['analysis'],
        "thoughts": result['thoughts'],
        "sources": result['sources'],
        "audio": audio_data
    })
    
    st.rerun()

# --- WELCOME MESSAGE ---
if not st.session_state.messages:
    pass  # No welcome message - clean interface

# --- FOOTER ---
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">
    EU Intelligence Briefing System | Database: {stats.get('articles', 0)} articles | {stats.get('chunks', 0)} indexed segments
</div>
""", unsafe_allow_html=True)
