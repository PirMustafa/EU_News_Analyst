"""
EU Daily Intelligence Briefing
A professional news analysis system with voice interaction
"""

import streamlit as st
import json
import os
from datetime import datetime
import tempfile
import asyncio
import numpy as np

# --- CONFIGURATION ---
CURRENT_DATE = datetime.now()
DATE_STR = CURRENT_DATE.strftime("%A, %d %B %Y")

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
    import google.generativeai as genai
    import faiss
    import pickle
    return genai, faiss, pickle

# --- API SETUP ---
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("CRITICAL: GOOGLE_API_KEY environment variable is missing. Please set it and restart.")
    st.stop()

genai, faiss, pickle = load_dependencies()
genai.configure(api_key=api_key)

# --- DATA LOADING ---
@st.cache_resource
def load_data():
    """Load FAISS index and embeddings data."""
    try:
        index = faiss.read_index("news_index.faiss")
        with open("items_with_embeddings.pkl", "rb") as f:
            items_with_embeddings = pickle.load(f)
        
        items = []
        for item in items_with_embeddings:
            items.append({
                "text": item["text"],
                "meta": item["metadata"]
            })
        
        # Load raw news data
        news_data = []
        if os.path.exists("eu_news_data.json"):
            with open("eu_news_data.json", 'r', encoding='utf-8') as f:
                news_data = json.load(f)
        
        unique_articles = len(set(item["metadata"]["title"] for item in items_with_embeddings))
        return index, items, news_data, {"articles": unique_articles, "chunks": len(items)}, "Online"
    except Exception as e:
        return None, [], [], {}, f"Error: {str(e)}"

# --- EMBEDDING FUNCTION ---
def get_embedding(text):
    """Generate embedding for text."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# --- VOICE FUNCTIONS ---
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

# --- INTELLIGENCE ANALYST FUNCTION ---
def analyze_query(query, news_data, index, items):
    """Generate detailed intelligence analysis with analyst thoughts."""
    
    # Get today's news
    todays_news = [a for a in news_data if a.get('date', '').strip() == DATE_STR]
    
    # Also search FAISS for relevant historical context
    query_emb = get_embedding(query)
    relevant_chunks = []
    if query_emb and index:
        query_array = np.array([query_emb], dtype=np.float32)
        distances, indices = index.search(query_array, 10)
        for idx in indices[0]:
            if idx < len(items):
                relevant_chunks.append(items[idx])
    
    # Build comprehensive context
    if not todays_news and not relevant_chunks:
        return {
            "analysis": f"No news data available for {DATE_STR}. The archive may need to be updated by running the scraper.",
            "thoughts": "Unable to provide analysis without current news data.",
            "sources": []
        }
    
    # Create detailed context
    context_parts = []
    sources = []
    
    # Add today's full articles
    for article in todays_news[:8]:
        context_parts.append(f"""
ARTICLE: {article.get('title', 'Untitled')}
SOURCE: {article.get('source', 'Unknown')}
DATE: {article.get('date', DATE_STR)}
CONTENT: {article.get('content', '')}
---""")
        sources.append({
            'title': article.get('title', 'Untitled'),
            'source': article.get('source', 'Unknown'),
            'date': article.get('date', DATE_STR),
            'link': article.get('link', '')
        })
    
    # Add relevant historical chunks for context
    for chunk in relevant_chunks[:5]:
        if chunk['meta'].get('title') not in [s['title'] for s in sources]:
            context_parts.append(f"""
RELATED CONTEXT: {chunk['meta'].get('title', 'Untitled')}
DATE: {chunk['meta'].get('date', 'Unknown')}
EXCERPT: {chunk['text'][:500]}
---""")
    
    full_context = "\n".join(context_parts)
    
    # Analyst prompt for detailed analysis with thoughts
    analyst_prompt = f"""You are a senior intelligence analyst at a European policy institute. 
Your role is to provide detailed, nuanced briefings on EU affairs.

TODAY'S DATE: {DATE_STR}

INTELLIGENCE BRIEFING MATERIALS:
{full_context}

USER QUERY: {query}

Provide your response in the following structure:

1. EXECUTIVE SUMMARY
Start with a clear, concise summary of the key points relevant to the query.

2. DETAILED ANALYSIS
Provide in-depth analysis covering:
- Key developments and their significance
- Stakeholders involved and their positions
- Policy implications
- Economic or social impacts where relevant

3. CONTEXTUAL BACKGROUND
Explain any necessary background information that helps understand the current developments.

4. FORWARD OUTLOOK
What are the likely next steps or developments to watch?

IMPORTANT GUIDELINES:
- Be thorough and detailed - this is a professional intelligence briefing
- Use formal, analytical language
- Do not use emojis or casual expressions
- Connect dots between different pieces of information
- Highlight contradictions or tensions if they exist
- Be specific with facts, figures, and dates mentioned in the sources
- If information is limited, acknowledge gaps in intelligence"""

    thoughts_prompt = f"""You are a senior intelligence analyst reviewing the following news materials.

NEWS MATERIALS:
{full_context}

USER QUERY: {query}

Based on these materials, share your professional assessment:

1. What patterns or trends do you observe in the reporting?
2. What are the underlying dynamics at play?
3. What aspects deserve closer monitoring going forward?
4. What questions remain unanswered by the available intelligence?

Keep your thoughts concise but insightful. Do not use emojis. Be analytical and professional."""

    try:
        # Generate main analysis
        model = genai.GenerativeModel("gemini-2.0-flash")
        analysis_response = model.generate_content(analyst_prompt)
        
        # Generate analyst thoughts
        thoughts_response = model.generate_content(thoughts_prompt)
        
        return {
            "analysis": analysis_response.text,
            "thoughts": thoughts_response.text,
            "sources": sources
        }
    except Exception as e:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            analysis_response = model.generate_content(analyst_prompt)
            thoughts_response = model.generate_content(thoughts_prompt)
            return {
                "analysis": analysis_response.text,
                "thoughts": thoughts_response.text,
                "sources": sources
            }
        except Exception as e2:
            return {
                "analysis": f"Analysis generation failed: {str(e2)}",
                "thoughts": "Unable to generate analyst thoughts.",
                "sources": sources
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
        st.write("Searching intelligence database...")
        
        # Generate analysis
        result = analyze_query(user_input, news_data, index, items)
        st.write("Analysis complete.")
        
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
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #64748b;">
        <h3>Welcome to the EU Intelligence Briefing System</h3>
        <p>Ask questions about European Union news, policy developments, economic updates, or current events.</p>
        <p style="font-size: 0.9rem; margin-top: 1rem;">
            Examples:<br>
            "What are the latest developments in EU technology regulation?"<br>
            "Summarize today's major EU policy announcements"<br>
            "What is happening with EU-US trade relations?"
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #94a3b8; font-size: 0.8rem;">
    EU Intelligence Briefing System | Database: {stats.get('articles', 0)} articles | {stats.get('chunks', 0)} indexed segments
</div>
""", unsafe_allow_html=True)
