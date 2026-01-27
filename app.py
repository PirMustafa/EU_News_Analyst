import streamlit as st
import json
import google.generativeai as genai
import faiss
import numpy as np
import os
import speech_recognition as sr
from io import BytesIO
import time
import asyncio
import edge_tts
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import base64
from datetime import datetime

# --- 1. DYNAMIC CONFIGURATION ---
# Sets "Today" to the actual current date automatically
CURRENT_DATE = datetime.now()
DATE_STR = CURRENT_DATE.strftime("%A, %d %B %Y")  # e.g., "Sunday, 18 January 2026"

st.set_page_config(
    page_title=f"EU Intel • {CURRENT_DATE.year}",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. CYBERPUNK UI THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 100%); color: #e2e8f0; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d1f 0%, #1a0a2e 100%); border-right: 1px solid rgba(139, 92, 246, 0.2); }
    
    .chat-bubble { padding: 1.5rem; border-radius: 12px; margin-bottom: 1rem; line-height: 1.6; position: relative; }
    .user-bubble { background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%); color: white; margin-left: 15%; border-radius: 12px 12px 0 12px; box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3); }
    .assistant-bubble { background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(0, 212, 255, 0.2); color: #e2e8f0; margin-right: 15%; border-radius: 12px 12px 12px 0; }
    
    .main-header { text-align: center; margin-bottom: 20px; padding: 20px; border-bottom: 1px solid rgba(0, 212, 255, 0.1); }
    h1 { color: #00d4ff; font-family: 'JetBrains Mono', monospace; letter-spacing: -1px; }
    p.status { color: #94a3b8; font-size: 0.9em; margin-top: -10px; }
    
    div[data-testid="stMetric"] { background: rgba(15, 15, 30, 0.5); border-radius: 8px; padding: 10px; border: 1px solid rgba(139, 92, 246, 0.2); }
    
    /* Standard Audio Player Styling */
    audio { width: 100%; margin-top: 5px; opacity: 0.9; }
    </style>
""", unsafe_allow_html=True)

# --- 3. API SETUP ---
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    st.error("❌ CRITICAL: GOOGLE_API_KEY environment variable is missing.")
    st.stop()
genai.configure(api_key=api_key)

# --- 4. DATA ENGINE (ROBUST & DYNAMIC) ---
def get_embedding_safe(text):
    """Tries multiple models to find one that works."""
    models = ["models/text-embedding-004", "models/embedding-001"]
    for model in models:
        try:
            result = genai.embed_content(model=model, content=text, task_type="retrieval_document")
            if 'embedding' in result: return result['embedding']
            elif 'embeddings' in result: return result['embeddings'][0]
        except: continue
    return None

@st.cache_resource
def load_and_index_data():
    """Loads Data. Handles Missing Files Gracefully."""
    
    news_data = []
    if os.path.exists("eu_news_data.json"):
        try:
            with open("eu_news_data.json", 'r', encoding='utf-8') as f:
                news_data = json.load(f)
        except: pass

    # FAIL-SAFE: If no data exists, create a tiny backup dataset so app works
    if not news_data:
        news_data = [
            {"title": "System Initialization", "date": DATE_STR, "content": "The EU Intelligence Hub has been initialized. Archive is currently empty. Please run scraper."},
        ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    items = []
    for article in news_data:
        full_text = f"Title: {article.get('title')}\nDate: {article.get('date')}\n\n{article.get('content')}"
        for chunk in text_splitter.split_text(full_text):
            items.append({"text": chunk, "meta": article})

    # Embed
    valid_embeddings = []
    valid_items = []
    
    # Progress Bar (Sidebar)
    progress_bar = st.sidebar.progress(0)
    
    for i, item in enumerate(items):
        # Rate limit protection
        time.sleep(0.01)
        emb = get_embedding_safe(item['text'])
        
        if emb and len(emb) == 768:
            valid_embeddings.append(emb)
            valid_items.append(item)
        
        if i % 10 == 0:
            progress_bar.progress(min(i / len(items), 1.0))

    progress_bar.empty()

    if not valid_embeddings:
        return None, [], {}, "API Error: Embeddings Failed"

    # Index
    emb_array = np.array(valid_embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(768)
    index.add(emb_array)
    
    stats = {"articles": len(news_data), "chunks": len(valid_items)}
    return index, valid_items, stats, "Operational"

# --- 5. NEURAL VOICE ENGINE (NATURAL) ---
async def generate_neural_voice(text):
    """Generates high-quality Neural voice using Edge TTS."""
    # CLEANER: Remove sources, markdown, and symbols
    clean_text = text.split("**Sources:**")[0].split("Sources:")[0]
    clean_text = clean_text.replace("**", "").replace("*", "").replace("#", "")
    
    # Remove URLs via Regex
    import re
    clean_text = re.sub(r'http\S+', '', clean_text)
    clean_text = clean_text[:800] # Limit for speed
    
    voice = "en-US-AriaNeural" # Professional Neural Voice
    communicate = edge_tts.Communicate(clean_text, voice)
    
    fp = BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            fp.write(chunk["data"])
    fp.seek(0)
    return fp

# --- 6. SPEECH TRANSCRIPTION ---
def transcribe_audio_safe(audio_bytes):
    r = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes.getvalue())
            f.close()
            with sr.AudioFile(f.name) as source:
                r.adjust_for_ambient_noise(source)
                return r.recognize_google(r.record(source))
    except: return None

# --- 7. ANALYST LOGIC (THE BRAIN) ---
def get_answer(query, index, items):
    try:
        # Search
        q_vec = get_embedding_safe(query)
        if not q_vec: return "Error: Could not embed query (API Limit).", []
        
        D, I = index.search(np.array([q_vec], dtype=np.float32), 5)
        
        context = ""
        sources = []
        seen = set()
        for i in I[0]:
            if i < len(items):
                meta = items[i]['meta']
                title = meta.get('title', 'Unknown')
                if title not in seen:
                    sources.append(meta)
                    seen.add(title)
                context += f"- {title} ({meta.get('date')}): {items[i]['text'][:300]}...\n"

        # ANALYST PROMPT
        sys_prompt = f"""You are an Elite Intelligence Analyst.
        TODAY'S DATE: {DATE_STR}.
        
        CONTEXT NEWS:
        {context}
        
        USER QUERY: {query}
        
        INSTRUCTIONS:
        1. **Bottom Line Up Front (BLUF):** Start directly with the answer. Do not use phrases like "Based on the provided text".
        2. **Natural Tone:** Speak like a human colleague briefing a senior officer.
        3. **Timeliness:** If asked for "News", prioritize events dated {DATE_STR} or recent days.
        4. **Sources:** Do NOT mention source names in the text (they are listed separately).
        """
        
        # Try Gemini 2.0 -> Fallback to 1.5
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(sys_prompt)
        except:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(sys_prompt)
            
        return response.text, sources
    except Exception as e:
        return f"System Error: {str(e)}", []

# --- 8. MAIN UI LAYOUT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "audio_trigger" not in st.session_state: st.session_state.audio_trigger = None

# Load Data
index, items, stats, status = load_and_index_data()
is_ready = (index is not None)

# SIDEBAR
with st.sidebar:
    st.header("🎛️ Command Center")
    col1, col2 = st.columns(2)
    col1.metric("Articles", stats.get("articles", 0))
    col2.metric("Chunks", stats.get("chunks", 0))
    
    if is_ready: st.success(f"System: {status}")
    else: st.error(f"System: {status}")
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat"): 
        st.session_state.messages = []
        st.session_state.audio_trigger = None
        st.rerun()

# HEADER
st.markdown(f"""
<div class="main-header">
    <h1> EU INTELLIGENCE HUB</h1>
    <p class="status">STATUS: ONLINE | DATE: {DATE_STR}</p>
</div>
""", unsafe_allow_html=True)

# --- RENDER CHAT HISTORY ---
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg['role'] == "user" else "assistant-bubble"
    st.markdown(f"<div class='chat-bubble {role_class}'><b>{msg['role'].upper()}</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    
    # Render historical audio (Standard Player, No Autoplay)
    if msg.get("audio_data"):
        st.audio(msg["audio_data"], format="audio/mp3")

# --- ONE-SHOT AUDIO TRIGGER (Plays ONCE then self-destructs) ---
if st.session_state.audio_trigger:
    b64 = base64.b64encode(st.session_state.audio_trigger).decode()
    md = f"""
        <audio autoplay="true" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
    st.session_state.audio_trigger = None # Reset immediately

# --- INPUT AREA ---
col1, col2 = st.columns([0.85, 0.15])
with col1: text_input = st.chat_input("Query the archive...")
with col2: audio_input = st.audio_input("Mic")

# --- INPUT HANDLING ---
query = None
input_type = None # "text" or "audio"

if text_input:
    query = text_input
    input_type = "text"
elif audio_input:
    # Hash check to prevent infinite loops on UI refresh
    audio_hash = hash(audio_input.getvalue())
    if "last_audio_hash" not in st.session_state or st.session_state.last_audio_hash != audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with st.spinner("🎙️ Transcribing..."):
            query = transcribe_audio_safe(audio_input)
            input_type = "audio"

# --- EXECUTION LOGIC ---
if query:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # 2. Assistant Response
    with st.spinner("🔍 Analyzing intelligence..."):
        ans, srcs = get_answer(query, index, items)
        
        # Format Sources (Text Only)
        src_text = ""
        if srcs: 
            src_text = "\n\n**Sources:**\n" + "\n".join([f"- {s.get('title')}" for s in srcs])
        
        final_text = ans + src_text
        
        # 3. Audio Generation (ONLY IF SPOKEN INPUT)
        audio_bytes = None
        if input_type == "audio":
            with st.spinner("🔊 Synthesizing Voice..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                fp = loop.run_until_complete(generate_neural_voice(ans))
                audio_bytes = fp.getvalue()
                
                # Set Trigger for Autoplay
                st.session_state.audio_trigger = audio_bytes
        
        # 4. Save to History
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_text,
            "audio_data": audio_bytes
        })
        
    st.rerun()