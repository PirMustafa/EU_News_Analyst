"""
Shared configuration and utilities for the EU News Analyst pipeline.
Used by scrape_eu_news.py, build_index.py, and app.py.
"""

import os

# --- FILE PATHS ---
DATA_FILE = "eu_news_data.json"
INDEX_FILE = "news_index.faiss"
ITEMS_FILE = "items.json"

# --- EMBEDDINGS ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- DATES ---
# Display format used for article dates, e.g. "Wednesday, 19 August 2026"
DISPLAY_DATE_FORMAT = "%A, %d %B %Y"


def load_embedding_model():
    """Load the sentence-transformers embedding model on CPU."""
    # Force CPU to avoid GPU compatibility issues
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed_text(model, text):
    """Generate an embedding vector (as a list) for a single text."""
    return model.encode(text).tolist()


def dedupe_by(items, key):
    """Remove duplicates from a list, keeping the first occurrence of each key."""
    seen = set()
    unique = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            unique.append(item)
    return unique
