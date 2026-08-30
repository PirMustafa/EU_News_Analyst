"""
Build FAISS Index from EU News Data
====================================
Generates embeddings for all articles and creates a FAISS index for semantic search.
Uses sentence-transformers (local, no API key needed) for embeddings.
"""

import json
import sys
import logging
import numpy as np
import faiss
from tqdm import tqdm

from common import (
    DATA_FILE,
    INDEX_FILE,
    ITEMS_FILE,
    EMBEDDING_DIM,
    embed_text,
    load_embedding_model,
)

# Windows consoles default to cp1252, which cannot encode the emoji used in the
# progress output below; force UTF-8 so the script runs without PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - stream may not support it
            pass

logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Load embedding model once
print("Loading embedding model...")
embed_model = load_embedding_model()

def simple_text_splitter(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start + overlap >= len(text):
            break
    return chunks

def generate_embedding(text):
    """Generate embedding for text using local sentence-transformers model."""
    try:
        return embed_text(embed_model, text)
    except Exception as e:
        print(f"Failed to embed text: {e}")
        return None

def main():
    print("=" * 60)
    print("🔨 BUILDING FAISS INDEX")
    print("=" * 60)

    # Load news data
    print(f"\n📂 Loading data from {DATA_FILE}...")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_FILE}. Run scrape_eu_news.py first.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {DATA_FILE}: {e}. Fix the file or run the scraper again.")
        sys.exit(1)
    except (OSError, UnicodeError) as e:
        print(f"❌ Could not read {DATA_FILE}: {e}")
        sys.exit(1)
    print(f"   Loaded {len(news_data)} articles")
    
    # Process articles into chunks
    print(f"\n📝 Processing articles into chunks...")
    items = []
    skipped_articles = 0
    for article_number, article in enumerate(news_data, start=1):
        if (
            not isinstance(article, dict)
            or not all(field in article for field in ("date", "title", "content"))
            or not all(isinstance(article[field], str) and article[field].strip()
                       for field in ("date", "title", "content"))
        ):
            skipped_articles += 1
            print(f"   ⚠️ Skipping malformed article {article_number}: expected date, title, and content.")
            continue

        # Combine fields for embedding
        full_text = f"Date: {article['date']}\nTitle: {article['title']}\n\n{article['content']}"
        
        # Split into chunks
        chunks = simple_text_splitter(full_text)
        
        for chunk in chunks:
            items.append({
                "type": "text",
                "text": chunk,
                "metadata": {
                    "url": article.get('link', ''),
                    "title": article['title'],
                    "date": article['date'],
                    "source": article.get('source', 'EU Commission')
                }
            })
    
    print(f"   Created {len(items)} text chunks")
    print(f"   Skipped malformed articles: {skipped_articles}")

    if not items:
        print("❌ No valid article chunks were created; index build aborted.")
        sys.exit(1)
    
    # Generate embeddings in batches for speed
    print(f"\n🧠 Generating embeddings (local model, no API needed)...")
    texts = [item['text'] for item in items]
    batch_size = 64
    all_embeddings_list = []

    with tqdm(total=len(texts), desc="Embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_end = i + len(batch) - 1
            try:
                embeddings = embed_model.encode(batch, show_progress_bar=False)
            except Exception as e:
                logger.exception("Embedding batch %s-%s failed.", i, batch_end)
                print(f"❌ Embedding batch {i}-{batch_end} failed: {e}")
                sys.exit(1)
            all_embeddings_list.extend(embeddings.tolist())
            pbar.update(len(batch))

    for item, emb in zip(items, all_embeddings_list):
        item['embedding'] = emb

    # Build FAISS index
    print(f"\n🔍 Building FAISS index...")
    all_embeddings = np.array([item['embedding'] for item in items], dtype=np.float32)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(all_embeddings)
    
    print(f"   Index contains {index.ntotal} vectors")
    
    # Save everything
    print(f"\n💾 Saving files...")
    
    # Save FAISS index
    try:
        faiss.write_index(index, INDEX_FILE)
    except Exception as e:
        logger.exception("Failed to write %s.", INDEX_FILE)
        print(f"❌ Failed to write {INDEX_FILE}: {e}")
        sys.exit(1)
    print(f"   ✅ {INDEX_FILE}")
    
    # Save chunk text and metadata separately from the embeddings.
    items_data = [{key: value for key, value in item.items() if key != "embedding"}
                  for item in items]
    try:
        with open(ITEMS_FILE, "w", encoding="utf-8") as f:
            json.dump(items_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Failed to write %s.", ITEMS_FILE)
        print(f"❌ Failed to write {ITEMS_FILE}: {e}")
        sys.exit(1)
    print(f"   ✅ {ITEMS_FILE}")
    
    print("\n" + "=" * 60)
    print("✅ INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"   Articles: {len(news_data)}")
    print(f"   Chunks: {len(items)}")
    print(f"   Vectors: {index.ntotal}")
    print("=" * 60)

if __name__ == "__main__":
    main()
