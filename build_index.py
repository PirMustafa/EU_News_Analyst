"""
Build FAISS Index from EU News Data
====================================
Generates embeddings for all articles and creates a FAISS index for semantic search.
Uses sentence-transformers (local, no API key needed) for embeddings.
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configuration
DATA_FILE = "eu_news_data.json"
INDEX_FILE = "news_index.faiss"
EMBEDDINGS_FILE = "items_with_embeddings.pkl"
METADATA_FILE = "items_metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Force CPU to avoid GPU compatibility issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load embedding model once
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

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
        return embed_model.encode(text).tolist()
    except Exception as e:
        print(f"Failed to embed text: {e}")
        return None

def main():
    print("=" * 60)
    print("🔨 BUILDING FAISS INDEX")
    print("=" * 60)
    
    # Load news data
    print(f"\n📂 Loading data from {DATA_FILE}...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    print(f"   Loaded {len(news_data)} articles")
    
    # Process articles into chunks
    print(f"\n📝 Processing articles into chunks...")
    items = []
    for article in news_data:
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
    
    # Generate embeddings in batches for speed
    print(f"\n🧠 Generating embeddings (local model, no API needed)...")
    texts = [item['text'] for item in items]
    batch_size = 64
    all_embeddings_list = []

    with tqdm(total=len(texts), desc="Embeddings") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = embed_model.encode(batch, show_progress_bar=False)
            all_embeddings_list.extend(embeddings.tolist())
            pbar.update(len(batch))

    for item, emb in zip(items, all_embeddings_list):
        item['embedding'] = emb
    failed_count = 0
    
    # Build FAISS index
    print(f"\n🔍 Building FAISS index...")
    all_embeddings = np.array([item['embedding'] for item in items], dtype=np.float32)
    
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(all_embeddings)
    
    print(f"   Index contains {index.ntotal} vectors")
    
    # Save everything
    print(f"\n💾 Saving files...")
    
    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)
    print(f"   ✅ {INDEX_FILE}")
    
    # Save items with embeddings
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(items, f)
    print(f"   ✅ {EMBEDDINGS_FILE}")
    
    # Save metadata only (without embeddings, for lighter loading)
    items_metadata = [{k: v for k, v in item.items() if k != 'embedding'} for item in items]
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(items_metadata, f)
    print(f"   ✅ {METADATA_FILE}")
    
    print("\n" + "=" * 60)
    print("✅ INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"   Articles: {len(news_data)}")
    print(f"   Chunks: {len(items)}")
    print(f"   Vectors: {index.ntotal}")
    print("=" * 60)

if __name__ == "__main__":
    main()
