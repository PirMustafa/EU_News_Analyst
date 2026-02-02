"""
Build FAISS Index from EU News Data
====================================
Generates embeddings for all articles and creates a FAISS index for semantic search.
"""

import os
import json
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from tqdm import tqdm

# Configuration
API_KEY = "AIzaSyARdSFJJChAWGAv38mICrYohancGw0YIG8"
DATA_FILE = "eu_news_data.json"
INDEX_FILE = "news_index.faiss"
EMBEDDINGS_FILE = "items_with_embeddings.pkl"
METADATA_FILE = "items_metadata.pkl"
EMBEDDING_DIM = 768
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Configure API
os.environ["GOOGLE_API_KEY"] = API_KEY
genai.configure(api_key=API_KEY)

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

def generate_embedding(text, retries=3):
    """Generate embedding for text using Google's API."""
    for attempt in range(retries):
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
                output_dimensionality=EMBEDDING_DIM
            )
            return result['embedding']
        except Exception as e:
            if attempt < retries - 1:
                import time
                time.sleep(1)
            else:
                print(f"Failed to embed text: {e}")
                return None
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
    
    # Generate embeddings
    print(f"\n🧠 Generating embeddings (this may take a few minutes)...")
    failed_count = 0
    
    with tqdm(total=len(items), desc="Embeddings") as pbar:
        for item in items:
            embedding = generate_embedding(item['text'])
            if embedding:
                item['embedding'] = embedding
            else:
                failed_count += 1
                # Use zero vector as fallback
                item['embedding'] = [0.0] * EMBEDDING_DIM
            pbar.update(1)
    
    if failed_count > 0:
        print(f"   ⚠️ {failed_count} embeddings failed")
    
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
