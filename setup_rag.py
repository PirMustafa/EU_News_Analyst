import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyC4_yBb3B7Zo0NHpioVIlPqP2k3em-WQBI'

import json
import google.generativeai as genai
import faiss
import numpy as np
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure API
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Load data
print('📥 Loading EU news data...')
with open('eu_news_data.json', 'r', encoding='utf-8') as f:
    news_data = json.load(f)
print(f'✅ Loaded {len(news_data)} articles')

# Process data
print('🔄 Processing articles into chunks...')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

items = []
for article in news_data:
    full_text = f"Date: {article['date']}\nTitle: {article['title']}\n\n{article['content']}"
    chunks = text_splitter.split_text(full_text)

    for chunk in chunks:
        items.append({
            'type': 'text',
            'text': chunk,
            'metadata': {
                'url': article.get('link', ''),
                'title': article['title'],
                'date': article['date']
            }
        })

print(f'✅ Created {len(items)} text chunks')

# Generate embeddings
print('🧠 Generating embeddings...')
embedding_vector_dimension = 768

def generate_multimodal_embeddings(prompt, output_embedding_length=768):
    result = genai.embed_content(
        model='models/text-embedding-004',
        content=prompt,
        task_type='retrieval_document',
        output_dimensionality=output_embedding_length
    )
    return result['embedding']

with tqdm(total=len(items), desc='Generating Embeddings') as pbar:
    for item in items:
        item['embedding'] = generate_multimodal_embeddings(
            prompt=item['text'],
            output_embedding_length=embedding_vector_dimension
        )
        pbar.update(1)

print('✅ Embeddings generated')

# Build FAISS index
print('🏗️ Building FAISS index...')
all_embeddings = np.array([item['embedding'] for item in items], dtype=np.float32)
index = faiss.IndexFlatL2(embedding_vector_dimension)
index.add(all_embeddings)

print(f'✅ FAISS index built with {index.ntotal} vectors')

# Save index and metadata
import pickle
faiss.write_index(index, 'news_index.faiss')
with open('items_with_embeddings.pkl', 'wb') as f:
    pickle.dump(items, f)

items_metadata = [{k: v for k, v in item.items() if k != 'embedding'} for item in items]
with open('items_metadata.pkl', 'wb') as f:
    pickle.dump(items_metadata, f)

print('💾 Index and data saved')
print('🎉 RAG system ready!')