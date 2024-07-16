import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load scraped data
with open('output.json', 'r') as f:
    data = json.load(f)

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_data(text, max_chunk_size=100):
    sentences = text.split('.')
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= max_chunk_size:
            chunks.append('.'.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append('.'.join(current_chunk))

    return chunks

chunks = []
for item in data:
    url = item['url']
    text_chunks = chunk_data(item['text'])
    for chunk in text_chunks:
        embedding = model.encode(chunk)
        chunks.append({
            'url': url,
            'text': chunk,
            'embedding': embedding
        })

# Save chunks and embeddings
with open('chunks.json', 'w') as f:
    json.dump(chunks, f, indent=4, default=lambda x: x.tolist())
