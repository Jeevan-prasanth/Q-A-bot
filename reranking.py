from rank_bm25 import BM25Okapi
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from nltk.corpus import wordnet
import numpy as np
import json
import torch
import nltk
from sentence_transformers import SentenceTransformer, util


# Ensure you have downloaded the NLTK WordNet corpus
nltk.download('wordnet')

# Connect to Milvus
from pymilvus import MilvusClient, connections, utility, Collection

import pymilvus
print(f"pymilvus version: {pymilvus.__version__}")

# Milvus connection details
TOKEN = ""
CLUSTER_ENDPOINT = ""

# Set up gRPC channel options for maximum message size
grpc_channel_options = [
    ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50 MB
    ("grpc.max_receive_message_length", 50 * 1024 * 1024)  # 50 MB
]

connections.connect(
    alias='default',
    uri=CLUSTER_ENDPOINT,
    token=TOKEN,
    options=grpc_channel_options
)

mc = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN
)

# Check server version
print(f"Type of server: {utility.get_server_version()}")

# Load collection
collection = Collection("cuda_chunks")
collection.load()

# Retrieve data from Milvus in smaller batches
batch_size = 50  # Further reduced batch size
offset = 0
results = []

while True:
    print(f"Querying batch with offset {offset} and limit {batch_size}")
    batch = collection.query(expr="id >= 0", output_fields=["url", "text"], offset=offset, limit=batch_size)
    if not batch:
        break
    results.extend(batch)
    offset += batch_size
    print(f"Retrieved {len(batch)} records")

# Extract data from Milvus results
documents = [result['text'] for result in results]
urls = [result['url'] for result in results]

# BM25 Retrieval
tokenized_corpus = [doc.split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
from transformers import AutoTokenizer, AutoModel

# BERT-based Retrieval
print("Loading DPRQuestionEncoderTokenizer and DPRQuestionEncoder model...")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load document embeddings from JSON file
with open('chunks.json', 'r') as f:
    chunk_data = json.load(f)

document_embeddings = [chunk["embedding"] for chunk in chunk_data]

# Query Expansion
def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    expanded_query = query + " " + " ".join(synonyms)
    return expanded_query

# Hybrid Retrieval
def hybrid_retrieval(query):
    expanded_query = expand_query(query)

    # BM25 Scores
    bm25_scores = bm25.get_scores(expanded_query.split())

    # BERT Scores
    inputs = tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model(**inputs).pooler_output.numpy().squeeze()
    dpr_scores = [np.dot(query_embedding, np.array(doc_embedding)) for doc_embedding in document_embeddings]

    # Combine Scores
    combined_scores = bm25_scores + dpr_scores  # This is a simple combination, can be weighted
    ranked_indices = np.argsort(combined_scores)[::-1]

    # Re-rank results
    ranked_results = [(documents[i], urls[i], combined_scores[i]) for i in ranked_indices[:5]]
    return ranked_results

# Example Query
if __name__ == "__main__":
    query = "How to install CUDA"
    print("Performing hybrid retrieval...")
    results = hybrid_retrieval(query)

    # Display Top Results
    for result in results[:5]:
        print(f"URL: {result[1]}")
        print(f"Score: {result[2]}")
        print(f"Text: {result[0]}")  # Displaying only the first 200 characters for brevity
        print("\n")
