import streamlit as st
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import nltk
from nltk.corpus import wordnet
from langchain_core.runnables import RunnableSequence

from pymilvus import connections, Collection
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

from langchain.chains import LLMChain
import os

from huggingface_hub import login
#your huggingface token
login(token="")
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

# BERT-based Retrieval
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

# Generate Answer
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_answer(query, context):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

    prompt_template = """
    In the given context, analyze the data and provide the relevant answer to the query. Understand the question first and extract the relevant information from the context to answer it in as detailed manner as possible.
    Instructions:
    Comprehend the Question: Ensure you fully understand what the question is asking.
    Extract Relevant Information: Identify and extract the key pieces of information from the context that directly relate to the question.
    Formulate a Detailed Response: Using the extracted information, provide a comprehensive and well-structured answer to the question, ensuring all relevant points are covered.
    Context: {context}
    Question: {question}
    Answer: 
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    response = chain({"input_documents": context, "question": query}, return_only_outputs=True)
    return response["output_text"]

# Set up the Contextual Compression Retriever


if __name__ == "__main__":
    # Streamlit UI
    st.title("NVIDIA CUDA Documentation Search")

    query = st.text_input("Enter your query:")
    if st.button("Search") and query:
        results = hybrid_retrieval(query)
        c=0
        context = [Document(page_content=result[0], metadata={"url": result[1]}) for result in results]
        #print(context)
        #compressed_context = get_compressed_context(context)
        answer = generate_answer(query, context)

        st.subheader("Answer:")
        st.write(answer)
        
