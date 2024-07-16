from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import json


# # STEP 1. CONNECT TO ZILLIZ CLOUD
from pymilvus import MilvusClient


import pymilvus
print(f"pymilvus version: {pymilvus.__version__}")
from pymilvus import connections, utility
TOKEN = ""

# # Connect to Zilliz cloud using endpoint URI and API key TOKEN.
# # TODO change this.
CLUSTER_ENDPOINT=""
# CLUSTER_ENDPOINT="https://in03-48a5b11fae525c9.api.gcp-us-west1.zillizcloud.com:443"
connections.connect(
    alias='default',
     #  Public endpoint obtained from Zilliz Cloud
    uri=CLUSTER_ENDPOINT,
#   # API key or a colon-separated cluster username and password
    token=TOKEN,
)

# # Use no-schema Milvus client uses flexible json key:value format.
# # https://milvus.io/docs/using_milvusclient.md
mc = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    # API key or a colon-separated cluster username and password
    token=TOKEN)

# # Check if the server is ready and get colleciton name.
print(f"Type of server: {utility.get_server_version()}")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
]
schema = CollectionSchema(fields, "NVIDIA CUDA documentation chunks")

# Create collection
collection_name = "cuda_chunks"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# Load chunks data
with open('chunks.json', 'r') as f:
    chunks = json.load(f)

# Prepare data for Milvus
ids = []
embeddings = []
urls = []
texts = []

for i, chunk in enumerate(chunks):
    embeddings.append(chunk['embedding'])
    urls.append(chunk['url'])
    texts.append(chunk['text'])

# Insert data into Milvus
collection.insert([embeddings, urls, texts])

# Create HNSW index
index_params = {
    "metric_type": "L2",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 64},
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

