import json
import torch
from elasticsearch import Elasticsearch
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

# Initialize encoders and tokenizers
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Global variables
collection = None
es = None

def setup():
    global collection, es
    connections.connect("default", host="localhost", port="19530")
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': "http"}])
    
    collection_name = "documents"
    dim = 768  # DPR embedding dimension
    
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields, description=collection_name)
        collection = Collection(collection_name, schema)
        
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        collection.create_index("embedding", index)

def encode_passage(passage):
    inputs = context_tokenizer(passage, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        embeddings = context_encoder(**inputs).pooler_output
    return embeddings[0].numpy()

def index_documents(documents):
    global collection, es
    
    ids = []
    embeddings = []
    contents = []
    
    for doc in documents:
        doc_id = doc['id']
        content = doc['content']
        
        # Index in Elasticsearch
        es.index(index="documents", id=doc_id, body={"content": content})
        
        # Prepare for Milvus indexing
        embedding = encode_passage(content)
        
        ids.append(doc_id)
        embeddings.append(embedding.tolist())
        contents.append(content)
    
    # Index in Milvus
    collection.insert([ids, embeddings, contents])

def load_data(input_path):
    with open(input_path, 'r') as content:
        documents = json.load(content)
    return documents

if __name__ == "__main__":
    setup()
    
    input_path = "Steps/documents_export.json"
    documents = load_data(input_path)
    
    index_documents(documents)
    print("Database setup and data ingestion completed.")