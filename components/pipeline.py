from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from sentence_transformers import SentenceTransformer
import torch
import math
import json
from elasticsearch import Elasticsearch
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np
from collections import Counter

#--------------------------------------------------------------
"""
Todo:
 - Add the connection and collection as global variable or define a class and set them as class variable, as these will be required to pass again and again to many functions. 
 - You have to put the input content in the following format:
    {"id":"i", "content": "this is the content"}
 - Then after the embeddings are generated this should be the format:
    {"id":"i", "content": "this is the content", "vector":"0u34u3u"}
"""

#-----------------------------------------------------------------------------------
# Load models
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
cross_encoder = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L-6-v2')



# setup--------------------------------------------------------------
connections.connect("default", host="localhost", port="19530")
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# Connect to Milvus
# def connect_to_milvus(host="localhost", port="19530"):
#     try:
#         connections.connect("default", host=host, port=port)
#         print(f"Successfully connected to Milvus")
#     except Exception as e:
#         print(f"Failed to connect to Milvus: {e}")


# def connect_to_elasticsearch(host="localhost", port="9200"):
#     try:
#         es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
#         print(f"Successfully connected to Milvus")
#     except Exception as e:
#         print(f"Failed to connect to Milvus: {e}")  
#     return es

def create_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        # FieldSchema(name="tokens", dtype=DataType.JSON),  # Store tokenized text
        # FieldSchema(name="doc_length", dtype=DataType.INT64),  # Document length
    ]
    schema = CollectionSchema(fields, "MyCollection")
    collection = Collection(collection_name, schema)
    
    # Create an IVF_FLAT index for the collection
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index("milvus_index", index)
    return collection

def encode_passage(passage):
    inputs = context_tokenizer(passage, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        embeddings = context_encoder(**inputs).pooler_output
    return embeddings[0].numpy()


def insert_data(collection, content):
    embeddings = encode_passage(content)
    doc_id = list(range(1, len(content) + 1))
    # tokens_list = [text.split() for text in texts]  # Simple tokenization
    # doc_lengths = [len(tokens) for tokens in tokens_list]
    # token_frequencies = [dict(Counter(tokens)) for tokens in tokens_list]
    
    entities = [
        [embedding.tolist() for embedding in embeddings],
        content,
        doc_id,
        # token_frequencies,
        # doc_lengths
    ]
    collection.insert(entities)

def index_document(doc_id, content):
    # Index in Elasticsearch for BM25
    es.index(index="documents", id=doc_id, body={"content": content})

    # Index in Milvus for DPR
    embedding = encode_passage(content)
    collection.insert([[doc_id], [embedding.tolist()]])




# Query Expansion----------------------------------------------------------------------
# query expansion 
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def expand_query_with_keywords(query, num_expansions=3, num_keywords=5):
    # Generate expanded queries
    input_text = f"expand query: {query}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=num_expansions,
        num_beams=num_expansions,
        temperature=0.7
    )

    expanded_queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    # Generate keywords
    keyword_input = f"generate keywords for: {query}"
    keyword_ids = tokenizer(keyword_input, return_tensors="pt").input_ids

    keyword_outputs = model.generate(
        keyword_ids,
        max_length=30,
        num_return_sequences=1,
        num_beams=num_keywords,
        temperature=0.7
    )

    keywords = tokenizer.decode(keyword_outputs[0], skip_special_tokens=True).split()

    # Combine original query, expanded queries, and keywords
    final_queries = [query] + expanded_queries
    final_queries = [f"{q} {' '.join(keywords)}" for q in final_queries]

    return final_queries

# Example usage
# original_query = "What is the capital of France?"
# expanded_queries = expand_query_with_keywords(original_query)
# print("Original query:", original_query)
# print("Expanded queries with keywords:")
# for i, q in enumerate(expanded_queries):
#     print(f"{i+1}. {q}")



# Retrieval-----------------------------------------------------------------------

def load_collection(collection):
    collection.load()
    print(f"Collection '{collection.name}' loaded into memory")

def encode_query(query):
    input_ids = question_tokenizer(query, return_tensors='pt')['input_ids']
    with torch.no_grad():
        embeddings = question_encoder(input_ids).pooler_output
    return embeddings[0].numpy()

def dpr_search(query, top_k=100):
    query_vector = encode_query(query)
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id"]
    )
    return [(hit.entity.get('id'), hit.score) for hit in results[0]]


def bm25_search(query, top_k=100):
    response = es.search(index="documents", body={
        "query": {"match": {"content": query}},
        "size": top_k
    })
    return [(hit['_id'], hit['_score']) for hit in response['hits']['hits']]


def hybrid_search(query, top_k=100, alpha=0.5):
    bm25_results = bm25_search(query, top_k)
    dpr_results = dpr_search(query, top_k)
    
    # Combine and normalize scores
    all_ids = set([id for id, _ in bm25_results + dpr_results])
    combined_scores = {}
    for id in all_ids:
        bm25_score = next((score for doc_id, score in bm25_results if doc_id == id), 0)
        dpr_score = next((score for doc_id, score in dpr_results if doc_id == id), 0)
        combined_scores[id] = alpha * bm25_score + (1 - alpha) * dpr_score

    results=sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return  results

# For now we are using this method, later we can compair between the Milvus re-ranking method and this one. 
def rerank(query, documents, top_k=10):
    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(documents, scores))
    results= sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
    return results


def retrieve_and_rerank(query, top_k=10):
    ### Add re-ranking from milvus------------------------------------------------
    # Hybrid search
    search_results = hybrid_search(query, top_k=top_k*2)
    
    # Fetch full documents (assuming we have a function to do this): This fetch method assumes that we are storing the content embedding and the actual content together with the doc id.


    top_doc_texts = fetch_documents([doc_id for doc_id, _ in search_results])
    
    # Rerank
    reranked_docs = rerank(query, top_doc_texts, top_k)
    
    return reranked_docs
# Main execution
# if __name__ == "__main__":
#     connect_to_milvus()
    
#     collection_name = "example_collection"
#     dim = 128  # dimension of the vector
#     collection = create_collection(collection_name, dim)
    
#     # Generate some example vectors
#     num_entities = 1000
#     vectors = np.random.rand(num_entities, dim).tolist()
#     ids = list(range(num_entities))
    
#     # Insert the vectors
#     insert_vectors(collection, vectors, ids)
    
#     # Load the collection into memory
#     load_collection(collection)
    
#     # Perform a search
#     query = np.random.rand(1, dim).tolist()
#     results = search_vectors(collection, query, top_k=5)
    
#     # Print results
#     for hits in results:
#         for hit in hits:
#             print(f"ID: {hit.id}, Distance: {hit.distance}")
    
#     # Disconnect
#     connections.disconnect("default")
#     print("Disconnected from Milvus")