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
from transformers import T5ForConditionalGeneration, T5Tokenizer

question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
cross_encoder = SentenceTransformer('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Query Expansion specifications
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)



# Global variables
collection = None
es = None

def setup():
    global collection, es
    connections.connect("default", host="localhost", port="19530")
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    
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

def encode_query(query):
    input_ids = question_tokenizer(query, return_tensors='pt')['input_ids']
    with torch.no_grad():
        embeddings = question_encoder(input_ids).pooler_output
    return embeddings[0].numpy()

def hybrid_search(query, top_k=100, alpha=0.5):
    global collection, es
    
    # BM25 search
    es_response = es.search(index="documents", body={
        "query": {"match": {"content": query}},
        "size": top_k
    })
    bm25_results = [(hit['_id'], hit['_score']) for hit in es_response['hits']['hits']]
    
    # DPR search with Milvus reranking
    query_vector = encode_query(query)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
        "offset": 0,
        "limit": top_k,
        "with_distance": True,
        "expr": None,
        "output_fields": ["id", "content"],
        "round_decimal": -1,
        "rerank": {
            "metric_type": "IP",  # Inner Product
            "params": {"rerank_topk": min(top_k * 2, 100)}  # Rerank top 100 or top_k * 2, whichever is smaller
        }
    }
    milvus_results = collection.search(
        data=[query_vector.tolist()],
        anns_field="embedding",
        param=search_params,
    )
    dpr_results = [(hit.entity.get('id'), hit.score) for hit in milvus_results[0]]
    
    # Combine and normalize scores
    all_ids = set([id for id, _ in bm25_results + dpr_results])
    combined_scores = {}
    for id in all_ids:
        bm25_score = next((score for doc_id, score in bm25_results if doc_id == id), 0)
        dpr_score = next((score for doc_id, score in dpr_results if doc_id == id), 0)
        combined_scores[id] = alpha * bm25_score + (1 - alpha) * dpr_score

    results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return results

def fetch_documents(doc_ids):
    global collection
    
    # Fetch documents from Milvus
    results = collection.query(
        expr=f"id in {doc_ids}",
        output_fields=["id", "content"]
    )
    return [result['content'] for result in results]


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

# Modify the retrieve_and_rerank function to use query expansion
def retrieve_and_rerank(query, top_k=10):
    # Query expansion
    expanded_queries = expand_query_with_keywords(query)
    
    all_results = []
    for expanded_query in expanded_queries:
        # Hybrid search with Milvus reranking
        search_results = hybrid_search(expanded_query, top_k=top_k*2)
        all_results.extend(search_results)
    
    # Remove duplicates and get top results
    unique_results = list(dict.fromkeys(all_results))[:top_k*2]
    
    # Fetch full documents
    doc_ids = [doc_id for doc_id, _ in unique_results]
    top_doc_texts = fetch_documents(doc_ids)
    
    # At this point, we've already used Milvus' reranking, so we can either:
    # 1. Return the results as is, or
    # 2. Apply an additional reranking step if needed
    
    # Option 1: Return results as is
    return [(doc, score) for (doc_id, score), doc in zip(unique_results, top_doc_texts)]


def load_data(input_path):
    with open(input_path,'r') as content:
        documents=json.load(content)
    return documents
# Main execution
if __name__ == "__main__":
    setup()
    
    # Example usage
    input_path="Steps/documents_export.json"
    documents = load_data()
    
    index_documents(documents)
    
    query = "Example query"
    results = retrieve_and_rerank(query)
    
    for doc, score in results:
        print(f"Score: {score}, Document: {doc}")