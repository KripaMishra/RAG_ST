# query_retrieval.py

import torch
from elasticsearch import Elasticsearch
from pymilvus import (
    connections,
    Collection,
)
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)

# Initialize models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

model_name = 't5-base'
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

# Global variables
collection = None
es = None

def setup():
    global collection, es
    connections.connect("default", host="localhost", port="19530")
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    collection = Collection("documents")

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
            "params": {"rerank_topk": min(top_k * 2, 100)}
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
    
    results = collection.query(
        expr=f"id in {doc_ids}",
        output_fields=["id", "content"]
    )
    return [result['content'] for result in results]

def expand_query_with_keywords(query, num_expansions=3, num_keywords=5):
    input_text = f"expand query: {query}"
    input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids

    outputs = t5_model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=num_expansions,
        num_beams=num_expansions,
        temperature=0.7
    )

    expanded_queries = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    keyword_input = f"generate keywords for: {query}"
    keyword_ids = t5_tokenizer(keyword_input, return_tensors="pt").input_ids

    keyword_outputs = t5_model.generate(
        keyword_ids,
        max_length=30,
        num_return_sequences=1,
        num_beams=num_keywords,
        temperature=0.7
    )

    keywords = t5_tokenizer.decode(keyword_outputs[0], skip_special_tokens=True).split()

    final_queries = [query] + expanded_queries
    final_queries = [f"{q} {' '.join(keywords)}" for q in final_queries]

    return final_queries

def retrieve_and_rerank(query, top_k=10):
    expanded_queries = expand_query_with_keywords(query)
    
    all_results = []
    for expanded_query in expanded_queries:
        search_results = hybrid_search(expanded_query, top_k=top_k*2)
        all_results.extend(search_results)
    
    unique_results = list(dict.fromkeys(all_results))[:top_k*2]
    
    doc_ids = [doc_id for doc_id, _ in unique_results]
    top_doc_texts = fetch_documents(doc_ids)
    
    return [(doc, score) for (doc_id, score), doc in zip(unique_results, top_doc_texts)]

if __name__ == "__main__":
    setup()
    
    query = "Example query"
    results = retrieve_and_rerank(query)
    
    for doc, score in results:
        print(f"Score: {score}, Document: {doc}")