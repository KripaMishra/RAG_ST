# query_retrieval.py
import json
from datetime import datetime
import torch
from elasticsearch import Elasticsearch
from pymilvus import connections, Collection
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

class CustomRetrieval:
    def __init__(self):
        pass

    def setup(self):
        global collection, es
        try:
            connections.connect("default", host="localhost", port="19530")
            es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': "http"}])
            collection = Collection("documents")
            collection.load()
            print("Setup complete: Elasticsearch and Milvus collection initialized.")
        except Exception as e:
            print(f"Setup failed: {e}")
            
    def encode_query(self, query):
        input_ids = question_tokenizer(query, return_tensors='pt')['input_ids']
        with torch.no_grad():
            embeddings = question_encoder(input_ids).pooler_output
        return embeddings[0].numpy()

    def hybrid_search(self, query, top_k=100, alpha=0.5):
        global collection, es
        
        # BM25 search
        es_response = es.search(index="documents", body={
            "query": {"match": {"content": query}},
            "size": top_k
        })
        bm25_results = [(hit['_id'], hit['_score']) for hit in es_response['hits']['hits']]
        
        # DPR search with Milvus reranking
        query_vector = self.encode_query(query)
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
            limit=top_k
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

    def fetch_documents(self, doc_ids):
        global collection
        
        # Remove None values and convert to integers
        valid_ids = [int(id) for id in doc_ids if id is not None]
        
        if not valid_ids:
            return []  # Return empty list if no valid ids
        
        results = collection.query(
            expr=f"id in {valid_ids}",
            output_fields=["id", "content"]
        )
        return [result['content'] for result in results]

    def expand_query_with_keywords(self, query, num_expansions=3, num_keywords=5):
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

    def retrieve_and_rerank(self, query, top_k=3):
        expanded_queries = self.expand_query_with_keywords(query)
        
        all_results = []
        for expanded_query in expanded_queries:
            search_results = self.hybrid_search(expanded_query, top_k=top_k*2)
            all_results.extend(search_results)
        
        unique_results = list(dict.fromkeys(all_results))[:top_k*2]
        
        doc_ids = [doc_id for doc_id, _ in unique_results]
        top_doc_texts = self.fetch_documents(doc_ids)
        
        results = [(doc, score) for (doc_id, score), doc in zip(unique_results, top_doc_texts)]
        
        # Create the JSON format data
        query_data = {
            "timestamp": datetime.now().isoformat(),
            "original_query": query,
            "expanded_queries": expanded_queries,
            "results": [
                {
                    "document": doc,
                    "score": float(score)  # Convert score to float for JSON serialization
                }
                for doc, score in results
            ]
        }
        
        return query_data
    
    def save_query_results(self, query_data, file_path=None):
        """
        Save the query information and results as a JSON file.
        
        :param query_data: Dictionary containing query information and results
        :param file_path: Optional file path. If None, a default path will be used.
        """
        # Generate a default file path if none is provided
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"/home/ubuntu/project/Steps/retrieved_result/query_results_{timestamp}.json"
        
        # Save the data to a JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(query_data, f, ensure_ascii=False, indent=4)
        
        print(f"Query results saved to {file_path}")

    # Example usage:
    # save_query_results(original_query, expanded_queries, results)


# if __name__ == "__main__":
#     setup()
    
#     query = " How do I install the Toolkit in a different location? "
#     results = retrieve_and_rerank(query)
#     expanded_query=expand_query_with_keywords(query)
#     save_query_results(query,expanded_query, results)

#     for doc, score in results:
#         print(f"Score: {score}, Document: {doc}")


class QueryEnhancer:
    def __init__(self) -> None:
        pass
    def expand_query_with_keywords(self, query, num_expansions=3, num_keywords=5):
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