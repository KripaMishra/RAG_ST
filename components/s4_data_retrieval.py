import json
import logging
import argparse
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            collection = Collection("Test_collection")
            collection.load()
            logger.info("Setup complete: Elasticsearch and Milvus collection initialized.")
        except Exception as e:
            logger.error(f"Setup failed: {e}")

    def encode_query(self, query):
        try:
            input_ids = question_tokenizer(query, return_tensors='pt')['input_ids']
            with torch.no_grad():
                embeddings = question_encoder(input_ids).pooler_output
            return embeddings[0].numpy()
        except Exception as e:
            logger.error(f"Query encoding failed: {e}")
            return None

    def hybrid_search(self, query, top_k=100, alpha=0.5):
        global collection, es
        try:
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
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def fetch_documents(self, doc_ids):
        global collection
        try:
            # Remove None values and convert to integers
            valid_ids = [int(id) for id in doc_ids if id is not None]

            if not valid_ids:
                return []  # Return empty list if no valid ids

            results = collection.query(
                expr=f"id in {valid_ids}",
                output_fields=["id", "content"]
            )
            return [result['content'] for result in results]
        except Exception as e:
            logger.error(f"Fetching documents failed: {e}")
            return []

    def expand_query_with_keywords(self, query, num_expansions=3, num_keywords=5):
        try:
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
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []

    def retrieve_and_rerank(self, query, top_k=1):
        try:
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
        except Exception as e:
            logger.error(f"Retrieve and rerank failed: {e}")
            return {}

    def save_query_results(self, query_data, file_path=None):
        """
        Save the query information and results as a JSON file.
        
        :param query_data: Dictionary containing query information and results
        :param file_path: Optional file path. If None, a default path will be used.
        """
        try:
            # Generate a default file path if none is provided
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"/home/ubuntu/project/Steps/retrieved_result/query_results_{timestamp}.json"

            # Save the data to a JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(query_data, f, ensure_ascii=False, indent=4)

            logger.info(f"Query results saved to {file_path}")
        except Exception as e:
            logger.error(f"Saving query results failed: {e}")

class QueryEnhancer:
    def __init__(self) -> None:
        pass

    def expand_query_with_keywords(self, query, num_expansions=3, num_keywords=5):
        try:
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
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return []

def main(query, top_k, file_path):
    retrieval = CustomRetrieval()
    retrieval.setup()
    results = retrieval.retrieve_and_rerank(query, top_k)
    retrieval.save_query_results(results, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Retrieval and Reranking')
    parser.add_argument('query', type=str, help='The query string')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top results to retrieve')
    parser.add_argument('--file_path', type=str, default=None, help='Path to save query results')
    args = parser.parse_args()
    main(args.query, args.top_k, args.file_path)


# sample usage: python query_retrieval.py "How do I install the Toolkit in a different location?" --top_k 5 --file_path "/path/to/save/query_results.json"
