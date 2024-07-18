import json
from datetime import datetime
from typing import List, Dict
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from s4_data_retrieval import CustomRetrieval, QueryEnhancer
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGModel:
    def __init__(self, model_name: str):
        try:
            self.retriever = CustomRetrieval()
            self.retriever.setup()
            self.query_enhancer = QueryEnhancer()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.max_length = self.model.config.max_position_embeddings
            self.stride = self.max_length // 2
            logging.info("RAGModel initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing RAGModel: {e}")
            raise

    def generate_answer(self, query: str) -> str:
        try:
            # Perform custom retrieval
            retrieval_results = self.retriever.retrieve_and_rerank(query)

            # Enhance the query
            expanded_queries = self.query_enhancer.expand_query_with_keywords(query)

            # Combine original query, expanded queries, and retrieved documents
            context = f"Original query: {query}\n"
            context += "Retrieved documents:\n"
            context += f"- {retrieval_results['results'][0]}\n"

            # Generate answer using the language model
            prompt = (f"{context}\n You are an expert in the matters of CPU setups and system management, "
                      "especially for GPU. You will be asked questions about GPU setup, CUDA errors, and other "
                      "queries related to CUDA setup and CUDA functioning. Based on the above information, and your "
                      "prior knowledge, please answer the following questions: {query}")

            # Tokenize the input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length - 100)
            logging.info(f"Token IDs: {inputs.input_ids}")
            logging.info(f"Token IDs shape: {inputs.input_ids.shape}")

            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decode the generated answer
            answer = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return answer.strip()
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "An error occurred while generating the answer."

    
    def process_query(self, query: str) -> Dict:
        try:
            timestamp = datetime.now().isoformat()
            retrieval_results = self.retriever.retrieve_and_rerank(query)['results']
            expanded_queries = self.query_enhancer.expand_query_with_keywords(query)
            answer = self.generate_answer(query)

            return {
                "timestamp": timestamp,
                "original_query": query,
                "context":retrieval_results,
                "answer": answer,
                
            }
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "original_query": query,
                "error": str(e)
            }
    

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
                file_path = f"/home/ubuntu/project/Steps/result/query_results_{timestamp}.json"
            
            # Save the data to a JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(query_data, f, ensure_ascii=False, indent=4)
            
            logging.info(f"Query results saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving query results: {e}")


def main(query, top_k, file_path):
    model_name = "mistralai/Mistral-7B-v0.3"  # Replace with your preferred model
    rag_model = RAGModel(model_name)
    
    result = rag_model.process_query(query)
    rag_model.save_query_results(result, file_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query Retrieval and Reranking')
    parser.add_argument('query', type=str, help='The query string')
    parser.add_argument('--top_k', type=int, default=1, help='Number of top results to retrieve')
    parser.add_argument('--file_path', type=str, default=None, help='Path to save query results')
    args = parser.parse_args()
    main(args.query, args.top_k, args.file_path)
# sample usage : python /home/ubuntu/project/Steps/components/s5_RAG_LLM.py "what is cuda used for?" --top_k 5 --file_path "/home/ubuntu/project/Steps/result/query_results/"
