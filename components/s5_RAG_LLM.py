import json
from datetime import datetime
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from s4_data_retrieval import * 



class RAGModel:
    def __init__(self, model_name: str):
        self.retriever = CustomRetrieval()
        self.retriever.setup()
        self.query_enhancer = QueryEnhancer()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_answer(self, query: str) -> str:
        # Perform custom retrieval
        retrieval_results = self.retriever.retrieve_and_rerank(query)

        # Enhance the query
        expanded_queries = self.query_enhancer.expand_query_with_keywords(query)

        # Combine original query, expanded queries, and retrieved documents
        context = f"Original query: {query}\n"
        context += "Retrieved documents:\n"
        for result in retrieval_results['results'][:2]:
            context += f"- {result['document']}\n"

        # Generate answer using the language model
        prompt = f"{context}\n You are an exper in the matters of cpu setups and system management, specially for GPU. You will be asked questions about gpu setup, cuda errors and other queries realted to cuda setup and cuda funtioning.Based on the above information, and your prior knowledge, please answer the following questions: {query}"
        input_ids = self.tokenizer.encode(prompt ,return_tensors="pt", truncation=True, max_length=512)
        output = self.model.generate(input_ids, max_length=1024, num_return_sequences=1, temperature=0.7,do_sample=True)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return answer

    def process_query(self, query: str) -> Dict:
        timestamp = datetime.now().isoformat()
        retrieval_results = self.retriever.retrieve_and_rerank(query)
        expanded_queries = self.query_enhancer.expand_query_with_keywords(query)
        answer = self.generate_answer(query)

        return {
            "timestamp": timestamp,
            "original_query": query,
            # "expanded_queries": expanded_queries,
            # "results": retrieval_results['results'],
            "answer": answer
        }
    

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

# Usage example
if __name__ == "__main__":
    model_name = "gpt2"  # Replace with your preferred model
    rag_model = RAGModel(model_name)
    
    query = "How do I install the Toolkit in a different location?"
    result = rag_model.process_query(query)
    rag_model.save_query_results(result)
    print(json.dumps(result, indent=2))