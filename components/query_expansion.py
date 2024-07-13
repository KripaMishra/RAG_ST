from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


from gpt4all import GPT4All
import re

class QueryExpander:
    def __init__(self, model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf"):
        self.model = GPT4All(model_name, device='cpu')

    def expand_query(self, query, num_expansions=2):
        prompt = f"""Generate {num_expansions} alternative ways to ask this question: '{query}'
        Make sure each alternative is a complete question ending with a question mark.
        Format your response as a numbered list:

        1. 
        2. 
        """

        response = self.model.generate(prompt, max_tokens=100, temp=0.5, top_k=20, top_p=0.8)

        # Extract questions from the response
        expanded_queries = re.findall(r'\d+\.\s*(.*?\?)', response, re.DOTALL)

        # Clean up the extracted questions
        expanded_queries = [q.strip() for q in expanded_queries if q.strip()]

        return [query] + expanded_queries

# Usage
expander = QueryExpander()
query = "What are the benefits of using the query expander?"
expanded_queries = expander.expand_query(query)
print("Original query:", query)
print("Expanded queries:")
for i, q in enumerate(expanded_queries[1:], 1):
    print(f"{i}. {q}")


#------------------------------------------------------------------------
# class RAGWithExpansion:
#     def __init__(self, retriever, generator, expander):
#         self.retriever = retriever
#         self.generator = generator
#         self.expander = expander

#     def retrieve_and_generate(self, query):
#         expanded_queries = self.expander.expand_query(query)
#         all_documents = []

#         for expanded_query in expanded_queries:
#             documents = self.retriever.retrieve(expanded_query)
#             all_documents.extend(documents)

#         # Remove duplicates and rank
#         unique_documents = list(set(all_documents))
#         ranked_documents = self.rank_documents(unique_documents, query)

#         context = self.prepare_context(ranked_documents)
#         response = self.generator.generate(query, context)

#         return response

#     def rank_documents(self, documents, original_query):
#         # Implement ranking logic here (e.g., by relevance to original query)
#         # This is a placeholder implementation
#         return documents[:5]  # Return top 5 for simplicity

#     def prepare_context(self, documents):
#         # Combine documents into a single context string
#         return " ".join(documents)

# Usage example
# expander = QueryExpander()
# retriever = YourRetrieverClass()  # Implement or use existing retriever
# generator = YourGeneratorClass()  # Implement or use existing generator

# rag_system = RAGWithExpansion(retriever, generator, expander)

# query = "What are the effects of climate change on polar bears?"
# response = rag_system.retrieve_and_generate(query)
# print(response)