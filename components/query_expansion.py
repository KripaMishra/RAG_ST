from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class QueryExpander:
    def __init__(self, model_name="gpt2-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def expand_query(self, query, num_expansions=1, max_length=200):
        prompt = f"Generate {num_expansions} alternative ways to ask this question more precisely and in detailed manner: '{query}'\n\n1."
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_expansions,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        expanded_queries = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
        expanded_queries = [q.split("\n")[0].strip() for q in expanded_queries]
        return [query] + expanded_queries  # Include original query

class RAGWithExpansion:
    def __init__(self, retriever, generator, expander):
        self.retriever = retriever
        self.generator = generator
        self.expander = expander

    def retrieve_and_generate(self, query):
        expanded_queries = self.expander.expand_query(query)
        all_documents = []

        for expanded_query in expanded_queries:
            documents = self.retriever.retrieve(expanded_query)
            all_documents.extend(documents)

        # Remove duplicates and rank
        unique_documents = list(set(all_documents))
        ranked_documents = self.rank_documents(unique_documents, query)

        context = self.prepare_context(ranked_documents)
        response = self.generator.generate(query, context)

        return response

    def rank_documents(self, documents, original_query):
        # Implement ranking logic here (e.g., by relevance to original query)
        # This is a placeholder implementation
        return documents[:5]  # Return top 5 for simplicity

    def prepare_context(self, documents):
        # Combine documents into a single context string
        return " ".join(documents)

# Usage example
expander = QueryExpander()
retriever = YourRetrieverClass()  # Implement or use existing retriever
generator = YourGeneratorClass()  # Implement or use existing generator

rag_system = RAGWithExpansion(retriever, generator, expander)

query = "What are the effects of climate change on polar bears?"
response = rag_system.retrieve_and_generate(query)
print(response)