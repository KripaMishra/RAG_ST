from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from pymilvus import model
from pymilvus import MilvusClient
---------------
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
client_db="milvus_demo.db"


def semantic_chunking(self):
    with open("/home/ubuntu/Steps/formatted_data.txt") as f:
        data = f.read()
    
    text_splitter = SemanticChunker(
        GPT4AllEmbeddings(
            model_name=model_name,
            gpt4all_kwargs=gpt4all_kwargs
        ),
        breakpoint_threshold_type="percentile"
    )

    docs = text_splitter.create_documents([data])
    if len(docs) == 0:
        raise Exception("Document obtained after chunking is empty") 
    
    return docs


class Milvus:
    def __init__(self):
        pass

    def milvus_setup(self):
        # Assuming client_db and collection_name are defined or passed correctly
        client = MilvusClient(client_db)
        
        # Drop collection if exists
        if client.has_collection(collection_name="demo_collection"):
            client.drop_collection(collection_name="demo_collection")
        
        # Create collection with dimensionality
        client.create_collection(
            collection_name="demo_collection",
            dimension=768  # Dimensionality of vectors
        )

        embedding_fn = model.DefaultEmbeddingFunction()
        
        docs = self.semantic_chunking()  # Assuming this method returns docs
        
        vectors = embedding_fn.encode_documents(docs)
        
        # Prepare data for insertion
        data = [
            {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
            for i in range(len(vectors))
        ]
        
        # Insert data into Milvus collection
        res = client.insert(collection_name="demo_collection", data=data)
        print(res)

# query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

# res = client.search(
#     collection_name="demo_collection",  # target collection
#     data=query_vectors,  # query vectors
#     limit=2,  # number of returned entities
#     output_fields=["text", "subject"],  # specifies fields to be returned
# )

# print(res)
