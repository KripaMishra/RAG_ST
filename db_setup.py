from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from pymilvus import model
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

#---------------
model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': 'True'}
client_db="milvus_demo.db"
#---------------


model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': True}  # Assuming it's a boolean flag

class Milvus:
    def __init__(self, client_db, collection_name):
        self.client_db = client_db
        self.create_collection=collection_name
        self.client = MilvusClient(client_db)
    
    def semantic_chunking(self, content_path):
        with open(content_path) as f:
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

    def create_collection(self):
        # Drop collection if it exists
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True)

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=5)
        self.client.create_collection(
        collection_name=self.collection_name, 
        schema=schema)


        index_params = MilvusClient.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="FLAT",
            index_name="vector_index",
            params={ "nlist": 128 }
        )
        self.client.create_index(
            collection_name=self.collection_name, 
,
            index_params=index_params
        )
        # Create collection with dimensionality
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=768  # Dimensionality of vectors
        )
        
    def insert_data(self, content_path):
        embedding_fn = model.DefaultEmbeddingFunction()

        docs = self.semantic_chunking(content_path)  # Get chunked documents
        
        vectors = embedding_fn.encode_documents(docs)
        
        # Prepare data for insertion
        data = [
            {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
            for i in range(len(vectors))
        ]
        
        # Insert data into Milvus collection
        res = self.client.insert(collection_name=self.collection_name, data=data)
               
        return res  # Return the response instead of printing it
    
    def check_indexing(self):
        indexing_details=self.client.list_indexes(
            collection_name=self.collection_name
        )
        return indexing_details
    
    def describe_index(self):
        index_description=self.client.list_indexes(
            collection_name=self.collection_name
        )
        return index_description



















# query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

# res = client.search(
#     collection_name="demo_collection",  # target collection
#     data=query_vectors,  # query vectors
#     limit=2,  # number of returned entities
#     output_fields=["text", "subject"],  # specifies fields to be returned
# )

# print(res)
