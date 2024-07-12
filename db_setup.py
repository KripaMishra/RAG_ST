from langchain_experimental.text_splitter import SemanticChunker
from pymilvus import model
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
from langchain_ai21 import AI21Embeddings
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration
api_key = os.getenv('AI21_API_KEY')
client_db = "milvus_demo_test.db"
collection_name = "tes_Collection"

class MilvusHandler:
    def __init__(self, client_db, collection_name):
        self.client_db = client_db
        self.collection_name = collection_name
        self.client = MilvusClient(client_db)
    
    def semantic_chunking(self, content_path):
        with open(content_path) as f:
            data = f.read()
        data = data[:1000]  # Truncate data for testing
        
        text_splitter = SemanticChunker(
            AI21Embeddings(api_key=api_key),
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

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)  # Ensure dimension matches

        self.client.create_collection(
            collection_name=self.collection_name, 
            schema=schema
        )
        
        print('Collection created successfully!')

    def create_custom_index(self, metric_type, index_type, index_name):
        index_params = self.client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            metric_type=metric_type,
            index_type=index_type,
            index_name=index_name,
            params={"nlist": 128}
        )
        
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )
        
        print('Index created successfully!')

    def insert_data(self, content_path):
        embedding_fn = AI21Embeddings(api_key=api_key)

        docs = self.semantic_chunking(content_path)  # Get chunked documents
        
        # Ensure the documents are in the correct format
        texts = [doc.page_content for doc in docs]
        
        vectors = embedding_fn.embed_documents(texts)
        
        # Prepare data for insertion
        data = []
        
        for i, doc in enumerate(tqdm(docs, desc="Creating embeddings")):
            data.append({"id": i, "vector": vectors[i], "text": doc.page_content, "subject": "history"})

        # Insert data into Milvus collection
        res = self.client.insert(collection_name=self.collection_name, data=data)
               
        return res  # Return the response instead of printing it
    
    def check_indexing(self):
        indexing_details = self.client.list_indexes(
            collection_name=self.collection_name
        )
        return indexing_details
    
    def describe_index(self):
        index_description = self.client.describe_index(
            collection_name=self.collection_name
        )
        return index_description

if __name__ == "__main__":
    content_path = "/home/ubuntu/Steps/formatted_data.txt"
    
    milvus_instance = MilvusHandler(client_db, collection_name)

    milvus_instance.create_collection()
    insert_result = milvus_instance.insert_data(content_path)
    print("Insert data result:", insert_result)
