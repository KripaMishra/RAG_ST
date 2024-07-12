from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import GPT4AllEmbeddings
from pymilvus import model
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
gpt4all_kwargs = {'allow_download': True}
client_db = "milvus_demo_test.db"
collection_name = "tes_Collection"

class Milvus:
    def __init__(self, client_db, collection_name):
        self.client_db = client_db
        self.collection_name = collection_name
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

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)  # Ensure dimension matches

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
    
    milvus_instance = Milvus(client_db, collection_name)

    milvus_instance.create_collection()
    insert_result = milvus_instance.insert_data(content_path)
    print("Insert data result:", insert_result)

    index_result = milvus_instance.check_indexing()
    print("Indexing details:", index_result)

    describe_result = milvus_instance.describe_index()
    print("Index description:", describe_result)