from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np

# Connect to Milvus
def connect_to_milvus(host="localhost", port="19530"):
    try:
        connections.connect("default", host=host, port=port)
        print(f"Successfully connected to Milvus")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")

# Create a collection
def create_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, "Example collection")
    collection = Collection(collection_name, schema)
    
    # Create an IVF_FLAT index for the collection
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index("embeddings", index)
    return collection

# Insert vectors
def insert_vectors(collection, vectors, ids):
    entities = [
        ids,
        vectors
    ]
    insert_result = collection.insert(entities)
    collection.flush()
    print(f"Inserted {insert_result.insert_count} entities")

# Load the collection
def load_collection(collection):
    collection.load()
    print(f"Collection '{collection.name}' loaded into memory")

# Search vectors
def search_vectors(collection, search_vectors, top_k):
    search_param = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    results = collection.search(
        search_vectors, "embeddings", search_param, limit=top_k,
        output_fields=["id"]
    )
    return results

# Main execution
if __name__ == "__main__":
    connect_to_milvus()
    
    collection_name = "example_collection"
    dim = 128  # dimension of the vector
    collection = create_collection(collection_name, dim)
    
    # Generate some example vectors
    num_entities = 1000
    vectors = np.random.rand(num_entities, dim).tolist()
    ids = list(range(num_entities))
    
    # Insert the vectors
    insert_vectors(collection, vectors, ids)
    
    # Load the collection into memory
    load_collection(collection)
    
    # Perform a search
    query = np.random.rand(1, dim).tolist()
    results = search_vectors(collection, query, top_k=5)
    
    # Print results
    for hits in results:
        for hit in hits:
            print(f"ID: {hit.id}, Distance: {hit.distance}")
    
    # Disconnect
    connections.disconnect("default")
    print("Disconnected from Milvus")