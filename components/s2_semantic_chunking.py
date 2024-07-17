import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

class CustomEmbeddings:
    def __init__(self, model_name, device):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

def semantic_chunk_to_documents(file_path, chunk_size=8000, chunk_overlap=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_name = 'all-MiniLM-L6-v2'

    # Create custom embedding object
    embeddings = CustomEmbeddings(model_name, device)

    # Initialize the semantic chunker
    text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_amount= 0.9)

    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into chunks
    print("Semantically chunking the text...")
    chunks = text_splitter.split_text(text)

    # Create the documents list with IDs
    print("Creating document objects...")
    documents = []

    # Process chunks
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing chunks"):
        documents.append({"id": i + 1, "content": chunk})

    return documents

import json

def export_to_json(documents, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # Usage example
    input_file_path = "/home/ubuntu/project/Steps/clean_text_3.txt"
    output_file_path="/home/ubuntu/project/Steps/chunk_data.json"
    documents = semantic_chunk_to_documents(input_file_path)
    export_to_json(documents,output_file_path)

