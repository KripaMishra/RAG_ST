# Web Crawling Nvidia Documentation

## Purpose

This repository contains the implementation of a comprehensive web crawler and Retrieval-Augmented Generation (RAG) model designed to scrape and process data from the NVIDIA CUDA documentation. The primary goal of this project is to develop a system that can effectively retrieve and answer questions based on the scraped documentation. This involves several key steps: web crawling, data chunking, vector database creation, retrieval and re-ranking, and question answering.

### Key Features:
1. **Web Crawling**: Scrapes data from the NVIDIA CUDA documentation and its sub-links up to a depth of 5 levels.
2. **Data Chunking**: Uses advanced techniques for chunking data based on semantic similarity and topic relevance.
3. **Vector Database Creation**: Converts chunks into embedding vectors and stores them in the Milvus vector database using FLAT and IVF indexing methods.
4. **Retrieval and Re-ranking**: Implements hybrid retrieval methods and query expansion techniques to retrieve and re-rank data.
5. **Question Answering**: Utilizes a Language Model (LLM) to generate accurate answers based on the retrieved data.
6. **User Interface**: Provides a user-friendly interface for inputting queries and displaying answers using Streamlit.

## Tech Stack

### Programming Language
- **Python**

### Libraries and Frameworks
- **Web Crawling**: 
  - Scrapy
  - BeautifulSoup
- **Data Processing**:
  - NLTK
- **Embedding and Vector Database**:
  - Transformers (for BERT/bi-encoder embeddings)
  - Milvus (for vector database creation)
- **Retrieval Methods**:
  - BM25
  - DPR (Dense Passage Retrieval)
- **Question Answering**:
  - Hugging Face Transformers
- **User Interface**:
  - Streamlit

## Setup Instructions

### Step 0: Install the Packages

Start by clonning the model and proceed as follows:

```bash
pip install -r requirements.txt
```

### Step 1: The Setup

First step is to scrape the data from the URL: https://docs.nvidia.com/cuda/

1. Navigate to the directory containing our spider:
    ```bash
    cd /home/ubuntu/Steps/nvidia_docs/nvidia_docs/spiders/
    ```

2. Run the following command to crawl data and save it as a JSON file:
    ```bash
    scrapy runspider nvidia_docs -o output.json
    ```
    Note: The name of the spider will be specified in the spider file. This will save the output of the web crawling into a JSON file in the directory. Due to resource constraints, we are only scraping data up to depth=1, but you can change it to 5 in the `settings.py` file.

### Step 2: Setting up Milvus and Elasticsearch

Note: We don't need Elasticsearch right now but we'll need it for hybrid indexing. I'll suggest getting done with all installations and setups at once.

#### Milvus Setup Guide

**Installation:**
```bash
# Install docker
sudo apt-get install docker.io

# Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start the Docker container
bash standalone_embed.sh start
```

**Stop/Delete:**
```bash
# Stop Milvus
bash standalone_embed.sh stop

# Delete Milvus data
bash standalone_embed.sh delete
```

#### Elasticsearch Setup

Download and install archive for Linux:
```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.14.3-linux-x86_64.tar.gz

wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.14.3-linux-x86_64.tar.gz.sha512

shasum -a 512 -c elasticsearch-8.14.3-linux-x86_64.tar.gz.sha512

tar -xzf elasticsearch-8.14.3-linux-x86_64.tar.gz

cd elasticsearch-8.14.3/
```
For other devices, refer to the [official documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html#install-linux).

Enable automatic creation of system indices:
```bash
action.auto_create_index: .monitoring*,.watches,.triggered_watches,.watcher-history*,.ml*
```

Run Elasticsearch from the command line:
```bash
./bin/elasticsearch
```

### Step 3: Cleaning the Data

Run the following command to clean the data:
```bash
python data_cleaning.py --file_path /home/ubuntu/project/Steps/nvidia_docs/nvidia_docs/spiders/output.json --output_path /home/ubuntu/project/Steps/result/claened_data.txt

```

### Step 4: Chunking the Data

Run the following command to chunk the data:
```bash
python Steps/components/s2_semantic_chunking.py --file_path /home/ubuntu/project/Steps/result/claened_data.txt --similarity_threshold 0.15 --max_chunk_length 400 --output_json_file Steps/result/preprocessed_chunks.json --micro_json_file Steps/result/micro_chunks.json --micro_threshold 100
```

### Step 5: Data Ingestion

Run the following command to ingest the data:
```bash
python Steps/components/s3_data_ingestion.py --input_path Steps/result/preprocessed_chunks.json --collection_name Test_collection --es_host localhost --es_port 9200 --milvus_host localhost --milvus_port 19530

```

### Step 6: Testing the Retriever Component (optional)

Run the following command to test the model:
```bash
python Steps/components/s4_data_retrieval.py "How do I install the Toolkit in a different location?" --top_k 5 --file_path "/path/to/save/query_results.json"

```

### Step 7: Generating output using LLM and retriever

Run the following command to test the model:
```bash
python Steps/components/s5_RAG_LLM.py "what is cuda used for?" --top_k 5 --file_path "/home/ubuntu/project/Steps/result/query_results/"

```

## Update: Accessing streamlit UI

Run the following command to access the UI:
```bash
streamlit run main.py

```