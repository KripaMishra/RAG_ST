Sure, here's the entire content formatted as a `README.md` file:


# Web Crawling Nvidia Documentation

This document outlines the steps for running this repository.

#### Description

This is a RAG (Retrieval-Augmented Generation) model. We will scrape our data using Scrapy and web crawling techniques. The data will be stored in the Milvus vector database, which will later be used as a knowledge source for our Q&A model.

## Steps

### Step 0: Install the Packages

Install the required packages by running:

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
python3 Steps/components/s1_data_cleaning.py
```

### Step 4: Chunking the Data

Run the following command to chunk the data:
```bash
python3 Steps/components/s2_semantic_chunking.py
```

### Step 5: Data Ingestion

Run the following command to ingest the data:
```bash
python3 Steps/components/s3_data_ingestion.py
```

### Step 6: Testing the Model

Run the following command to test the model:
```bash
python3 Steps/components/s5_RAG_LLM.py
```
