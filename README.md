## Web Crawling Nvidia Documentation

This document outlines the steps for crawling Nvidia documentation using Scrapy.

**Running the Crawler**

1. Navigate to the project directory:

```
cd /home/ubuntu/Steps/nvidia_docs/nvidia_docs/
```

2. Run the following command to crawl data and save it as a JSON file:

```
scrapy runspider nvidia_docs_spider.py -o output.json
```

**Data Formatting**

- Refer to the `formatting.ipynb` notebook for instructions on formatting the scraped data.
- The `chunking.ipynb` notebook explores techniques for chunking the data, potentially for future pipeline implementation.

**Note:** The `chunking.ipynb` notebook currently explores chunking but might require further development for integration into a production pipeline.
