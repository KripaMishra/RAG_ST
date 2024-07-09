# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import re
from html import unescape
from itemadapter import ItemAdapter
from nltk.tokenize import sent_tokenize
class NvidiaDocsPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        
        # Clean the 'content' field
        if 'content' in adapter:
            # Remove unwanted characters
            content = adapter['content']
            content = re.sub(r'[\r\n\t]', ' ', content)
            content = unescape(content)
            content = re.sub(r'<[^>]+>', '', content)
            content = ' '.join(content.split())

            # Split content into sentences
            sentences = sent_tokenize(content)

            # Remove duplicate sentences
            unique_sentences = list(dict.fromkeys(sentences))

            # Join sentences with newline characters
            cleaned_content = ' '.join(unique_sentences)

            # Update the item with cleaned content
            adapter['content'] = cleaned_content
        
        return item
