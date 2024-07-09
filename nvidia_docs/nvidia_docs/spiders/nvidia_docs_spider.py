import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

class NvidiaDocsSpider(CrawlSpider):
    name = 'nvidia_docs'
    allowed_domains = ['docs.nvidia.com']
    start_urls = ['https://docs.nvidia.com/cuda/']

    rules = (
        Rule(LinkExtractor(restrict_xpaths=['//a[@href]', '//body//a[not(ancestor::footer)]']), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        content = ' '.join(response.xpath('//body//*[not(self::footer or self::table)]//text()').getall()).strip()
        item = {
            'url': response.url,
            'content': content  # Extract content from <p> tags only
        }
        yield item

# /html/body/div[2]/div[2]
# /html/body/div[2]   /html/body/div[1]/div/div[2]/div[2]