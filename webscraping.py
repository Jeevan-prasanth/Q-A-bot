

import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse

class WebCrawler:
    def __init__(self, base_url, max_depth=5):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited = set()
        self.results = []

    def crawl(self, url, depth):
        if depth > self.max_depth or url in self.visited:
            return

        print(f'Crawling URL: {url} at depth: {depth}')
        self.visited.add(url)
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to retrieve {url}")
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        article_body = soup.find('div', itemprop='articleBody')
        if article_body:
            text = ' '.join(article_body.stripped_strings)
            self.results.append({
                'url': url,
                'text': text
            })

        if depth < self.max_depth:
            links = self.extract_links(soup, url)
            for link in links:
                self.crawl(link, depth + 1)

    def extract_links(self, soup, current_url):
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('#'):
                continue  # Skip in-page anchors
            link = urljoin(current_url, href)
            if self.is_valid_link(link):
                links.append(link)
        return links

    def is_valid_link(self, link):
        # Ensure the link is within the same domain
        base_netloc = urlparse(self.base_url).netloc
        link_netloc = urlparse(link).netloc
        return base_netloc == link_netloc

    def save_results(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=4)

if __name__ == "__main__":
    start_url = 'https://docs.nvidia.com/cuda/'
    crawler = WebCrawler(start_url, max_depth=5)
    crawler.crawl(start_url, depth=0)
    crawler.save_results('output.json')

    print(f"Scraping completed. Results saved to output.json")
