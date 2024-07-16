"""import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import deque

class WebCrawler:
    def __init__(self, base_url, max_depth=5):
        self.base_url = base_url
        self.max_depth = max_depth
        self.visited = set()
        self.data = {}

    def fetch_page(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_links(self, html, base_url):
        soup = BeautifulSoup(html, 'html.parser')
        links = set()
        for tag in soup.find_all('a', href=True):
            link = urljoin(base_url, tag['href'])
            if link.startswith(self.base_url):
                links.add(link)
        return links

    def scrape_data(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        # Extracting text from the HTML
        text = soup.get_text(separator=' ', strip=True)
        return text

    def crawl(self):
        queue = deque([(self.base_url, 0)])
        
        while queue:
            url, depth = queue.popleft()
            if url in self.visited or depth > self.max_depth:
                continue
            
            print(f"Visiting: {url} at depth {depth}")
            html = self.fetch_page(url)
            if not html:
                continue
            
            self.visited.add(url)
            self.data[url] = self.scrape_data(html)
            
            if depth < self.max_depth:
                links = self.parse_links(html, url)
                for link in links:
                    if link not in self.visited:
                        queue.append((link, depth + 1))

    def save_data(self, filename='scraped_data.txt'):
        with open(filename, 'w') as file:
            for url, data in self.data.items():
                file.write(f"URL: {url}\n\n{data}\n\n{'-'*80}\n\n")

if __name__ == "__main__":
    #crawler = WebCrawler(base_url="https://docs.nvidia.com/cuda/")
    #crawler.crawl()
    #crawler.save_data()
    
    print("Crawling finished and data saved.")
"""


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
