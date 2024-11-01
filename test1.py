import os
import re
import logging
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from konlpy.tag import Komoran  # For Korean NLP
from soynlp.word import WordExtractor
from pytrends.request import TrendReq
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StopwordManager:
    _instance = None

    def __new__(cls, file_path='stopwords-ko.txt'):
        if cls._instance is None:
            cls._instance = super(StopwordManager, cls).__new__(cls)
            cls._instance.stopwords = cls.load_stopwords(file_path)
        return cls._instance

    @staticmethod
    def load_stopwords(file_path):
        default_stopwords = set([
            '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과',
            '도', '를', '으로', '자', '에', '와', '한', '하다', '것', '말',
            '그', '저', '및', '더', '등', '데', '때', '년', '월', '일',
            '그리고', '하지만', '그러나', '그래서', '또는', '즉', '만약', '아니면',
            '때문에', '그런데', '그러므로', '따라서', '뿐만 아니라',
            '이런', '저런', '합니다', '있습니다', '했습니다', '있다', '없다',
            '됩니다', '되다', '이다'
        ])
        if not os.path.exists(file_path):
            logging.warning(f"Stopword file not found: {file_path}. Using default stopwords.")
            return default_stopwords
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f if line.strip())
            stopwords.update(default_stopwords)
            return stopwords
        except Exception as e:
            logging.error(f"Error loading stopword file: {e}")
            return default_stopwords

class Scraper:
    def __init__(self):
        self.session = requests.Session()

    def scrape_webpage(self, url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = self.session.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except Exception as e:
            logging.error(f"Error scraping webpage ({url}): {e}")
            return None

    def scrape_webpage_for_google_search(self, url):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            html = response.text

            soup = BeautifulSoup(html, 'html.parser')

            # Tags and classes to remove
            unwanted_ids = [
                'newsSidebar', 'newsMainBanner', 'rightSlideDiv_1', 
                'rightSlideDiv_2', 'rightSlideDiv_3',
            ]
            unwanted_classes = [
                'sidebar', 'rankingNews', 'photo_slide', 'ad290x330', 
                'socialAD', 'AdIbl', 'rankingEmotion', 'ofhe_head', 
                'ofhe_body', 'outside_area_inner', 'outside_area', 
                '_OUTSIDE_AREA', '_GRID_TEMPLATE_COLUMN_ASIDE', 
                '_OUTSIDE_AREA_INNER',
            ]

            for unwanted_id in unwanted_ids:
                for tag in soup.find_all(id=unwanted_id):
                    tag.decompose()

            for unwanted_class in unwanted_classes:
                for tag in soup.find_all(class_=unwanted_class):
                    tag.decompose()

            candidate_blocks = soup.find_all(['div', 'article', 'section'])
            blocks_with_density = []
            for block in candidate_blocks:
                density = self.calculate_text_density(block)
                blocks_with_density.append((density, block))

            blocks_with_density.sort(key=lambda x: x[0], reverse=True)

            article_text = ""
            for density, block in blocks_with_density:
                if density > 0.1:
                    for unwanted in block(['script', 'style', 'figure', 'iframe', 'br', 'noscript']):
                        unwanted.decompose()
                    text = block.get_text(separator=' ', strip=True)
                    if len(text) > len(article_text):
                        article_text = text
                else:
                    break

            if len(article_text) < 200:
                paragraphs = soup.find_all('p')
                article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])

            return article_text

        except Exception as e:
            logging.error(f"Error scraping webpage ({url}): {e}")
            return ""

    def calculate_text_density(self, html_element):
        text_length = len(html_element.get_text(strip=True))
        tag_length = len(str(html_element))
        return text_length / max(tag_length, 1)

class TextProcessor:
    def __init__(self):
        self.komoran = Komoran()

    def preprocess_text(self, text):
        if not text or not isinstance(text, str) or not text.strip():
            logging.warning("Invalid input text.")
            return ""
        text = re.sub(r'[^0-9가-힣a-zA-Z\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def extract_keywords(self, text, stopwords, top_n=10):
        tokens = self.komoran.nouns(text)
        filtered_tokens = [word for word in tokens if 2 <= len(word) <= 3 and word not in stopwords]
        token_counts = Counter(filtered_tokens)
        top_keywords = [word for word, count in token_counts.most_common(top_n)]
        return top_keywords

class GoogleTrends:
    def __init__(self):
        self.pytrends = TrendReq(hl='ko', tz=540)

    def get_trending_keywords(self):
        try:
            df_trending = self.pytrends.trending_searches(pn='south_korea')
            return df_trending[0].tolist()
        except Exception as e:
            logging.error(f"Error getting Google trends: {e}")
            return []

class NewsCollector:
    def __init__(self):
        self.scraper = Scraper()
        self.stopwords = StopwordManager().stopwords

    def search_naver_news(self, keyword):
        search_url = f"https://search.naver.com/search.naver?&where=news&query={requests.utils.quote(keyword)}"
        soup = self.scraper.scrape_webpage(search_url)
        if not soup:
            return []
        news_elements = soup.select('.list_news .news_area')
        return [{'title': elem.select_one('.news_tit').get_text(strip=True), 'link': elem.select_one('.news_tit')['href']} for elem in news_elements if elem.select_one('.news_tit')]

    def collect_news_texts(self, news_items):
        articles_texts = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_news = {executor.submit(self.scraper.scrape_webpage_for_google_search, news['link']): news for news in news_items}
            for future in tqdm(as_completed(future_to_news), total=len(future_to_news), desc="Scraping news articles"):
                news = future_to_news[future]
                try:
                    article_text = future.result()
                    if article_text:
                        full_text = self.scraper.preprocess_text(news['title'] + ' ' + article_text)
                        if full_text:
                            articles_texts.append(full_text)
                except Exception as e:
                    logging.error(f"Error processing news article ({news['title']}): {e}")
        return articles_texts

class MainProcessor:
    def __init__(self):
        self.stopwords = StopwordManager().stopwords
        self.text_processor = TextProcessor()
        self.google_trends = GoogleTrends()
        self.news_collector = NewsCollector()

    def run(self):
        trending_keywords = self.google_trends.get_trending_keywords()
        for keyword in trending_keywords:
            if keyword in self.stopwords:
                continue
            news_items = self.news_collector.search_naver_news(keyword)
            articles_texts = self.news_collector.collect_news_texts(news_items)
            combined_keywords = []
            for text in articles_texts:
                keywords = self.text_processor.extract_keywords(text, self.stopwords)
                combined_keywords.extend(keywords)
            # Further processing can be done here...

if __name__ == "__main__":
    processor = MainProcessor()
    processor.run()
