# Import necessary libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from collections import Counter
import time
import requests
from selenium.common.exceptions import NoSuchElementException

from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank
from krwordrank.sentence import MaxScoreTokenizer

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count

from konlpy.tag import Okt  # Morphological analyzer

# Function to calculate text density
def calculate_text_density(html_element):
    text_length = len(html_element.get_text(strip=True))
    tag_length = len(str(html_element))
    return text_length / max(tag_length, 1)

# Function to scrape webpage content
def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Select the block with the highest text density
        candidate_blocks = soup.find_all(['div', 'article', 'section'])
        blocks_with_density = [(calculate_text_density(block), block) for block in candidate_blocks]
        blocks_with_density.sort(key=lambda x: x[0], reverse=True)

        article_text = ""
        for density, block in blocks_with_density:
            if density > 0.1:
                article_text = block.get_text(separator=' ', strip=True)
                if len(article_text) > 200:
                    break
        return article_text
    except Exception as e:
        print(f"Error scraping webpage: {e}")
        return ""

# Expanded stopword list
stopwords = {
    '이', '그', '저', '것', '수', '등', '년', '있다', '하다', '들', '되다', '보다', '않다', '없다',
    '우리', '같다', '때문', '때문에', '위해', '대한', '그리고', '그러나', '이지만', '더욱', '가장',
    '이제', '저는', '그래서', '즉', '게다가', '하지만', '또한', '그리고', '한편', '그러나', '때문에',
    '이라며', '로서', '하면', '이다', '에게', '에서는', '로써', '이다', '만큼', '에서', '으로', '까지',
    '부터', '에도', '하고', '하며', '하거나', '이든', '가장', '모든', '이다', '이라', '으로', '그런데',
    # Particles
    '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '로', '으로', '에서', '부터', '까지', '에게',
    # Verb endings
    '다', '나', '라', '며', '면', '니', '네', '데', '고', '지만', '든지', '든가', '하게', '이고', '도록', '야', '서', '여', '요',
    # Conjunctions
    '그리고', '그러나', '하지만', '그래서', '그러면', '그러나', '즉', '게다가', '또한', '지난',
    # Other stopwords
    '합니다', '했다', '있습니다', '됩니다', '것이다', '합니다', '하는', '했다', '있다', '한다', '하지', '하면', '했으며',
}

# Initialize morphological analyzer
okt = Okt()

# Function to extract keywords using morphological analysis
def extract_keywords(texts, min_count=5, max_length=10, beta=0.85, max_iter=10, rank_threshold=0.1):
    # Extract nouns using morphological analysis
    tokenized_texts = []
    for text in texts:
        tokens = okt.nouns(text)
        tokenized_texts.append(' '.join(tokens))

    # Create KRWordRank object
    wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length, verbose=True)
    keywords, rank, graph = wordrank_extractor.extract(tokenized_texts, beta, max_iter)

    # Apply stopword filtering and rank threshold
    filtered_keywords = {word: score for word, score in keywords.items()
                         if word not in stopwords and score >= rank_threshold and len(word) > 1}

    # Select top N keywords
    top_keywords = Counter(filtered_keywords).most_common(10)

    return top_keywords

# Function to generate word cloud
def generate_wordcloud(frequencies):
    # Set font path (modify according to your environment)
    font_path = 'C:/Windows/Fonts/D2Coding.ttf'  # Example for Windows
    # font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # Example for Linux

    wordcloud = WordCloud(
        font_path=font_path,
        background_color='white',
        width=800,
        height=600
    )
    wordcloud = wordcloud.generate_from_frequencies(frequencies)

    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Function to fetch article text (for multiprocessing)
def fetch_article_text(news):
    article_text = scrape_webpage(news['link'])
    if article_text:
        # Return combined title and content
        return news['title'] + ' ' + article_text
    else:
        return ''

# Main function
def main():
    site_url = "https://news.naver.com/main/ranking/popularDay.naver"

    # Set browser options
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(site_url)
    time.sleep(3)

    # Click "Load more rankings" button repeatedly
    while True:
        try:
            more_button = driver.find_element(By.CSS_SELECTOR, '.button_rankingnews_more')
            if more_button.is_displayed():
                driver.execute_script("arguments[0].click();", more_button)
                time.sleep(2)
            else:
                break
        except NoSuchElementException:
            break

    # Collect all news article elements
    news_elements = driver.find_elements(By.CSS_SELECTOR, '.rankingnews_list .list_title')
    news_items = [{'title': elem.text, 'link': elem.get_attribute('href')} for elem in news_elements]
    driver.quit()

    # Use multiprocessing to fetch article contents
    with Pool(cpu_count()) as pool:
        texts = pool.map(fetch_article_text, news_items)

    # Remove empty strings
    texts = [text for text in texts if text]

    # Extract keywords
    top_keywords = extract_keywords(texts)

    # Print top keywords
    print("\n실시간 키워드 순위:")
    for rank, (keyword, score) in enumerate(top_keywords, 1):
        print(f"{rank}. {keyword} ({score}점)")

    # Generate word cloud
    frequencies = dict(top_keywords)
    generate_wordcloud(frequencies)

if __name__ == "__main__":
    main()
