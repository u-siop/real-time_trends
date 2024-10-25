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

# 웹페이지 스크래핑 함수
def calculate_text_density(html_element):
    text_length = len(html_element.get_text(strip=True))
    tag_length = len(str(html_element))
    return text_length / max(tag_length, 1)

def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 텍스트 밀도 계산 후 가장 높은 블록 선택
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

# 불용어 리스트 생성
stopwords = {'것', '수', '등', '년', '이', '있', '하다', '들', '그', '되다', '보다', '않다', '없다',
             '나', '주다', '한', '때', '안', '중', '로', '고', '말', '거', '일', '더', '와', '의',
             '가', '은', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '한', '하다',
             '때문', '씨', '대한', '하지만', '그리고', '오늘', '어제', '내일', '사람', '또한', '그러나'}

# 키워드 추출 함수 수정: krwordrank 사용 및 어구 추출
def extract_keywords(texts, min_count=5, max_length=10, wordrank_config={'beta': 0.85, 'max_iter': 10},
                     rank_threshold=0.05):
    # texts: 기사들의 텍스트 리스트
    wordrank_extractor = KRWordRank(**wordrank_config)
    keywords, rank, graph = wordrank_extractor.extract(texts, min_count=min_count, max_length=max_length)
    
    # 불용어 필터링 및 rank_threshold 적용
    filtered_keywords = {word: score for word, score in keywords.items()
                         if word not in stopwords and score >= rank_threshold}
    
    # 어구 추출을 위한 토크나이저 생성
    tokenizer = MaxScoreTokenizer(scores=filtered_keywords)
    
    # 각 텍스트에서 어구 추출
    phrases = []
    for text in texts:
        text = normalize(text, english=False, number=False)  # 정규화
        tokens = tokenizer.tokenize(text)
        phrases.extend(tokens)
    
    # 어구 빈도수 계산
    phrase_counts = Counter(phrases)
    
    # 상위 N개 어구 선택
    top_phrases = phrase_counts.most_common(10)
    
    return top_phrases

# 워드클라우드 생성 함수
def generate_wordcloud(frequencies):
    wordcloud = WordCloud(font_path='NanumGothic.ttf', background_color='white', width=800, height=600)
    wordcloud = wordcloud.generate_from_frequencies(frequencies)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# 기사 본문 스크래핑 함수 (병렬 처리용)
def fetch_article_text(news):
    article_text = scrape_webpage(news['link'])
    if article_text:
        # 제목과 본문을 합쳐서 반환
        return news['title'] + ' ' + article_text
    else:
        return ''

# 메인 함수 수정
def main():
    site_url = "https://news.naver.com/main/ranking/popularDay.naver"
    
    # 브라우저 옵션 설정
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(site_url)
    time.sleep(3)
    
    # "다른 언론사 랭킹 더보기" 버튼 클릭 반복
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
    
    # 모든 뉴스 기사 요소 수집
    news_elements = driver.find_elements(By.CSS_SELECTOR, '.rankingnews_list .list_title')
    news_items = [{'title': elem.text, 'link': elem.get_attribute('href')} for elem in news_elements]
    driver.quit()
    
    # 멀티프로세싱을 사용하여 기사 본문 수집
    with Pool(cpu_count()) as pool:
        texts = pool.map(fetch_article_text, news_items)
    
    # 빈 문자열 제거
    texts = [text for text in texts if text]
    
    # 키워드 추출
    top_phrases = extract_keywords(texts)
    
    # 상위 키워드 출력
    print("\n실시간 키워드 순위:")
    for rank, (phrase, count) in enumerate(top_phrases, 1):
        print(f"{rank}. {phrase} ({count}회)")
    
    # 워드클라우드 생성
    frequencies = dict(top_phrases)
    generate_wordcloud(frequencies)

if __name__ == "__main__":
    main()
