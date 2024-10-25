from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options  # 옵션 추가
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from konlpy.tag import Komoran
from collections import Counter
import time
import requests
import networkx as nx
import community  # 커뮤니티 탐지용
from selenium.common.exceptions import NoSuchElementException  # 예외 처리 추가

# Komoran 초기화
komoran = Komoran()

# 불용어 리스트를 파일에서 읽어오기
def load_stopwords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        return stopwords
    except FileNotFoundError:
        print(f"불용어 파일을 찾을 수 없습니다: {file_path}")
        return set()

# 불용어 리스트 로드
stopwords_file = r'C:\news_cash\hot_issue\stopwords-ko.txt'
stopwords = load_stopwords(stopwords_file)

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

# 키워드 추출 함수 (명사와 형용사 포함)
def extract_keywords(text):
    tokens = komoran.pos(text)
    words = [word for word, pos in tokens if (pos.startswith('NN') or pos == 'VA') and word not in stopwords and len(word) > 1]
    return words

# 커뮤니티 탐지 함수
def detect_communities(G):
    partition = community.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    return communities

# 대표 키워드로 문구 생성 함수
def generate_phrase(keywords):
    return ' '.join(keywords)

# 메인 함수 수정
def main():
    site_url = "https://news.naver.com/main/ranking/popularDay.naver"

    # 브라우저 옵션 설정
    options = Options()
    options.add_argument("--headless")  # 브라우저를 표시하지 않음 (백그라운드 실행)
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
                driver.execute_script("arguments[0].click();", more_button)  # 자바스크립트로 클릭
                time.sleep(2)  # 페이지 로딩 시간 대기
            else:
                break
        except NoSuchElementException:
            # 버튼이 없으면 루프를 종료합니다.
            break

    # 모든 뉴스 기사 요소 수집 (상위 60개)
    news_elements = driver.find_elements(By.CSS_SELECTOR, '.rankingnews_list .list_title')
    news_items = [{'title': elem.text, 'link': elem.get_attribute('href')} for elem in news_elements[:60]]
    driver.quit()

    graphs = []  # 여러 개의 그래프를 저장할 리스트

    for news in news_items:
        print(f"Title: {news['title']}")
        article_text = scrape_webpage(news['link'])
        if article_text:
            # 제목과 본문에서 키워드 추출
            keywords = extract_keywords(news['title'] + ' ' + article_text)

            # 키워드 빈도수 계산 후 상위 10개 키워드 선택
            keyword_counts = Counter(keywords)
            top_keywords = [word for word, count in keyword_counts.most_common(10)]

            # 키워드로 그래프 생성
            G_new = nx.Graph()
            # 중복 제거
            top_keywords = list(set(top_keywords))
            for i in range(len(top_keywords)):
                for j in range(i+1, len(top_keywords)):
                    w1, w2 = top_keywords[i], top_keywords[j]
                    G_new.add_edge(w1, w2, weight=1)

            # 기존 그래프들과 연결성 확인
            connected = False
            for G in graphs:
                # 공통 노드가 있는지 확인
                common_nodes = set(G.nodes()).intersection(set(top_keywords))
                if common_nodes:
                    # 그래프 합치기
                    G.add_edges_from(G_new.edges(data=True))
                    connected = True
                    break
            if not connected:
                # 연결되는 그래프가 없으면 새로운 그래프 추가
                graphs.append(G_new)

    # 생성된 그래프의 개수 출력
    print(f"\n생성된 그래프의 개수: {len(graphs)}")

    # 모든 그래프를 합쳐서 대표 키워드 추출
    all_representative_phrases = []

    for G in graphs:
        # 커뮤니티 탐지
        communities = detect_communities(G)

        # 커뮤니티 크기로 정렬하여 상위 N개 선택
        sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        for comm_id, nodes in sorted_communities:
            subgraph = G.subgraph(nodes)
            centrality = nx.degree_centrality(subgraph)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            # 상위 5개 키워드를 문구로 생성
            top_nodes = [node for node, _ in sorted_nodes[:3]]
            phrase = generate_phrase(top_nodes)
            all_representative_phrases.append(phrase)
            # 상위 10개 문구만 수집
            if len(all_representative_phrases) >= 10:
                break
        if len(all_representative_phrases) >= 10:
            break

    print("\n실시간 이슈:")
    for rank, phrase in enumerate(all_representative_phrases[:10], 1):
        print(f"{rank}. {phrase}")

if __name__ == "__main__":
    main()
