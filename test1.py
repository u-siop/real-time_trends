from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from konlpy.tag import Komoran
from collections import Counter
import time
import requests
import networkx as nx
import community
from selenium.common.exceptions import NoSuchElementException
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
stopwords_file = 'stopwords-ko.txt'  # 불용어 파일 경로를 적절히 수정하세요
stopwords = load_stopwords(stopwords_file)

# 텍스트 전처리 함수
def preprocess_text(text):
    tokens = komoran.morphs(text)
    tokens = [word for word in tokens if word not in stopwords and len(word) > 1]
    return ' '.join(tokens)

# 웹페이지 스크래핑 함수 (이전과 동일)
def scrape_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 텍스트 밀도 계산 후 가장 높은 블록 선택
        candidate_blocks = soup.find_all(['div', 'article', 'section'])
        blocks_with_density = [
            (calculate_text_density(block), block) for block in candidate_blocks
        ]
        blocks_with_density.sort(key=lambda x: x[0], reverse=True)

        for density, block in blocks_with_density:
            if density > 0.1:
                # 불필요한 태그 제거
                for tag in block.find_all(['em', 'strong', 'script', 'style',
                                           'span', 'div', 'a', 'img']):
                    tag.decompose()
                article_text = block.get_text(separator=' ', strip=True)
                if len(article_text) > 200:
                    # 여러 개의 공백을 하나로
                    article_text = ' '.join(article_text.split())
                    return article_text
        print("본문을 찾을 수 없습니다.")
        return ""
    except Exception as e:
        print(f"Error scraping webpage: {e}")
        return ""

# 텍스트 밀도 계산 함수 (이전과 동일)
def calculate_text_density(html_element):
    text_length = len(html_element.get_text(strip=True))
    tag_length = len(str(html_element))
    return text_length / max(tag_length, 1)

# 기사 요약 추출 함수 (첫 번째 문장 추출)
def extract_summary(text):
    sentences = text.split('.')
    if sentences:
        return sentences[0].strip()
    else:
        return ""

# 키워드 추출 함수 (명사 및 고유명사 추출)
def extract_keywords(text):
    tokens = komoran.pos(text)
    words = [
        word for word, pos in tokens
        if (pos.startswith('NN') or pos == 'NNP') and word not in stopwords and len(word) > 1
    ]
    return words

# 커뮤니티 탐지 함수 (이전과 동일)
def detect_communities(G):
    partition = community.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    return communities

# 대표 키워드로 문구 생성 함수 (이전과 동일)
def generate_phrase(keywords):
    return ' '.join(keywords)

# 메인 함수
def main():
    site_url = "https://news.naver.com/main/ranking/popularDay.naver"

    # 브라우저 옵션 설정
    options = Options()
    options.add_argument("--headless")  # 브라우저를 표시하지 않음
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    driver.get(site_url)
    time.sleep(3)

    # "다른 언론사 랭킹 더보기" 버튼 클릭 반복
    while True:
        try:
            more_button = driver.find_element(
                By.CSS_SELECTOR, '.button_rankingnews_more'
            )
            if more_button.is_displayed():
                driver.execute_script("arguments[0].click();", more_button)
                time.sleep(2)  # 페이지 로딩 시간 대기
            else:
                break
        except NoSuchElementException:
            # 버튼이 없으면 루프를 종료합니다.
            break

    # 모든 뉴스 기사 요소 수집 (상위 60개)
    news_elements = driver.find_elements(
        By.CSS_SELECTOR, '.rankingnews_list .list_title'
    )
    news_items = [
        {'title': elem.text, 'link': elem.get_attribute('href')}
        for elem in news_elements
    ]
    driver.quit()

    # 기사 본문 및 요약 수집
    documents = []
    summaries = []
    for news in news_items:
        print(f"Title: {news['title']}")
        article_text = scrape_webpage(news['link'])
        if article_text:
            # 기사 요약 추출
            summary = extract_summary(article_text)
            summaries.append(summary)
            # 제목과 본문 결합
            full_text = news['title'] + ' ' + article_text
            documents.append(full_text)

    # 텍스트 전처리
    preprocessed_texts = [preprocess_text(doc) for doc in documents]

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)

    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(tfidf_matrix)

    # 기사 그룹화 (유사도 임계치 설정)
    similarity_threshold = 0.2  # 유사도 임계값 (조정 가능)
    groups = []
    grouped_indices = set()

    for idx in range(len(documents)):
        if idx in grouped_indices:
            continue
        group = [idx]
        grouped_indices.add(idx)
        for jdx in range(idx + 1, len(documents)):
            if jdx in grouped_indices:
                continue
            if cosine_similarities[idx, jdx] >= similarity_threshold:
                group.append(jdx)
                grouped_indices.add(jdx)
        groups.append(group)

    print(f"\n생성된 기사 그룹의 개수: {len(groups)}")

    # 그룹별로 그래프 생성 및 대표 키워드 추출
    all_representative_phrases = []

    for group in groups:
        group_documents = [documents[i] for i in group]
        group_summaries = [summaries[i] for i in group]
        # 그룹 내 모든 키워드 추출
        keywords = []
        for doc in group_documents:
            keywords.extend(extract_keywords(doc))
        if not keywords:
            continue
        # 키워드 빈도수 계산 후 상위 10개 키워드 선택
        keyword_counts = Counter(keywords)
        top_keywords = [word for word, count in keyword_counts.most_common(10)]
        # 그래프 생성
        G = nx.Graph()
        # 중복 제거
        top_keywords = list(set(top_keywords))
        for keyword in top_keywords:
            G.add_node(keyword, summaries=group_summaries)
        for i in range(len(top_keywords)):
            for j in range(i + 1, len(top_keywords)):
                w1, w2 = top_keywords[i], top_keywords[j]
                G.add_edge(w1, w2, weight=1)
        # 커뮤니티 탐지
        communities = detect_communities(G)
        # 커뮤니티 크기로 정렬하여 상위 N개 선택
        sorted_communities = sorted(
            communities.items(), key=lambda x: len(x[1]), reverse=True
        )
        for comm_id, nodes in sorted_communities:
            subgraph = G.subgraph(nodes)
            # 노드에서 summaries를 수집
            community_summaries = []
            for node in subgraph.nodes():
                community_summaries.extend(subgraph.nodes[node]['summaries'])
            # 가장 많이 등장한 요약문 선택
            summary_counter = Counter(community_summaries)
            most_common_summary, _ = summary_counter.most_common(1)[0]
            # 중심성 계산
            centrality = nx.degree_centrality(subgraph)
            sorted_nodes = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )
            # 상위 5개 키워드를 문구로 생성
            top_nodes = [node for node, _ in sorted_nodes[:5]]
            phrase = generate_phrase(top_nodes)
            # 요약문과 키워드 결합
            combined_phrase = f"{most_common_summary} - {phrase}"
            all_representative_phrases.append(combined_phrase)
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
