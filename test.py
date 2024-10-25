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
import community.community_louvain as community_louvain  # 수정된 부분
from selenium.common.exceptions import NoSuchElementException
import numpy as np

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

# 웹페이지 스크래핑 함수
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

# 텍스트 밀도 계산 함수
def calculate_text_density(html_element):
    text_length = len(html_element.get_text(strip=True))
    tag_length = len(str(html_element))
    return text_length / max(tag_length, 1)

# 키워드 추출 함수
def extract_keywords(text):
    tokens = komoran.pos(text)
    words = [
        word for word, pos in tokens
        if (pos.startswith('NN') or pos == 'NNP') and word not in stopwords and len(word) > 1
    ]
    return words

# 기사 요약 추출 함수 (첫 번째 문장 추출)
def extract_summary(text):
    sentences = text.split('.')
    if sentences:
        return sentences[0].strip()
    else:
        return ""

# 커뮤니티 탐지 함수
def detect_communities(G):
    partition = community_louvain.best_partition(G)
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    return communities

# 대표 키워드로 문구 생성 함수
def generate_phrase(keywords):
    return ' '.join(keywords)

# 그래프 병합 함수
def merge_graphs(G1, G2):
    for node, data in G2.nodes(data=True):
        if G1.has_node(node):
            # Node exists, merge summaries
            G1.nodes[node]['summaries'].extend(data['summaries'])
        else:
            # Node does not exist, add it with its attributes
            G1.add_node(node, **data)
    for u, v, data in G2.edges(data=True):
        if G1.has_edge(u, v):
            # Edge exists, update weight
            G1[u][v]['weight'] += data.get('weight', 1)
        else:
            # Edge does not exist, add it
            G1.add_edge(u, v, **data)

# 그래프 간의 유사도를 계산하는 함수
def calculate_graph_similarity(G1, G2):
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    # Jaccard 유사도를 사용하여 그래프 간 유사도 계산
    intersection = len(nodes1.intersection(nodes2))
    union = len(nodes1.union(nodes2))
    return intersection / union if union != 0 else 0

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

    # 개별 그래프 생성
    graphs = []

    for news in news_items:
        article_text = scrape_webpage(news['link'])
        if article_text:
            # 기사 요약 추출
            summary = extract_summary(article_text)
            # 키워드 추출
            keywords = extract_keywords(news['title'] + ' ' + article_text)
            if not keywords:
                continue
            # 그래프 생성
            G_new = nx.Graph()
            keywords = list(set(keywords))
            for keyword in keywords:
                G_new.add_node(keyword, summaries=[summary])
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    G_new.add_edge(keywords[i], keywords[j], weight=1)
            graphs.append(G_new)

    # 전체 그래프 병합 단계
    final_graphs = []
    while graphs:
        G1 = graphs.pop(0)
        merged = False
        for i, G2 in enumerate(final_graphs):
            # 그래프 간 유사도 계산
            similarity = calculate_graph_similarity(G1, G2)
            if similarity >= 0.3:  # 임계값 설정
                # 그래프 병합
                merge_graphs(G2, G1)
                merged = True
                break
        if not merged:
            # 병합되지 않은 그래프는 최종 그래프 리스트에 추가
            final_graphs.append(G1)

    print(f"\n최종 생성된 그래프의 개수: {len(final_graphs)}")

    # 그래프 별로 대표 이슈 추출
    all_representative_phrases = []

    # 그래프를 노드 수에 따라 정렬 (내림차순)
    sorted_final_graphs = sorted(final_graphs, key=lambda G: G.number_of_nodes(), reverse=True)

    for G in sorted_final_graphs:
        # 커뮤니티 탐지
        communities = detect_communities(G)
        # 커뮤니티 크기로 정렬 (내림차순)
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
            most_common_summary, count = summary_counter.most_common(1)[0]
            # 중심성 계산
            centrality = nx.degree_centrality(subgraph)
            sorted_nodes = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )
            # 상위 5개 키워드를 문구로 생성
            top_nodes = [node for node, _ in sorted_nodes[:5]]
            phrase = generate_phrase(top_nodes)
            # 요약문과 키워드 결합 및 커뮤니티 크기 정보 추가
            combined_phrase = {
                'summary': most_common_summary,
                'phrase': phrase,
                'community_size': len(nodes),
                'summary_count': count
            }
            all_representative_phrases.append(combined_phrase)
            # 상위 10개 문구만 수집
            if len(all_representative_phrases) >= 50:  # 충분히 수집 후 정렬하여 상위 10개 선택
                break
        if len(all_representative_phrases) >= 50:
            break

    # 이슈의 중요도에 따라 정렬 (요약문 수와 커뮤니티 크기를 고려)
    sorted_phrases = sorted(
        all_representative_phrases,
        key=lambda x: (x['summary_count'], x['community_size']),
        reverse=True
    )

    print("\n실시간 이슈:")
    for rank, item in enumerate(sorted_phrases[:10], 1):
        print(f"{rank}. {item['summary']} - {item['phrase']}")

if __name__ == "__main__":
    main()