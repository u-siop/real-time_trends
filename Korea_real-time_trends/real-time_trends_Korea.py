import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # FutureWarning 숨기기 설정

import os
import re
import time
import logging
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from konlpy.tag import Komoran  # Okt 대신 Komoran 사용
from soynlp.word import WordExtractor
from pytrends.request import TrendReq

import numpy as np
from tqdm import tqdm  # 진행 표시기 추가

# 추가된 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys

# OpenAI API key 설정

if sys.version_info >= (3, 12):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)
else:
    import openai
    openai.api_key = API_KEY
    client = openai  # 이후 코드에서 일관된 사용을 위해


import tkinter as tk
from tkinter import scrolledtext

# 로깅 설정
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# 형태소 분석기 초기화 (Komoran)
komoran = Komoran()

# 불용어 리스트를 파일에서 읽어오기
def load_stopwords(file_path):
    default_stopwords = set([
        # 기존 불용어에 조사, 접속사 등 추가
        '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과',
        '도', '를', '으로', '자', '에', '와', '한', '하다', '것', '말',
        '그', '저', '및', '더', '등', '데', '때', '년', '월', '일',
        '그리고', '하지만', '그러나', '그래서', '또는', '즉', '만약', '아니면',
        '때문에', '그런데', '그러므로', '따라서', '뿐만 아니라',
        '이런', '저런', '합니다', '있습니다', '했습니다', '있다', '없다',
        '됩니다', '되다', '이다'
    ])
    if not os.path.exists(file_path):
        logging.warning(f"불용어 파일을 찾을 수 없습니다: {file_path}. 기본 불용어를 사용합니다.")
        return default_stopwords
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        stopwords.update(default_stopwords)
        # 추가 불용어
        additional_stopwords = {'대표', 'A씨', 'B씨', '것으로', '등등', '있다', '없다'}
        stopwords.update(additional_stopwords)
        logging.info(f"불용어 {len(stopwords)}개를 로드했습니다.")
        return stopwords
    except Exception as e:
        logging.error(f"불용어 파일 로드 중 오류 발생: {e}")
        return default_stopwords

# 불용어 리스트 로드
stopwords_file = 'stopwords-ko.txt'  # 불용어 파일 경로를 적절히 수정하세요
stopwords = load_stopwords(stopwords_file)

# 구텐베르크 알고리즘을 사용하기 위한 텍스트 밀도 계산 함수
def calculate_text_density(html_element):
    text_length = len(html_element.get_text(strip=True))
    tag_length = len(str(html_element))
    return text_length / max(tag_length, 1)

# 웹페이지 스크래핑 함수
def scrape_webpage(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 네이버 뉴스 본문 추출
        article_body = soup.select_one('#dic_area')
        if article_body:
            article_text = article_body.get_text(separator=' ', strip=True)
            return article_text
        else:
            logging.warning(f"본문을 찾을 수 없습니다: {url}")
            return ""
    except Exception as e:
        logging.error(f"웹페이지 스크래핑 오류 ({url}): {e}")
        return ""

# 텍스트 밀도를 이용하여 뉴스 본문만을 스크랩하여 반환하는 함수
def scrape_webpage_for_google_search(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')

        # 태그 밀도가 높을 수 있는 사이드, 배너, 광고 등을 수작업으로 제거하기 위한 ID 및 클래스 목록
        unwanted_ids = [
            'newsSidebar', 'newsMainBanner', 'rightSlideDiv_1', 'rightSlideDiv_2', 'rightSlideDiv_3',
        ]
        unwanted_classes = [
            'sidebar', 'rankingNews', 'photo_slide', 'ad290x330', 'socialAD', 'AdIbl', 'rankingEmotion',
            'ofhe_head', 'ofhe_body', 'outside_area_inner', 'outside_area', '_OUTSIDE_AREA', '_GRID_TEMPLATE_COLUMN_ASIDE', '_OUTSIDE_AREA_INNER',
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
            density = calculate_text_density(block)
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
        logging.error(f"웹페이지 스크래핑 중 오류 발생 ({url}): {e}")
        return ""

# Google 트렌드 키워드 수집 함수
def get_google_trends_keywords():
    try:
        pytrends = TrendReq(hl='ko', tz=540)
        # 실시간 검색어는 pytrends에서 직접적으로 지원하지 않으므로, 일반 트렌딩 검색어를 사용
        df_trending = pytrends.trending_searches(pn='south_korea')  # 한국의 실시간 트렌드
        trending_keywords = df_trending[0].tolist()
        return trending_keywords
    except Exception as e:
        logging.error(f"Google 트렌드 키워드 수집 중 오류 발생: {e}")
        return []

# 텍스트 전처리 함수
def preprocess_text(text):
    if not text or not isinstance(text, str) or not text.strip():
        logging.warning("유효하지 않은 입력 텍스트.")
        return ""
    # 특수 문자 제거
    text = re.sub(r'[^0-9가-힣a-zA-Z\s]', ' ', text)
    # 여러 개의 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# N-그램 기반 키워드 추출 함수 (멀티 워드 지원)
def extract_ngrams(tokens, n=2):
    if len(tokens) < n:
        return []
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]  # 단어 순서 유지

# 키워드 유사도 계산 함수 (Jaccard 유사도)
def calculate_jaccard_similarity(keywords1, keywords2):
    set1 = set(keywords1)
    set2 = set(keywords2)
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)

# 대표 키워드 선정 함수 (Google 트렌드 우선, 불용어 필터링 추가)
def select_representative_keyword(top_keywords, used_keywords, google_trends_keywords):
    # Google 트렌드 키워드를 우선적으로 선택
    for kw in top_keywords:
        if kw in google_trends_keywords and kw not in used_keywords and kw not in stopwords:
            return kw
    # 나머지 키워드 중 사용되지 않았고 불용어가 아닌 키워드 선택
    for kw in top_keywords:
        if kw not in used_keywords and kw not in stopwords:
            return kw
    return None

# 중복 키워드 제거 함수 (부분 문자열 및 동일 키워드 제거)
def remove_substrings_and_duplicates(keywords):
    unique_keywords = []
    sorted_keywords = sorted(keywords, key=lambda x: len(x), reverse=True)
    for kw in sorted_keywords:
        if kw in unique_keywords:
            continue
        if not any((kw != other and kw in other) for other in unique_keywords):
            unique_keywords.append(kw)
    return unique_keywords

# 의미 없는 키워드 제거 함수
def remove_invalid_keywords(keywords):
    # 한글만 포함되고, 2자 이상인 키워드만 유지
    valid_keywords = [kw for kw in keywords if re.match(r'^[0-9a-zA-Z가-힣]{2,}(?: [0-9a-zA-Z가-힣]{2,})*$', kw)]
    return valid_keywords

# 트리 별로 대표 이슈 추출 함수 (키워드 구문 생성)
def extract_representative_info(trees, source='Naver', max_words=5):  # max_words를 5로 증가
    trees_info = []
    used_keywords = set()  # 중복 방지를 위한 키워드 사용 기록
    for tree_info in trees:
        articles = tree_info['articles']
        # 트리 중요도 계산 (뉴스 기사 수 또는 검색량)
        importance = tree_info.get('importance', len(articles))
        # 대표 키워드 선정 (트리 내 가장 많이 등장한 키워드)
        keyword_counter = Counter()
        for news in articles:
            if 'keywords' in news:
                keyword_counter.update([kw for kw in news.get('keywords', []) if kw not in stopwords])
            else:
                continue
        top_keywords = [word for word, freq in keyword_counter.most_common(5)]
        if not top_keywords:
            continue  # 키워드가 없으면 스킵
        # 대표 키워드 선정 시 Google 트렌드 키워드 우선
        rep_keyword = select_representative_keyword(top_keywords, used_keywords, tree_info.get('google_trends_keywords', []))
        if not rep_keyword:
            rep_keyword = top_keywords[0] if top_keywords else None
        if not rep_keyword:
            continue  # 대표 키워드가 없으면 스킵
        used_keywords.add(rep_keyword)  # 사용된 키워드 기록
        # 대표 키워드 제외 상위 키워드
        top_other_keywords = [kw for kw in top_keywords if kw != rep_keyword]
        if top_other_keywords:
            # 이슈 구문은 최대 5개의 키워드로 제한
            phrase_keywords = [rep_keyword] + top_other_keywords[:max_words-1]
            # 중복 단어 제거 및 부분 문자열 제거
            phrase_keywords = remove_substrings_and_duplicates(phrase_keywords)
            # 의미 없는 키워드 제거
            phrase_keywords = remove_invalid_keywords(phrase_keywords)
            # 최대 단어 수 제한
            if len(phrase_keywords) > max_words:
                phrase_keywords = phrase_keywords[:max_words]
            # 조건에 맞지 않으면 대표 키워드만 사용
            if len(phrase_keywords) < 2:
                phrase_keywords = [rep_keyword]
            phrase = ', '.join(phrase_keywords)
        else:
            phrase = rep_keyword

        combined_info = {
            'phrase': phrase,
            'importance': importance,
            'source': source  # 트리의 출처 추가
        }
        trees_info.append(combined_info)
        logging.info(f"Representative issue added: {phrase} - Importance: {importance} - Source: {source}")
    return trees_info

# 키워드 추출 함수 (soynlp 이용 + N-그램 + 멀티 워드)
def extract_keywords(text, stopwords, top_n=10):
    try:
        # konlpy를 이용한 명사 추출 (Komoran 사용)
        tokens = komoran.nouns(text)
        # 불용어 및 1단어, 제한 없는 단어 필터링 (최소 2자 이상)
        filtered_tokens = [word for word in tokens if len(word) >= 2 and word not in stopwords]
        # 단어 빈도수 계산
        token_counts = Counter(filtered_tokens)
        top_keywords = [word for word, count in token_counts.most_common(top_n)]

        # 멀티 워드 추출 (2-그램부터 5-그램까지)
        combined_bigrams = []
        for n in range(2, 6):  # 2-그램부터 5-그램까지
            ngrams = extract_ngrams(filtered_tokens, n=n)
            combined_bigrams.extend(ngrams)

        # N-그램 빈도수 계산
        ngram_counts = Counter([bg for bg in combined_bigrams if bg not in stopwords and len(bg.replace(' ', '')) >= 2])
        sorted_ngrams = [bg for bg, cnt in ngram_counts.most_common(top_n)]

        # 결합된 키워드 목록
        combined_keywords = top_keywords + sorted_ngrams

        # 중복 제거 및 상위 N개 선택
        unique_keywords = []
        # Sort by length descending to keep longer keywords first
        combined_keywords_sorted = sorted(combined_keywords, key=lambda x: len(x), reverse=True)
        for kw in combined_keywords_sorted:
            if kw in unique_keywords:
                continue
            if not any((kw != other and kw in other) for other in unique_keywords):
                unique_keywords.append(kw)
            if len(unique_keywords) >= top_n:
                break

        # 의미 없는 키워드 제거
        unique_keywords = remove_invalid_keywords(unique_keywords)

        return unique_keywords[:top_n]
    except Exception as e:
        logging.error(f"키워드 추출 중 오류 발생: {e}")
        return []

# 네이버 뉴스 검색 및 상위 3개 키워드 추출 함수
def search_naver_news_with_keyword(keyword, stopwords):
    try:
        search_url = f"https://search.naver.com/search.naver?&where=news&query={requests.utils.quote(keyword)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 뉴스 검색 결과에서 제목과 링크 추출
        news_elements = soup.select('.list_news .news_area')
        news_items = []
        for elem in news_elements:
            title_elem = elem.select_one('.news_tit')
            if title_elem:
                title = title_elem.get_text(strip=True)
                link = title_elem.get('href')
                news_items.append({'title': title, 'link': link})

        # 상위 10개 뉴스 기사에서 텍스트 추출
        articles_texts = []
        for news in news_items[:10]:
            article_text = scrape_webpage_for_google_search(news['link'])
            if article_text:
                full_text = preprocess_text(news['title'] + ' ' + article_text)
                if full_text:
                    articles_texts.append(full_text)

        if not articles_texts:
            logging.warning(f"네이버 뉴스 검색 결과에서 텍스트를 추출할 수 없습니다: {keyword}")
            return []

        # 키워드 추출
        keywords = extract_keywords(' '.join(articles_texts), stopwords, top_n=10)
        top_3_keywords = keywords[:3] if len(keywords) >= 3 else keywords

        logging.info(f"키워드 '{keyword}'에 대한 상위 3개 키워드: {top_3_keywords}")
        return top_3_keywords
    except Exception as e:
        logging.error(f"네이버 뉴스 검색 중 오류 발생 ({keyword}): {e}")
        return []

# 커스텀 토크나이저 정의 (쉼표 또는 공백을 기준으로 단어 분리)
def custom_tokenizer(x):
    return re.split(r',\s*|\s+', x)

# 연결 요소 기반 이슈 병합 함수 (토크나이저 수정됨)
def merge_similar_issues(issues, similarity_threshold=0.3):
    """
    이슈 리스트에서 유사도가 임계값 이상인 이슈들을 병합합니다.
    Args:
        issues (list of dict): 병합할 이슈 리스트. 각 dict는 'phrase' 키를 가져야 합니다.
        similarity_threshold (float): 유사도 임계값 (0 ~ 1).
    Returns:
        list of dict: 병합된 이슈 리스트.
    """
    if not issues:
        return []

    phrases = [issue['phrase'] for issue in issues]
    # TF-IDF 벡터화 (쉼표 또는 공백을 기준으로 단어 분리)
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '), lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(phrases)

    # 코사인 유사도 행렬 계산
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 연결 요소 찾기 (Union-Find 알고리즘)
    n = len(issues)
    parent = list(range(n))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return
        parent[v_root] = u_root

    # 유사도 임계값 이상인 쌍을 연결
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= similarity_threshold:
                union(i, j)

    # 그룹핑
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # 그룹별로 이슈 병합
    merged_issues = []
    for group in groups.values():
        if not group:
            continue
        if len(group) == 1:
            # 병합할 필요 없음
            merged_issues.append(issues[group[0]])
        else:
            # 병합된 이슈의 구문을 결합하고, 중요도는 합산
            combined_phrases = [issues[idx]['phrase'] for idx in group]
            combined_importance = sum([issues[idx]['importance'] if issues[idx]['importance'] else 0 for idx in group])
            combined_sources = set([issues[idx]['source'] for idx in group])

            # 중복 단어 및 부분 문자열 제거
            words = set(', '.join(combined_phrases).split(', '))
            words = remove_substrings_and_duplicates(words)
            # 의미 없는 키워드 제거
            words = remove_invalid_keywords(words)
            if not words:
                continue
            combined_phrases_cleaned = ', '.join(sorted(words))
            merged_issues.append({
                'phrase': combined_phrases_cleaned,
                'importance': combined_importance,
                'source': ', '.join(combined_sources)
            })
            logging.info(f"이슈 병합: {[issues[idx]['phrase'] for idx in group]} -> {combined_phrases_cleaned}")

    return merged_issues

# 키워드 요약 함수 (OpenAI API 이용)
def summarize_keywords(content):
    prompt = f"""
[요약하는 방법]
1. 아래에 나열된 여러 개의 키워드를 기반으로 핵심 키워드 10개를 2어절 이내로 요약해줘.
2. 유사하거나 중복된 내용의 키워드는 하나의 항목으로 병합해줘.
   - 예: "대통령 직무, 부정 평가, 긍정 평가" -> "대통령 직무 평가"
3. 정치에 관한 내용은 최대 2개로 제한하며, 비슷한 내용은 합치지 말고 제외해줘.
4. 각 요약 항목은 명확하고 간결해야 하며, 핵심 주제를 반영해야 해.
5. 동일한 키워드가 여러 번 등장하면 한 번만 포함시켜줘.
6. 최신의 실제 이슈를 반영하여 현재 상황에 맞는 용어와 표현을 사용해줘.

[예시1]
1. 대통령 직무, 부정 평가, 긍정 평가
2. 불법 영업, 사생활 논란, 음식점 운영, 트리플 스타, 트리플star, 흑백 요리사, 유비빔
3. 국정 감사, 국회 운영, 대통령 관저, 대통령 다혜, 대통령 명태, 명태균, 문재인 대통령, 여론 조사, 윤석열 대통령, 정진석 대통령, 참고인 조사
4. 아버지 살해, 아버지 둔기, 30대 남성
5. 소말리, 소녀상 모욕, 편의점 난동, 조니 말리
6. 23기 정숙, 출연자 검증, 논란 제작진, 유튜브 채널
7. GD, 베이비 몬스터, 더블 타이틀, 몬스터 정규
8. 기아 타이, 타이 거즈, 기아 세일
9. 테슬라 코리아, 김예지 국내, 최초 테슬라
10. 북한군 교전, 북한군 추정, 주장 북한군

==> 요약된 실시간 이슈:
1. 대통령 직무 평가
2. 불법 영업 논란
3. 국정 감사 및 여론
4. 아버지 둔기로 살해
5. 소녀상 모욕 사건
6. 23기 정숙, 출연자 검증 논란
7. GD와 베이비 몬스터
8. 기아타이거즈 세일
9. 테슬라, 김예지
10. 북한군 교전 주장

[예시2]
1. 도널드 트럼프 대통령, 해리스 부통령, 카멀 해리스, 여론 조사, 대선 후보
2. 탄도 미사일, 미사일 발사, 미사일 화성, 대륙 탄도, 시험 발사
3. 통영 어부 장터 축제, 백종원 코리아, 해산물 축제, 사과, 준비
4. 20대 여성, 면허 취득, 서울 강남, 운전 학원, 추돌 사고
5. 대통령 기자회견, 윤석열 대통령, 대통령 국민, 김건희 여사, 국민 담화
6. 한국 부채춤, 부채춤 중국, 중국 민간, 민간 전통, 전통 무용
7. 원내대표 대통령, 국민 원내대표, 추경호 국민, 원내 대책, 대책 회의
8. 아버지 시신 냉동, 이혼 소송 진행, 아버지 상태, 아버지 사망, 경찰
9. 최고 위원, 국민 담화, 여사 판단, 판단 변화, 변화 작동
10. 추경호 원내대표, 국민 소통 기회, 기자회견 최종, 대통령 국민, 말씀
11. 미국 대선, 단거리 탄도 미사일, 트럼프 대통령, 해리스 부통령
12. 롤드컵, 영국 런던 O2 아레나, O2 아레나 결승전, 페이 이상혁
13. 금투세, 금융 투자 소득세, 개인 투자자, 위탁 운용사
14. 사이버트럭, 테슬라 전기 픽업 트럭, 사이버 트럭 지드래곤, 테슬라 사이버 트럭
15. 김하온, 블록 토너먼트, 오리지널 예능, 티빙 오리지널
16. 원내대표 대통령, 국민 원내대표, 추경호 국민, 원내 대책, 대책 회의
17. 아버지 시신 냉동, 이혼 소송 진행, 아버지 상태, 아버지 사망, 경찰
18. 최고 위원, 국민 담화, 여사 판단, 판단 변화, 변화 작동
19. 추경호 원내대표, 국민 소통 기회, 기자회견 최종, 대통령 국민, 말씀
20. 온라인 커뮤니티, 커뮤니티 갈무리, 사고 당시 모습, 추돌 사고, 여성

요약된 실시간 이슈:
1. 미국 대선 후보
2. 북한, 탄도 미사일 시험 발사
3. 통영 해산물 축제
4. 강남 추돌 사고 
5. 윤석열 대통령 기자회견
6. 한국 부채춤 중국 논란
7. 국민의 힘, 원내대표 대책 회의
8. 아버지 시신 냉동 사건
9. 금투세 금융 투자 소득세
10. 지드래곤, 테슬라 사이버트럭


다음은 요약할 텍스트: {content}
"""

    if sys.version_info >= (3, 12):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 사용 가능한 모델 이름으로 변경
                messages=[
                    {"role": "system", "content": "You are a helpful newsletter artist that summarizes keywords to news keywords for SNS."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.15,
            )

            summary_text = response.choices[0].message.content.strip()
            return summary_text

        except Exception as e:
            logging.error(f"요약 생성 중 오류 발생: {e}")
            return "요약 생성에 실패했습니다."
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful newsletter artist that summarizes keywords to news keywords for SNS."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.15,
            )

            summary_text = response.choices[0].message.content.strip()
            return summary_text

        except Exception as e:
            logging.error(f"요약 생성 중 오류 발생: {e}")
            return "요약 생성에 실패했습니다."


# 메인 함수
def main():
    # Google 트렌드 키워드 수집
    trending_keywords = get_google_trends_keywords()
    logging.info(f"Google 트렌드 키워드 수집 완료: {len(trending_keywords)}개")
    
    keyword_volume = []
    trend_trees = []
    
    logging.info("Google 트렌드 검색량 수집 시작")
    
    # tqdm을 이용한 진행 표시기 추가
    for keyword in tqdm(trending_keywords, desc="Google 트렌드 키워드 처리"):
        try:
            if keyword in stopwords:
                logging.info(f"불용어 키워드 스킵: {keyword}")
                continue
            pytrends = TrendReq(hl='ko', tz=540)
            pytrends.build_payload([keyword], cat=0, timeframe='now 1-d', geo='KR')
            interest_over_time_df = pytrends.interest_over_time()

            if not interest_over_time_df.empty:
                avg_volume = interest_over_time_df[keyword].mean()
                keyword_volume.append((keyword, avg_volume))
                logging.info(f"키워드 '{keyword}'의 평균 검색량: {avg_volume}")
            else:
                keyword_volume.append((keyword, 0))
                logging.info(f"키워드 '{keyword}'의 검색량 데이터가 없습니다.")
            
        except Exception as e:
            logging.error(f"키워드 '{keyword}' 검색량 가져오는 중 오류 발생: {e}")
            keyword_volume.append((keyword, 0))
    
    # 검색량 기준으로 키워드 정렬 (내림차순) 및 상위 5개 선택
    sorted_keywords = sorted(keyword_volume, key=lambda x: x[1], reverse=True)[:5]  # 상위 5개만 선택
    logging.info("검색량 기준으로 정렬된 상위 5개 트렌드 키워드:")
    for i, (kw, vol) in enumerate(sorted_keywords, 1):
        logging.info(f"{i}. {kw} - 검색량: {vol}")
    
    # 트렌드 트리 생성 (검색량 기준으로 중요도 설정)
    for keyword, volume in tqdm(sorted_keywords, desc="Google 트렌드 트리 생성"):
        if keyword in stopwords or volume <= 0:
            continue
        # 키워드 전처리 및 추출
        full_text = preprocess_text(keyword)
        keywords = extract_keywords(full_text, stopwords, top_n=10)
        if not keywords:
            continue
        # 트렌드 트리 생성
        trend_trees.append({
            'articles': [{
                'title': keyword,
                'link': 'https://trends.google.com/trends/trendingsearches/daily?geo=KR',
                'keywords': keywords
            }],
            'all_keywords': set([kw for kw in keywords if kw not in stopwords]),
            'google_trends_keywords': [keyword],  # 각 트렌드 트리에 해당하는 단일 트렌드 키워드 추가
            'importance': volume  # 검색량을 중요도로 설정
        })
        logging.info(f"Google 트렌드 새로운 트리 생성: {keyword} (검색량: {volume})")
    
    logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")
    
    # Google 트렌드 상위 5개 키워드를 네이버 뉴스에서 검색하여 상위 3개 키워드 추출 및 매핑 저장
    trend_top3_keywords = []
    logging.info("Google 트렌드 키워드를 네이버 뉴스에서 검색하여 상위 3개 키워드 추출 시작")
    for keyword, _ in tqdm(sorted_keywords, desc="Google 트렌드 키워드 네이버 뉴스 검색"):
        top3 = search_naver_news_with_keyword(keyword, stopwords)
        if top3:
            trend_top3_keywords.append({
                'trend_keyword': keyword,
                'naver_keywords': top3,
                'source': 'Google Trends (Naver Search)',
                'importance': next((vol for kw, vol in keyword_volume if kw == keyword), 1)  # 검색량을 중요도로 설정
            })
    
    logging.info(f"Google 트렌드 키워드 기반 추출된 상위 3개 키워드의 총 개수: {len(trend_top3_keywords) * 3}")
    
    # 트렌드 키워드와 네이버 키워드의 조합을 저장할 리스트
    combined_phrases = []
    
    for item in trend_top3_keywords:
        trend_kw = item['trend_keyword']
        naver_kws = item['naver_keywords']
        importance = item['importance']
        # 트렌드 키워드와 네이버 키워드를 결합하여 새로운 구문 생성
        # Google 트렌드 키워드를 맨 앞에 배치
        combined_set = [trend_kw] + naver_kws[:3]
        phrase = ', '.join(combined_set)
        # 의미 없는 키워드 제거
        phrase_keywords = remove_invalid_keywords(phrase.split(', '))
        phrase = ', '.join(phrase_keywords)
        if not phrase:
            continue
        combined_phrases.append({
            'phrase': phrase,
            'importance': importance,  # 검색량을 중요도로 설정
            'source': 'Google Trends + Naver Search'
        })
        logging.info(f"조합된 이슈 구문 추가: {phrase}")
    
    # 네이버 뉴스 처리
    site_url = "https://news.naver.com/main/ranking/popularDay.naver"

    # 브라우저 옵션 설정
    options = Options()
    options.add_argument("--headless")  # 브라우저를 표시하지 않음
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # WebDriver를 with 구문으로 안전하게 관리
    with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) as driver:
        driver.get(site_url)
        time.sleep(3)

        # "다른 언론사 랭킹 더보기" 버튼 클릭 반복
        while True:
            try:
                more_button = driver.find_element(By.CSS_SELECTOR, '.button_rankingnews_more')
                if more_button.is_displayed():
                    driver.execute_script("arguments[0].click();", more_button)
                    logging.info("더보기 버튼 클릭")
                    time.sleep(2)  # 페이지 로딩 시간 대기
                else:
                    break
            except NoSuchElementException:
                # 버튼이 없으면 루프를 종료합니다.
                logging.info("더보기 버튼이 더 이상 존재하지 않습니다.")
                break
            except Exception as e:
                logging.error(f"더보기 버튼 클릭 중 오류: {e}")
                break

        # 모든 뉴스 기사 요소 수집
        news_elements = driver.find_elements(By.CSS_SELECTOR, '.rankingnews_list .list_title')
        news_items = [
            {'title': elem.text, 'link': elem.get_attribute('href')}
            for elem in news_elements 
        ]
        logging.info(f"수집된 뉴스 기사 수: {len(news_items)}")

    # 기사별 텍스트 수집
    articles_texts = []
    articles_metadata = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_news = {executor.submit(scrape_webpage, news['link']): news for news in news_items}
        # tqdm을 이용한 진행 표시기 추가
        for future in tqdm(as_completed(future_to_news), total=len(future_to_news), desc="뉴스 기사 스크래핑"):
            news = future_to_news[future]
            try:
                article_text = future.result()
                if article_text:
                    # 텍스트를 합침 (제목 + 본문)
                    full_text = preprocess_text(news['title'] + ' ' + article_text)
                    if full_text:
                        articles_texts.append(full_text)
                        articles_metadata.append({'title': news['title'], 'link': news['link']})
                else:
                    logging.warning(f"기사 본문이 없습니다: {news['title']}")
            except Exception as e:
                logging.error(f"뉴스 기사 처리 중 오류 ({news['title']}): {e}")

    logging.info(f"수집된 유효한 기사 수: {len(articles_texts)}")

    if not articles_texts:
        logging.error("유효한 기사가 하나도 없습니다. 프로그램을 종료합니다.")
        return

    # WordExtractor 초기화 및 훈련 (전체 코퍼스 기반)
    word_extractor = WordExtractor()
    logging.info("WordExtractor 훈련 시작")
    word_extractor.train(articles_texts)
    logging.info("WordExtractor 훈련 완료")

    words = word_extractor.extract()
    logging.info(f"추출된 단어 수: {len(words)}")

    # 뉴스 기사별로 키워드 추출
    articles_keywords_list = []
    for idx, text in tqdm(enumerate(articles_texts), total=len(articles_texts), desc="뉴스 기사 키워드 추출"):
        keywords = extract_keywords(text, stopwords, top_n=10)
        if keywords:
            articles_keywords_list.append(keywords)
        else:
            articles_keywords_list.append([])
        # 각 뉴스 기사에 키워드 추가
        articles_metadata[idx]['keywords'] = keywords

    # 트리 및 뉴스 정보 저장
    naver_trees = []  # 네이버 뉴스 트리 정보의 리스트

    # 네이버 뉴스 기사별로 트리 생성
    for idx, news in enumerate(tqdm(articles_metadata, desc="네이버 트리 생성")):
        keywords = articles_keywords_list[idx]
        if not keywords:
            continue
        merged = False
        # 기존 네이버 트리들과 비교하여 유사하면 병합
        for tree_info in naver_trees:
            similarity = calculate_jaccard_similarity(
                keywords, 
                tree_info['all_keywords']
            )
            logging.debug(f"Comparing '{news['title']}' with tree keywords: {tree_info['all_keywords']} | Similarity: {similarity}")
            if similarity >= 0.2:  # 유사도 임계값을 0.2로 조정
                # 트리에 뉴스 추가
                tree_info['articles'].append(news)
                tree_info['all_keywords'].update([kw for kw in keywords if kw not in stopwords])  # 집합으로 업데이트
                # 트리에 Google 트렌드 키워드 포함 여부 갱신
                tree_info['contains_trend'] = tree_info['contains_trend'] or any(word in trending_keywords for word in keywords)
                # 트리의 중요도 업데이트 (기사 수)
                tree_info['importance'] += 1
                merged = True
                logging.info(f"네이버 트리 병합 완료: {news['title']} (유사도: {similarity:.2f})")
                break
        if not merged:
            # 새로운 네이버 트리 생성
            contains_trend = any(word in trending_keywords for word in keywords)
            naver_trees.append({
                'articles': [news],
                'all_keywords': set([kw for kw in keywords if kw not in stopwords]),  # 집합으로 초기화
                'contains_trend': contains_trend,
                'importance': 1
            })
            logging.info(f"네이버 새로운 트리 생성: {news['title']} (트렌드 포함: {'예' if contains_trend else '아니오'})")

    logging.info(f"네이버 트리의 개수: {len(naver_trees)}")
    logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    # 네이버 트리와 Google 트렌드 트리에서 대표 이슈 추출 (키워드 구문 생성)
    naver_trees_info = extract_representative_info(naver_trees, source='Naver', max_words=5)
    trend_trees_info = extract_representative_info(trend_trees, source='Google Trends', max_words=5)

    # 네이버 트리 내림차순 정렬 (트리의 중요도 기준)
    sorted_naver_trees_info = sorted(
        naver_trees_info,
        key=lambda x: -x['importance']
    )

    # Google 트렌드 트리 내림차순 정렬 (검색량 기준)
    sorted_trend_trees_info = sorted(
        trend_trees_info,
        key=lambda x: -x['importance']
    )

    # 상위 6개 네이버 이슈 선택
    top_naver_issues = sorted_naver_trees_info[:10]

    # 최종 이슈 리스트 생성: 네이버 상위 6개 이슈 + 조합된 구문
    final_issues = top_naver_issues + combined_phrases

    # 유사도 기반 이슈 병합 (연결 요소 기반 병합)
    final_issues = merge_similar_issues(final_issues, similarity_threshold=0.2)

    # 최종 이슈가 20개 미만일 경우, 부족한 만큼 네이버 전용에서 추가
    if len(final_issues) <= 20:
        # 네이버에서 추가
        for item in sorted_naver_trees_info[6:]:
            final_issues.append(item)
            if len(final_issues) >= 20:
                break
        # 트렌드에서 추가 (이 경우, 조합된 구문을 이미 추가했으므로 추가할 필요 없음)
        for item in combined_phrases[5:]:
            final_issues.append(item)
            if len(final_issues) >= 20:
                break

    # 최종 이슈가 10개 미만일 경우, 추가로 채우기

    # 최종 이슈 출력
    print("\n실시간 이슈:")

    # 네이버 뉴스 상위 6개 이슈 출력
    print("\n네이버 뉴스 상위 6개 이슈:")
    for rank, item in enumerate(top_naver_issues, 1):
        phrase = item['phrase']
        print(f"{rank}. {phrase}")

    # Google 트렌드 키워드 기반 네이버 뉴스 상위 3개 키워드의 조합 출력
    print("\nGoogle 트렌드 키워드 기반 네이버 뉴스 상위 3개 키워드의 조합:")
    for rank, item in enumerate(combined_phrases, 1):
        phrase = item['phrase']
        print(f"{rank}. {phrase}")

    # 전체 실시간 이슈 출력 (Google 트렌드 이슈 + 네이버 상위 6개 + 조합된 구문)
    print("\n전체 실시간 이슈 (Google 트렌드 + 네이버 상위 6개 + 조합된 구문):")
    for rank, item in enumerate(final_issues, 1):
        phrase = item['phrase']
        print(f"{rank}. {phrase}")

    # 최종 이슈를 요약하기 위한 텍스트 준비
    summary_content = ""
    for i, issue in enumerate(final_issues, 1):
        summary_content += f"{i}. {issue['phrase']}\n"

    # 키워드 요약 호출
    summarized_keywords = summarize_keywords(summary_content)
    print("\n요약된 실시간 이슈:")
    print(summarized_keywords)

    # 글로벌 실시간 이슈를 텍스트 파일로 저장
    try:
        with open('korea_real_time_issues.txt', 'w', encoding='utf-8') as f:
            f.write("한국 실시간 이슈 :\n")
            f.write(f"{summarized_keywords}\n")
        logging.info("한국 실시간 이슈가 'korea_real_time_issues.txt' 파일에 저장되었습니다.")
        print("\n한국 실시간 이슈가 'korea_real_time_issues.txt' 파일에 저장되었습니다.")
    except Exception as e:
        logging.error(f"한국 실시간 이슈 저장 중 오류 발생: {e}")
        print("\n한국 실시간 이슈 저장에 실패했습니다.")

        # **임시 GUI 화면 생성하여 이슈 출력**
    if summarized_keywords:
        root = tk.Tk()
        root.title("한국 실시간 이슈")
        root.geometry("300x400")

        label = tk.Label(root, text="한국 실시간 이슈 :", font=("Helvetica", 16, "bold"))
        label.pack(pady=10)

        # 스크롤 가능한 텍스트 위젯 사용
        text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20, font=("Helvetica", 12))
        text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        text_area.insert(tk.END, f"{summarized_keywords}\n")
        
        # 텍스트 영역을 편집 불가능하게 설정
        text_area.configure(state='disabled')

        root.mainloop()
    else:
        logging.warning("GUI에 표시할 한국 실시간 이슈가 없습니다.")

if __name__ == "__main__":
    main()
