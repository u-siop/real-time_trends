import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # FutureWarnings 무시

import os
import re
import time
import logging
from collections import Counter
from functools import wraps
import sys
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import pandas as pd

import requests

import spacy
import pytextrank

from dateutil import parser as date_parser
import datetime

from deep_translator import GoogleTranslator

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pytrends.request import TrendReq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from difflib import SequenceMatcher  # 편집 거리 계산을 위한 라이브러리

from openai import OpenAI

# OpenAI API key 설정
client = OpenAI(
)

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 개발 중에는 DEBUG 레벨로 설정

# 파일 핸들러 설정 (UTF-8 인코딩)
file_handler = logging.FileHandler("news_scraper.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 기존 핸들러 모두 제거
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# 파일 핸들러만 추가
logger.addHandler(file_handler)

# 콘솔 핸들러 설정 (선택 사항)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)  # 콘솔에는 ERROR 이상만 출력
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 포괄적인 영어 불용어 리스트
english_stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
    "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these',
    'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
    'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

# 재시도 데코레이터
def retry(exception_to_check, tries=3, delay=2, backoff=2):
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exception_to_check as e:
                    logger.warning(f"{f.__name__} 실패: {e}, 재시도 {tries - mtries + 1}/{tries}")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

# spaCy 모델 및 PyTextRank 초기화
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# newspaper3k import 및 설정
try:
    from newspaper import Article
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "newspaper3k"])
    from newspaper import Article

# 웹페이지 스크래핑 함수 (뉴스 기사 본문 추출)
@retry(Exception, tries=3, delay=2, backoff=2)
def scrape_webpage(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logger.error(f"Failed to scrape article at {url}: {e}")
        return ""

# 뉴스 RSS 피드 수집 함수
@retry(Exception, tries=3, delay=2, backoff=2)
def get_rss_news(rss_url, days=7, max_articles=20):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')

        # 현재 시간 기준으로 필터링
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

        news_list = []
        for item in items:
            if len(news_list) >= max_articles:
                break  # 최대 기사 수에 도달하면 중단

            title = item.title.text if item.title else 'No Title'
            link = item.link.text if item.link else 'No Link'
            description = item.description.text if item.description else ''

            if not link:
                continue  # 링크가 없는 경우 스킵

            parsed_url = urlparse(link)
            if parsed_url.path.startswith('/video'):
                logger.info(f"비디오 링크 스킵: {link}")
                continue  # 경로가 '/video'로 시작하면 스킵

            pub_date_str = item.pubDate.text if item.pubDate else None

            if pub_date_str:
                pub_date = date_parser.parse(pub_date_str)
                logger.debug(f"기사 제목: {title}, 발행일: {pub_date}")
                if pub_date >= cutoff_date:
                    news_list.append({'title': title, 'link': link, 'pubDate': pub_date, 'source': 'RSS Feed', 'description': description})

        logger.info(f"RSS 피드에서 필터링된 기사 수: {len(news_list)}개")
        return news_list
    except Exception as e:
        logger.error(f"RSS 뉴스 수집 중 오류 발생 ({rss_url}): {e}")
        return []

# NewsAPI를 사용하여 뉴스 수집 함수
@retry((requests.exceptions.RequestException), tries=3, delay=2, backoff=2)
def get_newsapi_news(api_key, country='us', category=None, max_articles=100):
    url = 'https://newsapi.org/v2/top-headlines'
    params = {
        'apiKey': api_key,
        'country': country,
        'pageSize': max_articles,
    }
    if category:
        params['category'] = category
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data['status'] != 'ok':
        logger.error(f"NewsAPI 오류: {data.get('message', 'Unknown error')}")
        return []
    articles = data.get('articles', [])
    news_list = []
    for article in articles:
        title = article.get('title', 'No Title')
        link = article.get('url', '')
        description = article.get('description', '')
        if not link:
            continue
        pub_date_str = article.get('publishedAt', None)
        pub_date = date_parser.parse(pub_date_str) if pub_date_str else datetime.datetime.now(datetime.timezone.utc)
        source_name = article.get('source', {}).get('name', 'Unknown')
        news_list.append({
            'title': title,
            'link': link,
            'pubDate': pub_date,
            'source': source_name,
            'description': description
        })
    logger.info(f"NewsAPI에서 수집된 기사 수: {len(news_list)}개")
    return news_list

# Google News를 RSS 피드로 수집하는 함수
@retry(Exception, tries=3, delay=2, backoff=2)
def get_google_news(rss_url, days=7, max_articles=100):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(rss_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')

        # 현재 시간 기준으로 필터링
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

        news_list = []
        for item in items:
            if len(news_list) >= max_articles:
                break  # 최대 기사 수에 도달하면 중단

            title = item.title.text if item.title else 'No Title'
            link = item.link.text if item.link else 'No Link'
            description = item.description.text if item.description else ''

            if not link:
                continue  # 링크가 없는 경우 스킵

            parsed_url = urlparse(link)
            if parsed_url.path.startswith('/video'):
                logger.info(f"비디오 링크 스킵: {link}")
                continue  # 경로가 '/video'로 시작하면 스킵

            pub_date_str = item.pubDate.text if item.pubDate else None

            if pub_date_str:
                pub_date = date_parser.parse(pub_date_str)
                logger.debug(f"기사 제목: {title}, 발행일: {pub_date}")
                if pub_date >= cutoff_date:
                    news_list.append({'title': title, 'link': link, 'pubDate': pub_date, 'source': 'Google News', 'description': description})

        logger.info(f"Google News에서 필터링된 기사 수: {len(news_list)}개")
        return news_list
    except Exception as e:
        logger.error(f"Google News 수집 중 오류 발생 ({rss_url}): {e}")
        return []

# 뉴스 수집 함수 통합
def collect_news(newsapi_key, google_news_rss_urls, days=7, max_articles=100):
    all_news = []

    # NewsAPI에서 뉴스 수집
    newsapi_news = get_newsapi_news(api_key=newsapi_key, max_articles=max_articles)
    all_news.extend(newsapi_news)

    # Google News에서 뉴스 수집
    for rss_url in google_news_rss_urls:
        google_news = get_google_news(rss_url=rss_url, days=days, max_articles=max_articles)
        all_news.extend(google_news)

    logger.info(f"총 수집된 기사 수: {len(all_news)}개")
    return all_news

# 텍스트 전처리 함수
def preprocess_text(text):
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("유효하지 않은 입력 텍스트.")
        return ""
    # HTML 엔티티 제거
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # 특수 문자 제거 (영문 기준)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', ' ', text)
    # 여러 개의 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 의미 있는 구문인지 확인하는 함수
def is_meaningful_phrase(phrase):
    # 적어도 하나의 알파벳 문자가 포함되어 있고, 숫자만으로 이루어지지 않았는지 확인
    return bool(re.search(r'[a-zA-Z]', phrase)) and not phrase.strip().isdigit()

# Jaccard 유사도 계산 함수
def calculate_jaccard_similarity(keywords1, keywords2):
    set1 = set(keywords1)
    set2 = set(keywords2)
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)

# 코사인 유사도 계산 함수 (추가된 검증 포함)
def calculate_cosine_similarity(phrase1, phrase2):
    if not phrase1.strip() or not phrase2.strip():
        logger.debug("One or both phrases are empty after preprocessing.")
        return 0.0
    try:
        vectorizer = TfidfVectorizer().fit_transform([phrase1, phrase2])
        vectors = vectorizer.toarray()
        if not vectors.any():
            logger.debug("TF-IDF vectors are empty. Possibly only stop words.")
            return 0.0
        cosine_sim = cosine_similarity(vectors)
        return cosine_sim[0][1]
    except ValueError as ve:
        logger.error(f"ValueError in cosine similarity: {ve} | Phrase1: '{phrase1}' | Phrase2: '{phrase2}'")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error in cosine similarity: {e}")
        return 0.0

# 편집 거리 계산 함수
def calculate_edit_distance(phrase1, phrase2):
    return SequenceMatcher(None, phrase1, phrase2).ratio()

# 키워드 클러스터링 함수
def cluster_keywords(keywords, eps=0.5, min_samples=2):
    if not keywords:
        return []
    vectorizer = TfidfVectorizer().fit_transform(keywords)
    vectors = vectorizer.toarray()
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(vectors)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:
            continue  # 노이즈 제외
        if label in clusters:
            clusters[label].append(keywords[idx])
        else:
            clusters[label] = [keywords[idx]]
    # 각 클러스터에서 가장 빈도가 높은 키워드 선택
    representative_keywords = []
    for cluster in clusters.values():
        keyword_counter = Counter(cluster)
        representative_keywords.append(keyword_counter.most_common(1)[0][0])
    return representative_keywords

# 키워드 추출 함수 (TextRank 기반, English)
def extract_keywords_textrank(text, stopwords, top_n=10):
    try:
        doc = nlp(text)
        keywords = []
        for phrase in doc._.phrases:
            # 불용어가 포함된 구문 제외
            if any(word.lower() in stopwords for word in phrase.text.split()):
                continue
            keywords.append(phrase.text)
            if len(keywords) >= top_n:
                break
        return keywords
    except Exception as e:
        logger.error(f"TextRank 키워드 추출 중 오류 발생: {e}")
        return []

# 대표 키워드 선정 함수 (Google 트렌드 우선, 불용어 필터링 추가)
def select_representative_keyword(top_keywords, used_keywords, google_trends_keywords, language='en'):
    # Google 트렌드 키워드를 우선적으로 선택
    for kw in top_keywords:
        if (kw in google_trends_keywords) and (kw not in used_keywords) and (kw.lower() not in english_stopwords):
            return kw
    # 나머지 키워드 중 사용되지 않았고 불용어가 아닌 키워드 선택
    for kw in top_keywords:
        if (kw not in used_keywords) and (kw.lower() not in english_stopwords):
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

# 트리 별로 대표 이슈 추출 함수 (키워드 구문 생성)
def extract_representative_info(trees, google_trends, source='Global News', language='en', max_words=5):
    trees_info = []
    used_keywords = set()
    for tree_info in trees:
        articles = tree_info['articles']
        # 트리 중요도 계산 (뉴스 기사 수 또는 검색량)
        importance = tree_info.get('importance', len(articles))
        # 대표 키워드 선정 (트리 내 가장 많이 등장한 키워드)
        keyword_counter = Counter()
        for news in articles:
            if 'keywords' in news and news['keywords']:
                keyword_counter.update([kw for kw in news.get('keywords', []) if kw.lower() not in english_stopwords])
            else:
                continue
        top_keywords = [word for word, freq in keyword_counter.most_common(5)]
        if not top_keywords:
            continue

        # Google 트렌드 키워드를 우선적으로 선택
        source_trends = google_trends.get(tree_info.get('source', ''), [])
        rep_keyword = select_representative_keyword(top_keywords, used_keywords, source_trends, language=language)
        if not rep_keyword:
            rep_keyword = top_keywords[0] if top_keywords else None
        if not rep_keyword:
            continue  # 대표 키워드가 없으면 스킵
        used_keywords.add(rep_keyword)

        # 대표 키워드 제외 상위 키워드
        top_other_keywords = [kw for kw in top_keywords if kw != rep_keyword]
        if top_other_keywords:
            # 이슈 구문은 최대 5개의 키워드로 제한
            phrase_keywords = [rep_keyword] + top_other_keywords[:max_words-1]
            # 중복 단어 제거 및 부분 문자열 제거
            phrase_keywords = remove_substrings_and_duplicates(phrase_keywords)
            # 의미 없는 키워드 제거
            phrase_keywords = [kw for kw in phrase_keywords if re.match(r'^[0-9a-zA-Z]{2,}(?: [0-9a-zA-Z]{2,})*$', kw)]
            # 최대 단어 수 제한
            if len(phrase_keywords) > max_words:
                phrase_keywords = phrase_keywords[:max_words]
            # 조건에 맞지 않으면 대표 키워드만 사용
            if len(phrase_keywords) < 2:
                phrase_keywords = [rep_keyword]
            phrase = ', '.join(phrase_keywords)
        else:
            phrase = rep_keyword

        # 의미 있는 구문인지 확인
        if not is_meaningful_phrase(phrase):
            logger.warning(f"Skipping non-meaningful phrase: {phrase}")
            continue

        # 유사도 검사: 이미 추가된 트리에 유사한 구문이 있는지 확인
        is_similar = False
        for existing in trees_info:
            similarity = calculate_cosine_similarity(phrase.lower(), existing['phrase'].lower())
            edit_similarity = calculate_edit_distance(phrase.lower(), existing['phrase'].lower())
            if similarity >= 0.85 or edit_similarity >= 0.85:  # 코사인 유사도와 편집 거리 임계값 상향 조정
                is_similar = True
                break
            # 부분 일치 검사
            if (phrase.lower() in existing['phrase'].lower()) or (existing['phrase'].lower() in phrase.lower()):
                is_similar = True
                break
        if is_similar:
            logger.info(f"유사한 이슈 발견, 제외됨: {phrase}")
            continue

        combined_info = {
            'phrase': phrase,
            'importance': importance,
            'source': source  # 트리의 출처 추가
        }
        trees_info.append(combined_info)
        logger.info(f"Representative issue added: {phrase} - Importance: {importance} - Source: {source}")
    return trees_info

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
        logger.error(f"요약 생성 중 오류 발생: {e}")
        return "요약 생성에 실패했습니다."
    
# Google 트렌드 키워드 수집 함수 (G10 국가)
def get_google_trends_g10():
    try:
        pytrends = TrendReq(hl='en', tz=360)
        countries = {
            'united_states': 'US',
            'united_kingdom': 'GB',
            'japan': 'JP',
            'germany': 'DE',
            'brazil': 'BR',
            'france': 'FR',
            'italy': 'IT',
            'canada': 'CA',
            'russia': 'RU'
        }

        all_trends = {}

        for country_name, country_code in countries.items():
            logger.info(f"Collecting trends for {country_name} ({country_code})")
            try:
                df_trending = pytrends.trending_searches(pn=country_name)
                trending_keywords = df_trending[0].tolist()[:20]  # 상위 20개 키워드
                all_trends[country_code] = trending_keywords
                logger.info(f"Collected {len(all_trends[country_code])} trends for {country_name}")
                time.sleep(1)  # API 호출 제한을 피하기 위해 잠시 대기
            except Exception as e:
                logger.error(f"Error collecting trends for {country_name}: {e}")

        return all_trends
    except Exception as e:
        logger.error(f"Google 트렌드 수집 중 오류 발생: {e}")
        return {}

# 트리 병합 함수 (유사도 기반)
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
    # Vectorize using TF-IDF (split by comma and space)
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '), lowercase=False)
    try:
        tfidf_matrix = vectorizer.fit_transform(phrases)
    except ValueError as ve:
        logger.error(f"ValueError during TF-IDF vectorization: {ve}")
        return []

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Union-Find algorithm to find connected components
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

    # Connect issues with similarity above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= similarity_threshold:
                union(i, j)

    # Group issues by their root parent
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Merge issues within each group
    merged_issues = []
    for group in groups.values():
        if not group:
            continue
        if len(group) == 1:
            # No need to merge
            merged_issues.append(issues[group[0]])
        else:
            # Combine phrases and sum importance
            combined_phrases = [issues[idx]['phrase'] for idx in group]
            combined_importance = sum([issues[idx].get('importance', 0) for idx in group])
            combined_sources = set([issues[idx].get('source', '') for idx in group])

            # Remove duplicates and substrings
            words = set(', '.join(combined_phrases).split(', '))
            words = remove_substrings_and_duplicates(words)
            # Remove meaningless keywords
            words = [kw for kw in words if re.match(r'^[0-9a-zA-Z]{2,}(?: [0-9a-zA-Z]{2,})*$', kw)]
            if not words:
                continue
            combined_phrases_cleaned = ', '.join(sorted(words))
            merged_issues.append({
                'phrase': combined_phrases_cleaned,
                'importance': combined_importance,
                'source': ', '.join(combined_sources)
            })
            logger.info(f"이슈 병합: {[issues[idx]['phrase'] for idx in group]} -> {combined_phrases_cleaned}")

    return merged_issues

# 메인 함수
def main():
    print("스크립트가 시작되었습니다.")
    logger.info("스크립트가 시작되었습니다.")

    # Google News RSS 피드 URL 설정
    google_news_rss_urls = [
        "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",  # 미국 영어 뉴스
        "https://news.google.com/rss?hl=en-US&gl=GB&ceid=GB:en",  # 영국 영어 뉴스
        # 필요한 경우 추가 RSS 피드 URL을 여기에 추가
    ]

    # 뉴스 소스별 RSS 피드 URL (글로벌 뉴스 포함)
    news_sources = {
        'BBC': "http://feeds.bbci.co.uk/news/rss.xml",
        'CNN': "http://rss.cnn.com/rss/edition.rss",
        'FOX NEWS': "http://feeds.foxnews.com/foxnews/latest",
        'NHK': "https://www3.nhk.or.jp/rss/news/cat0.xml",
        'New York Times': "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        'Reuters': "http://feeds.reuters.com/reuters/topNews",
        'Al Jazeera': "https://www.aljazeera.com/xml/rss/all.xml",
        'The Guardian': "https://www.theguardian.com/world/rss",
        'Bloomberg': "https://www.bloomberg.com/feed/podcast/top-news.xml",
        'Financial Times': "https://www.ft.com/?format=rss",
        'ABC News': "https://abcnews.go.com/abcnews/topstories",
        'CBS News': "https://www.cbsnews.com/latest/rss/main",
        'The Washington Post': "https://feeds.washingtonpost.com/rss/world",
        'CCTV': "http://news.cctv.com/rss/world.xml"
    }

    # 뉴스 소스별 뉴스 수집
    all_news = {}
    for source, rss_url in news_sources.items():
        logger.info(f"Collecting news from {source}")
        news = get_rss_news(rss_url, days=7, max_articles=20)  # 최근 7일 이내의 기사만 수집, 최대 20개
        all_news[source] = news
        logger.info(f"{source} 뉴스 수집 완료: {len(news)}개")

    if not all_news:
        logger.error("수집된 기사가 없습니다. 프로그램을 종료합니다.")
        return

    # Google 트렌드 키워드 수집 (G10 국가)
    google_trends = get_google_trends_g10()
    logger.info("Google 트렌드 수집 완료")

    # NewsAPI 및 Google News RSS를 통해 추가 뉴스 수집
    additional_news = collect_news(newsapi_key, google_news_rss_urls, days=7, max_articles=100)
    if additional_news:
        # 추가 뉴스를 기존 all_news에 통합
        for news in additional_news:
            source = news.get('source', 'NewsAPI/Google News')
            if source not in all_news:
                all_news[source] = []
            all_news[source].append(news)
        logger.info(f"추가 뉴스 수집 완료: {len(additional_news)}개")
    else:
        logger.warning("추가 뉴스가 수집되지 않았습니다.")

    # 기사별 텍스트 수집 (병렬 처리)
    articles_texts = []
    articles_metadata = []

    # 병렬 처리를 위한 리스트 준비
    articles_to_fetch = []
    for source, source_news in all_news.items():
        for news in source_news:
            articles_to_fetch.append(news)

    logger.info(f"총 크롤링할 기사 수: {len(articles_to_fetch)}개")

    # ThreadPoolExecutor를 사용하여 병렬로 기사 텍스트 수집
    with ThreadPoolExecutor(max_workers=12) as executor:
        # Future 객체 리스트 생성
        future_to_article = {
            executor.submit(scrape_webpage, news['link']): news
            for news in articles_to_fetch
        }

        # tqdm을 사용하여 진행 표시기 추가
        for future in tqdm(as_completed(future_to_article), total=len(future_to_article), desc="Fetching articles"):
            news = future_to_article[future]
            try:
                text = future.result()
                # 텍스트가 없으면 description을 대체로 사용 (HTML 태그 제거)
                if not text and 'description' in news and news['description']:
                    soup = BeautifulSoup(news['description'], 'html.parser')
                    text = soup.get_text()
                if text:
                    language = 'en'  # 모든 뉴스가 영어로 가정
                    full_text = preprocess_text(news['title'] + ' ' + text)
                    if full_text:
                        articles_texts.append(full_text)
                        articles_metadata.append({
                            'title': news['title'],
                            'link': news['link'],
                            'source': news['source'],
                            'pubDate': news['pubDate'],
                            'language': language,
                            'keywords': []
                        })
                        logger.info(f"기사 본문 추출 성공: {news['title']} (발행일: {news['pubDate']})")
                    else:
                        logger.warning(f"전처리된 본문이 없습니다: {news['title']}")
                else:
                    logger.warning(f"기사 본문이 없습니다: {news['title']}")
            except Exception as e:
                logger.error(f"기사 크롤링 중 오류 발생: {news['title']} - {e}")

    logger.info(f"수집된 유효한 기사 수: {len(articles_texts)}개")

    if not articles_texts:
        logger.error("유효한 기사가 하나도 없습니다. 프로그램을 종료합니다.")
        return

    # 키워드 추출
    articles_keywords_list = []
    for idx, text in enumerate(tqdm(articles_texts, desc="Extracting keywords")):
        keywords = extract_keywords_textrank(text, english_stopwords, top_n=10)
        articles_keywords_list.append(keywords)
        # 각 뉴스 기사에 키워드 추가
        articles_metadata[idx]['keywords'] = keywords

    # 트리 및 뉴스 정보 저장
    global_news_trees = []  # 글로벌 뉴스 트리 정보의 리스트

    # 글로벌 뉴스 기사별로 트리 생성
    for idx, news in enumerate(articles_metadata):
        keywords = articles_keywords_list[idx]
        if not keywords:
            continue
        merged = False
        # 기존 글로벌 트리들과 비교하여 유사하면 병합
        for tree_info in global_news_trees:
            similarity = calculate_jaccard_similarity(
                keywords, 
                tree_info['all_keywords']
            )
            if similarity >= 0.2:  # 유사도 임계값
                # 트리에 뉴스 추가
                tree_info['articles'].append(news)
                tree_info['all_keywords'].update([kw for kw in keywords if kw.lower() not in english_stopwords])  # 집합으로 업데이트
                # 트리의 중요도 업데이트 (기사 수 또는 검색량)
                tree_info['importance'] += 1
                merged = True
                logger.info(f"글로벌 트리 병합 완료: {news['title']} (유사도: {similarity:.2f})")
                break
        if not merged:
            # 새로운 글로벌 트리 생성
            global_news_trees.append({
                'articles': [news],
                'all_keywords': set([kw for kw in keywords if kw.lower() not in english_stopwords]),  # 집합으로 초기화
                'importance': 1  # 기사 수 초기화
            })
            logger.info(f"글로벌 새로운 트리 생성: {news['title']}")

    logger.info(f"글로벌 트리의 개수: {len(global_news_trees)}")

    # Google 트렌드 트리 생성 (중요도 수정)
    trend_trees = []
    for country, trends in google_trends.items():
        for idx, keyword in enumerate(trends[:10]):  # 각 국가에서 상위 10개 키워드만
            if keyword.lower() in english_stopwords:
                continue
            # 키워드 전처리 및 추출
            full_text = preprocess_text(keyword)
            keywords = extract_keywords_textrank(full_text, english_stopwords, top_n=10)
            if not keywords:
                continue
            # 중요도를 키워드의 순위에 따라 설정 (순위가 높을수록 중요도 높음)
            importance = 10 - idx  # idx는 0부터 시작하므로, 중요도는 10부터 1까지 할당
            # 트렌드 트리 생성
            trend_trees.append({
                'articles': [{
                    'title': keyword,
                    'link': f'https://trends.google.com/trends/trendingsearches/daily?geo={country}',  # 국가 코드에 맞는 트렌드 링크
                    'keywords': keywords,
                    'source': country  # 트렌드의 국가 정보 추가
                }],
                'all_keywords': set([kw for kw in keywords if kw.lower() not in english_stopwords]),
                'importance': importance  # 중요도를 키워드 순위로 설정
            })
            logger.info(f"Google 트렌드 새로운 트리 생성: {keyword} (국가: {country}, 중요도: {importance})")

    logger.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    # 글로벌 뉴스 트리에서 대표 이슈 추출 (키워드 구문 생성)
    global_trees_info = extract_representative_info(
        global_news_trees, 
        google_trends, 
        source='Global News', 
        language='en', 
        max_words=5
    )

    # Google 트렌드 트리에서 대표 이슈 추출 (키워드 구문 생성)
    trend_trees_info = extract_representative_info(
        trend_trees, 
        google_trends, 
        source='Google Trends', 
        language='en', 
        max_words=5
    )

    # 글로벌 뉴스 트리 내림차순 정렬 (트리의 중요도 기준)
    sorted_global_trees_info = sorted(
        global_trees_info,
        key=lambda x: -x['importance']
    )

    # Google 트렌드 트리 내림차순 정렬 (검색량 기준)
    sorted_trend_trees_info = sorted(
        trend_trees_info,
        key=lambda x: -x['importance']
    )

    # 상위 20개 글로벌 뉴스 이슈 선택
    top_global_issues = sorted_global_trees_info[:20]
    print("Top Global Issues:")
    for issue in top_global_issues:
        print(issue)
    print("\n")

    # 상위 20개 트렌드 이슈 선택
    top_trend_issues = sorted_trend_trees_info[:20]
    print("Top Trend Issues:")
    for issue in top_trend_issues:
        print(issue)
    print("\n")

    # 중복 이슈 제거 및 최종 이슈 리스트 생성
    final_issues = []
    seen_phrases = set()

    for issue in top_global_issues + top_trend_issues:
        phrase = issue['phrase']
        if phrase not in seen_phrases:
            final_issues.append(issue)
            seen_phrases.add(phrase)
        if len(final_issues) >= 20:
            break

    # 최종 이슈가 20개 미만일 경우, 부족한 만큼 글로벌 뉴스 전용이나 트렌드 전용에서 추가
    if len(final_issues) < 20:
        # 글로벌 뉴스에서 추가
        for item in sorted_global_trees_info[20:]:
            if item['phrase'] not in seen_phrases:
                final_issues.append(item)
                seen_phrases.add(item['phrase'])
                if len(final_issues) >= 20:
                    break
        # 트렌드에서 추가
        for item in sorted_trend_trees_info[20:]:
            if item['phrase'] not in seen_phrases:
                final_issues.append(item)
                seen_phrases.add(item['phrase'])
                if len(final_issues) >= 20:
                    break

    # 최종 이슈가 20개로 제한
    final_issues = final_issues[:20]

    # 추가적인 유사도 검사: 이미 선택된 이슈들 간의 유사도 확인 및 중복 제거
    unique_final_issues = []
    for issue in final_issues:
        is_similar = False
        for unique_issue in unique_final_issues:
            similarity = calculate_cosine_similarity(issue['phrase'].lower(), unique_issue['phrase'].lower())
            edit_similarity = calculate_edit_distance(issue['phrase'].lower(), unique_issue['phrase'].lower())
            if similarity >= 0.85 or edit_similarity >= 0.85:
                is_similar = True
                break
        if not is_similar:
            unique_final_issues.append(issue)
        if len(unique_final_issues) >= 20:
            break

    # 최종 이슈가 20개로 제한
    unique_final_issues = unique_final_issues[:20]

    # 결과 출력
    print("\nTop 20 Real-time Multi-keywords:")
    for rank, item in enumerate(unique_final_issues, 1):
        phrase = item['phrase']
        print(f"{rank}. {phrase} (Importance: {item['importance']})")

    # 최종 이슈를 요약하기 위한 텍스트 준비
    summary_content = ""
    for i, issue in enumerate(unique_final_issues, 1):
        summary_content += f"{i}. {issue['phrase']}\n"

    # 키워드 요약 호출
    summarized_keywords = summarize_keywords(summary_content)
    print("\nSummarized Real-time Issues:")
    print(summarized_keywords)

if __name__ == "__main__":
    main()
