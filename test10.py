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

# spaCy 모델 및 PyTextRank 초기화
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # DEBUG 레벨로 설정 (개발 중에는 유용)

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

# 포괄적인 영어 불용어 리스트 (기존과 동일)
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


# 재시도 데코레이터 (기존과 동일)
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

# NewsAPI를 사용하여 뉴스 수집 함수 추가
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
        if not link:
            continue
        pub_date_str = article.get('publishedAt', None)
        pub_date = date_parser.parse(pub_date_str) if pub_date_str else datetime.datetime.now(datetime.timezone.utc)
        news_list.append({
            'title': title,
            'link': link,
            'pubDate': pub_date,
            'source': article.get('source', {}).get('name', 'Unknown')
        })
    logger.info(f"NewsAPI에서 수집된 기사 수: {len(news_list)}개")
    return news_list

# Google News를 RSS 피드로 수집하는 함수 추가
@retry((requests.exceptions.RequestException), tries=3, delay=2, backoff=2)
def get_google_news(rss_url, days=7, max_articles=100):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/130.0.6723.70 Safari/537.36',
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
                    news_list.append({'title': title, 'link': link, 'pubDate': pub_date, 'source': 'Google News'})

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

# 텍스트 전처리 함수 (기존과 동일)
def preprocess_text(text):
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("유효하지 않은 입력 텍스트.")
        return ""
    # 특수 문자 제거 (영문 기준)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 여러 개의 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 키워드 유사도 계산 함수 (기존과 동일)
def calculate_jaccard_similarity(keywords1, keywords2):
    set1 = set(keywords1)
    set2 = set(keywords2)
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)

# 추가적인 유사도 계산 함수 (코사인 유사도) (기존과 동일)
def calculate_cosine_similarity(phrase1, phrase2):
    vectorizer = TfidfVectorizer().fit_transform([phrase1, phrase2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

# 편집 거리 계산 함수 (기존과 동일)
def calculate_edit_distance(phrase1, phrase2):
    return SequenceMatcher(None, phrase1, phrase2).ratio()

# 키워드 클러스터링 함수 (기존과 동일)
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

# 키워드 추출 함수 (기존과 동일)
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

# 번역 함수 (기존과 동일)
def translate_phrase(phrase, target='en'):
    try:
        return GoogleTranslator(source='auto', target=target).translate(phrase)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return phrase

# 대표 키워드 선정 함수 (기존과 동일)
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

# 트리 별로 대표 이슈 추출 함수 (기존과 동일)
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

        # 대표 키워드 선정 시 Google 트렌드 키워드 우선
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

# 중복 키워드 제거 함수 (기존과 동일)
def remove_substrings_and_duplicates(keywords):
    unique_keywords = []
    sorted_keywords = sorted(keywords, key=lambda x: len(x), reverse=True)
    for kw in sorted_keywords:
        if kw in unique_keywords:
            continue
        if not any((kw != other and kw in other) for other in unique_keywords):
            unique_keywords.append(kw)
    return unique_keywords

# 키워드 요약 함수 (수정됨)
def summarize_keywords(content):
    prompt = f"""
[요약하는 방법]
1. 다음 여러 개의 키워드를 보고 3~5어절의 키워드 10개로 요약해줘
2. 여러가지의 키워드가 합쳐져 있으면 두 개의 키워드로 분리해도 돼
예시 ) 국정 감사, 국회 운영, 대통령 관저, 대통령 다혜, 대통령 명태, 명태균, 문재인 대통령, 여론 조사, 윤석열 대통령, 정진석 대통령, 참고인 조사
--> 1. 국정 감사 및 여론
    2. 문재인 전 대통령, 다혜, 정진석

3. 같은 문맥 키워드의 내용은 합쳐줘
4. 핵심 키워드는 항상 있어야 해
예시 ) 5. 소말리, 소녀상 모욕, 편의점 난동, 조니 말리
-->  소말리, 소녀상 모욕 및 편의점 난동
예시 ) 불법 영업, 사생활 논란, 음식점 운영, 트리플 스타, 트리플star, 흑백 요리사, 유비빔
-->  흑백 요리사 유비빔, 불법 영업 논란

[예시]
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

1. 대통령 직무 평가
2. 흑백 요리사 유비빔, 불법 영업 논란
3. 국정 감사 및 여론
4. 아버지 둔기로 살해
5. 소녀상 모욕 사건
6. 23기 정숙, 출연자 검증 논란
7. GD와 베이비 몬스터
8. 기아타이거즈 세일
9. 테슬라, 김예지
10. 북한군 교전 주장

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
    
def scrape_webpage(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/130.0.6723.70 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'  # 영어 지원
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # 본문 추출 로직 (사이트별로 업데이트 필요)
    article_text = ""
    if 'bbc.co.uk' in url or 'bbc.com' in url:
        # BBC 뉴스 본문 추출 로직
        text_blocks = soup.find_all('div', {'data-component': 'text-block'})
        paragraphs = []
        for block in text_blocks:
            p_tags = block.find_all('p')
            for p in p_tags:
                for a in p.find_all('a'):
                    a.replace_with(a.get_text())
                paragraphs.append(p.get_text(separator=' ', strip=True))
        if paragraphs:
            article_text = ' '.join(paragraphs)
    elif 'cnn.com' in url:
        # CNN 뉴스 본문 추출 로직
        article_body = soup.find('div', class_='article__content')
        if article_body:
            for ad in article_body.find_all(['div', 'aside'], class_=re.compile('ad|related-content')):
                ad.decompose()
            paragraphs = article_body.find_all('p', class_=re.compile('paragraph'))
            if not paragraphs:
                paragraphs = article_body.find_all('p', class_=re.compile('vossi-paragraph'))
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs]) if paragraphs else ""
    elif 'foxnews.com' in url:
        # Fox News 본문 추출 로직
        article_body = soup.select_one('.article-body, .content-body')
        if article_body:
            article_text = article_body.get_text(separator=' ', strip=True)
    elif 'nytimes.com' in url:
        # New York Times 본문 추출 로직
        article_body = soup.select_one('section[name="articleBody"], article')
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs]) if paragraphs else ""
    elif 'reuters.com' in url:
        # Reuters 뉴스 본문 추출 로직
        article_body = soup.find('div', class_='StandardArticleBody_body')
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs]) if paragraphs else ""
    elif 'aljazeera.com' in url:
        # Al Jazeera 뉴스 본문 추출 로직
        article_body = soup.find('div', class_=re.compile('wysiwyg.*'))
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs]) if paragraphs else ""
    elif 'theguardian.com' in url:
        # The Guardian 뉴스 본문 추출 로직
        article_body = soup.find('div', class_=re.compile('article-body-commercial-selector|article-body-viewer-selector|content__article-body'))
        if article_body:
            paragraphs = article_body.find_all('p', class_=re.compile('dcr-1eu361v'))
            if not paragraphs:
                paragraphs = article_body.find_all('p')  # 클래스명이 없을 경우 모든 <p> 태그 추출
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs]) if paragraphs else ""
    elif 'abcnews.go.com' in url:
        # ABC News 본문 추출 로직
        article_body = soup.find('div', attrs={'data-testid': 'prism-article-body'})
        if article_body:
            paragraphs = article_body.find_all('p', class_=re.compile('EkqkG IGXmU.*'))
            if not paragraphs:
                paragraphs = article_body.find_all('p')  # 클래스명이 없을 경우 모든 <p> 태그 추출
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs]) if paragraphs else ""
    elif 'cbsnews.com' in url:
        # CBS News 본문 추출 로직
        article_body = soup.find('section', class_='content__body')
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs]) if paragraphs else ""
    elif 'washingtonpost.com' in url:
        # Washington Post 본문 추출 로직
        article_body = soup.find('div', class_='meteredContent grid-center')
        if article_body:
            paragraphs_divs = article_body.find_all('div', class_='wpds-c-PJLV article-body', attrs={'data-qa': 'article-body'})
            paragraphs = []
            for div in paragraphs_divs:
                p_tags = div.find_all('p')
                for p in p_tags:
                    paragraphs.append(p.get_text(separator=' ', strip=True))
            article_text = ' '.join(paragraphs) if paragraphs else ""
    elif 'bloomberg.com' in url:
        # Bloomberg 뉴스 본문 추출 로직
        article_body = soup.find('div', class_=re.compile('body-copy|article-body'))
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs]) if paragraphs else ""
    elif 'ft.com' in url:
        # Financial Times 뉴스 본문 추출 로직
        article_body = soup.find('div', class_=re.compile('article-body|body'))
        if article_body:
            paragraphs = article_body.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs]) if paragraphs else ""
    else:
        logger.warning(f"지원되지 않는 뉴스 소스: {url}")
        return ""

    if article_text:
        logger.debug(f"추출된 본문: {article_text[:100]}...")  # 본문 일부 로그 출력
        return article_text
    else:
        logger.warning(f"본문을 찾을 수 없습니다: {url}")
        return ""

    
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
            logging.info(f"Collecting trends for {country_name} ({country_code})")
            try:
                df_trending = pytrends.trending_searches(pn=country_name)
                trending_keywords = df_trending[0].tolist()[:20]  # 상위 20개 키워드
                all_trends[country_code] = trending_keywords
                logging.info(f"{country_name} trends collected: {len(all_trends[country_code])} keywords")
                time.sleep(1)  # API 호출 제한을 피하기 위해 잠시 대기
            except Exception as e:
                logging.error(f"{country_name} 트렌드 수집 중 오류 발생: {e}")

        return all_trends
    except Exception as e:
        logging.error(f"Google 트렌드 수집 중 오류 발생: {e}")
        return {}

# 메인 함수 (수정됨)
def main():
    print("스크립트가 시작되었습니다.")
    logger.info("스크립트가 시작되었습니다.")

    # NewsAPI 키 가져오기
    newsapi_key = "ee68d0e9110443c79a942f0c294abd9c"
    if not newsapi_key:
        logger.error("NewsAPI 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return

    # Google News RSS 피드 URL 설정
    # 원하는 지역이나 카테고리에 따라 RSS 피드를 추가하세요
    google_news_rss_urls = [
        "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",  # 미국 영어 뉴스
        "https://news.google.com/rss?hl=en-US&gl=GB&ceid=GB:en",  # 영국 영어 뉴스
        # 필요한 경우 추가 RSS 피드 URL을 여기에 추가
    ]

    # Google 트렌드 키워드 수집 (G10 국가)
    google_trends = get_google_trends_g10()
    logger.info("Google 트렌드 수집 완료")

    # 뉴스 수집 (NewsAPI + Google News)
    all_news = collect_news(newsapi_key, google_news_rss_urls, days=7, max_articles=100)

    if not all_news:
        logger.error("수집된 기사가 없습니다. 프로그램을 종료합니다.")
        return

    # 기사별 텍스트 수집 (병렬 처리)
    articles_texts = []
    articles_metadata = []

    # 병렬 처리를 위한 리스트 준비
    articles_to_fetch = []
    for news in all_news:
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
                            'language': language
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

    # Google 트렌드 트리 생성
    trend_trees = []
    for country, trends in google_trends.items():
        for keyword in trends[:10]:  # 각 국가에서 상위 10개 키워드만
            if keyword.lower() in english_stopwords:
                continue
            # 키워드 전처리 및 추출
            full_text = preprocess_text(keyword)
            keywords = extract_keywords_textrank(full_text, english_stopwords, top_n=10)
            if not keywords:
                continue
            # 트렌드 트리 생성
            trend_trees.append({
                'articles': [{
                    'title': keyword,
                    'link': 'https://trends.google.com/trends/trendingsearches/daily?geo=US',  # 예시: US 트렌드 링크
                    'keywords': keywords,
                    'source': country  # 트렌드의 국가 정보 추가
                }],
                'all_keywords': set([kw for kw in keywords if kw.lower() not in english_stopwords]),
                'importance': 3  # 검색량을 중요도로 설정 (트렌드 키워드에 가중치 부여)
            })
            logger.info(f"Google 트렌드 새로운 트리 생성: {keyword} (국가: {country})")

    logger.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    # 글로벌 뉴스 트리에서 대표 이슈 추출 (키워드 구문 생성)
    global_trees_info = extract_representative_info(global_news_trees, google_trends, source='Global News', language='en', max_words=5)

    # Google 트렌드 트리에서 대표 이슈 추출 (키워드 구문 생성)
    trend_trees_info = extract_representative_info(trend_trees, google_trends, source='Google Trends', language='en', max_words=5)

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

    # 상위 6개 글로벌 뉴스 이슈 선택
    top_global_issues = sorted_global_trees_info[:6]

    # 상위 4개 트렌드 이슈 선택
    top_trend_issues = sorted_trend_trees_info[:4]

    # 중복 이슈 제거 및 최종 이슈 리스트 생성
    final_issues = []
    seen_phrases = set()

    for issue in top_global_issues + top_trend_issues:
        phrase = issue['phrase']
        if phrase not in seen_phrases:
            final_issues.append(issue)
            seen_phrases.add(phrase)
        if len(final_issues) >= 10:
            break

    # 최종 이슈가 10개 미만일 경우, 부족한 만큼 글로벌 뉴스 전용이나 트렌드 전용에서 추가
    if len(final_issues) < 10:
        # 글로벌 뉴스에서 추가
        for item in sorted_global_trees_info[6:]:
            if item['phrase'] not in seen_phrases:
                final_issues.append(item)
                seen_phrases.add(item['phrase'])
                if len(final_issues) >= 10:
                    break
        # 트렌드에서 추가
        for item in sorted_trend_trees_info[4:]:
            if item['phrase'] not in seen_phrases:
                final_issues.append(item)
                seen_phrases.add(item['phrase'])
                if len(final_issues) >= 10:
                    break

    # 최종 이슈가 10개 미만일 경우, 추가로 채우기
    final_issues = final_issues[:10]

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
        if len(unique_final_issues) >= 10:
            break

    # 최종 이슈가 10개 미만일 경우, 추가로 채우기
    unique_final_issues = unique_final_issues[:10]

    # 결과 출력
    print("\n실시간 상위 10개 멀티 키워드:")

    for rank, item in enumerate(unique_final_issues, 1):
        phrase = item['phrase']
        print(f"{rank}. {phrase} (중요도: {item['importance']})")

    # 최종 이슈를 요약하기 위한 텍스트 준비
    summary_content = ""
    for i, issue in enumerate(unique_final_issues, 1):
        summary_content += f"{i}. {issue['phrase']}\n"

    # 키워드 요약 호출
    summarized_keywords = summarize_keywords(summary_content)
    print("\n요약된 실시간 이슈:")
    print(summarized_keywords)

if __name__ == "__main__":
    main()
