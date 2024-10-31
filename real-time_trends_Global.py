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

from googletrans import Translator

import spacy
import pytextrank

from dateutil import parser as date_parser
import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pytrends.request import TrendReq

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from difflib import SequenceMatcher  # 편집 거리 계산을 위한 라이브러리

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
logger.setLevel(logging.INFO)  # INFO 레벨로 설정

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

# 구글 번역기 초기화
translator = Translator()

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
                    logging.warning(f"{f.__name__} 실패: {e}, 재시도 {tries - mtries + 1}/{tries}")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

# 웹페이지 스크래핑 함수 (뉴스 기사 본문 추출) - Selenium 제거 및 Requests + BeautifulSoup 사용
@retry((requests.exceptions.RequestException), tries=3, delay=2, backoff=2)
def scrape_webpage(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/130.0.6723.70 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9'
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
        elif 'nhk.or.jp' in url:
            # NHK 뉴스 본문 추출 로직
            content_detail_body = soup.find('div', class_='content--detail-body')
            if content_detail_body:
                summary = content_detail_body.find('p', class_='content--summary')
                if summary:
                    article_text += summary.get_text(separator=' ', strip=True) + ' '
                detail_more = content_detail_body.find('div', class_='content--detail-more')
                if detail_more:
                    sections = detail_more.find_all('section', class_='content--body')
                    for section in sections:
                        body_text = section.find('div', class_='body-text')
                        if body_text:
                            p_tags = body_text.find_all('p')
                            for p in p_tags:
                                article_text += p.get_text(separator=' ', strip=True) + ' '
            article_text = article_text.strip()
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
            logging.warning(f"지원되지 않는 뉴스 소스: {url}")
            return ""

        if article_text:
            logging.debug(f"추출된 본문: {article_text[:100]}...")  # 본문 일부 로그 출력
            return article_text
        else:
            logging.warning(f"본문을 찾을 수 없습니다: {url}")
            return ""

# 뉴스 RSS 피드 수집 함수
@retry(Exception, tries=3, delay=2, backoff=2)
def get_rss_news(rss_url, days=7, max_articles=20):
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
                logging.info(f"비디오 링크 스킵: {link}")
                continue  # 경로가 '/video'로 시작하면 스킵

            pub_date_str = item.pubDate.text if item.pubDate else None

            if pub_date_str:
                pub_date = date_parser.parse(pub_date_str)
                logging.debug(f"기사 제목: {title}, 발행일: {pub_date}")
                if pub_date >= cutoff_date:
                    news_list.append({'title': title, 'link': link, 'pubDate': pub_date})

        logging.info(f"RSS 피드에서 필터링된 기사 수: {len(news_list)}개")
        return news_list
    except Exception as e:
        logging.error(f"RSS 뉴스 수집 중 오류 발생 ({rss_url}): {e}")
        return []

# Google 트렌드 키워드 수집 함수 (G10 국가)
def get_google_trends_g10():
    try:
        pytrends = TrendReq(hl='en', tz=360)
        countries = {
            'united_states': 'US',
            'united_kingdom': 'GB',
            'japan': 'JP',
            # 'china': 'CN',  # pytrends에서 지원하지 않으므로 제외
            'germany': 'DE',
            'brazil': 'BR',
            'france': 'FR',
            'italy': 'IT',
            'canada': 'CA',
            'russia': 'RU'
        }

        all_trends = {}

        for country_name in countries.items():
            logging.info(f"Collecting trends for {country_name}")
            try:
                df_trending = pytrends.trending_searches(pn=country_name)
                trending_keywords = df_trending[0].tolist()[:20]  # 상위 20개 키워드
                all_trends[country_name] = trending_keywords
                logging.info(f"{country_name} trends collected: {len(all_trends[country_name])} keywords")
                time.sleep(1)  # API 호출 제한을 피하기 위해 잠시 대기
            except Exception as e:
                logging.error(f"{country_name} 트렌드 수집 중 오류 발생: {e}")

        return all_trends
    except Exception as e:
        logging.error(f"Google 트렌드 수집 중 오류 발생: {e}")
        return {}

# 텍스트 전처리 함수
def preprocess_text(text):
    if not text or not isinstance(text, str) or not text.strip():
        logging.warning("유효하지 않은 입력 텍스트.")
        return ""
    # 특수 문자 제거 (영문 기준)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 여러 개의 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 키워드 유사도 계산 함수 (Jaccard 유사도)
def calculate_jaccard_similarity(keywords1, keywords2):
    set1 = set(keywords1)
    set2 = set(keywords2)
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)

# 추가적인 유사도 계산 함수 (코사인 유사도)
def calculate_cosine_similarity(phrase1, phrase2):
    vectorizer = TfidfVectorizer().fit_transform([phrase1, phrase2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

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

# 대표 키워드 선정 함수 (Google 트렌드 우선, 불용어 필터링 추가)
def select_representative_keyword(top_keywords, used_keywords, google_trends_keywords):
    for kw in top_keywords:
        if (kw in google_trends_keywords) and (kw not in used_keywords) and (kw.lower() not in english_stopwords):
            return kw
    for kw in top_keywords:
        if (kw not in used_keywords) and (kw.lower() not in english_stopwords):
            return kw
    return None

# 트리 별로 대표 이슈 추출 함수 (키워드 구문 생성)
def extract_representative_info(trees, google_trends_g10, source='Global News'):
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
        source_trends = google_trends_g10.get(tree_info.get('source', ''), [])
        rep_keyword = select_representative_keyword(top_keywords, used_keywords, source_trends)
        if not rep_keyword:
            rep_keyword = top_keywords[0] if top_keywords else None
        if not rep_keyword:
            continue  # 대표 키워드가 없으면 스킵
        used_keywords.add(rep_keyword)

        # 대표 키워드 제외 상위 키워드
        top_other_keywords = [kw for kw in top_keywords if kw != rep_keyword]
        if top_other_keywords:
            # 이슈 구문은 2~3단어로 제한
            phrase = f"{rep_keyword} {' '.join(top_other_keywords)}"
            phrase_words = phrase.split()
            if len(phrase_words) > 3:
                phrase = ' '.join(phrase_words[:3])
            # 중복 단어 제거
            unique_phrase_words = []
            for word in phrase.split():
                if word not in unique_phrase_words:
                    unique_phrase_words.append(word)
            phrase = ' '.join(unique_phrase_words)
            # 최종 이슈 구문이 2~3단어인지 다시 확인
            if not (2 <= len(phrase.split()) <= 3):
                # 조건에 맞지 않으면 대표 키워드만 사용
                phrase = rep_keyword
        else:
            phrase = rep_keyword

        # 번역된 구문
        translated_phrase = translate_phrase(phrase)

        # 유사도 검사: 이미 추가된 트리에 유사한 구문이 있는지 확인
        is_similar = False
        for existing in trees_info:
            similarity = calculate_cosine_similarity(translated_phrase.lower(), existing['phrase'].lower())
            edit_similarity = calculate_edit_distance(translated_phrase.lower(), existing['phrase'].lower())
            if similarity >= 0.85 or edit_similarity >= 0.85:  # 코사인 유사도와 편집 거리 임계값 상향 조정
                is_similar = True
                break
            # 부분 일치 검사: 하나의 구문이 다른 구문의 부분 문자열인지 확인
            if (translated_phrase.lower() in existing['phrase'].lower()) or (existing['phrase'].lower() in translated_phrase.lower()):
                is_similar = True
                break
        if is_similar:
            logging.info(f"유사한 이슈 발견, 제외됨: {translated_phrase}")
            continue

        combined_info = {
            'phrase': translated_phrase,
            'importance': importance,
            'source': source  # 트리의 출처 추가
        }
        trees_info.append(combined_info)
        logging.info(f"Representative issue added: {translated_phrase} - Importance: {importance} - Source: {source}")
    return trees_info

# 키워드 추출 함수 (TextRank 기반)
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
        logging.error(f"TextRank 키워드 추출 중 오류 발생: {e}")
        return []

# 번역 함수
def translate_phrase(phrase, src='auto', dest='en'):
    try:
        translation = translator.translate(phrase, src=src, dest=dest)
        return translation.text
    except Exception as e:
        logging.error(f"번역 중 오류 발생: {e}")
        return phrase  # 번역 실패 시 원문 유지

# 메인 함수
def main():
    print("스크립트가 시작되었습니다.")
    logging.info("스크립트가 시작되었습니다.")

    # Google 트렌드 키워드 수집 (G10 국가)
    google_trends_g10 = get_google_trends_g10()
    logging.info("G10 국가 구글 트렌드 수집 완료")

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
        'CCTV': "http://news.cctv.com/rss/world.xml"  # 업데이트된 CCTV RSS 피드 URL
    }

    # 뉴스 소스별 뉴스 수집
    all_news = {}
    for source, rss_url in news_sources.items():
        logging.info(f"Collecting news from {source}")
        news = get_rss_news(rss_url, days=7, max_articles=20)  # 최근 7일 이내의 기사만 수집, 최대 20개
        all_news[source] = news
        logging.info(f"{source} 뉴스 수집 완료: {len(news)}개")

    # 기사별 텍스트 수집 (병렬 처리)
    articles_texts = []
    articles_metadata = []

    # 병렬 처리를 위한 리스트 준비
    articles_to_fetch = []
    for source, source_news in all_news.items():
        for news in source_news:
            articles_to_fetch.append((news, source))

    logging.info(f"총 크롤링할 기사 수: {len(articles_to_fetch)}개")

    # ThreadPoolExecutor를 사용하여 병렬로 기사 텍스트 수집
    with ThreadPoolExecutor(max_workers=12) as executor:
        # Future 객체 리스트 생성
        future_to_article = {
            executor.submit(scrape_webpage, news['link']): (news, source)
            for news, source in articles_to_fetch
        }

        # tqdm을 사용하여 진행 표시기 추가
        for future in tqdm(as_completed(future_to_article), total=len(future_to_article), desc="Fetching articles"):
            news, source = future_to_article[future]
            try:
                text = future.result()
                if text:
                    full_text = preprocess_text(news['title'] + ' ' + text)
                    if full_text:
                        articles_texts.append(full_text)
                        articles_metadata.append({
                            'title': news['title'],
                            'link': news['link'],
                            'source': source,
                            'pubDate': news['pubDate']
                        })
                        logging.info(f"기사 본문 추출 성공: {news['title']} (발행일: {news['pubDate']})")
                    else:
                        logging.warning(f"전처리된 본문이 없습니다: {news['title']}")
                else:
                    logging.warning(f"기사 본문이 없습니다: {news['title']}")
            except Exception as e:
                logging.error(f"기사 크롤링 중 오류 발생: {news['title']} - {e}")

    logging.info(f"수집된 유효한 기사 수: {len(articles_texts)}개")

    if not articles_texts:
        logging.error("유효한 기사가 하나도 없습니다. 프로그램을 종료합니다.")
        return

    # TextRank 기반 키워드 추출
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
            if similarity >= 0.3:  # 유사도 임계값
                # 트리에 뉴스 추가
                tree_info['articles'].append(news)
                tree_info['all_keywords'].update([kw for kw in keywords if kw.lower() not in english_stopwords])  # 집합으로 업데이트
                # 트리의 중요도 업데이트 (기사 수)
                tree_info['importance'] += 1
                merged = True
                logging.info(f"글로벌 트리 병합 완료: {news['title']} (유사도: {similarity:.2f})")
                break
        if not merged:
            # 새로운 글로벌 트리 생성
            global_news_trees.append({
                'articles': [news],
                'all_keywords': set([kw for kw in keywords if kw.lower() not in english_stopwords]),  # 집합으로 초기화
                'importance': 1  # 기사 수 초기화
            })
            logging.info(f"글로벌 새로운 트리 생성: {news['title']}")

    logging.info(f"글로벌 트리의 개수: {len(global_news_trees)}")

    # Google 트렌드 트리 생성
    trend_trees = []
    for country, trends in google_trends_g10.items():
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
                    'link': 'https://trends.google.com/trends/trendingsearches/daily?geo=KR',
                    'keywords': keywords,
                    'source': country  # 트렌드의 국가 정보 추가
                }],
                'all_keywords': set([kw for kw in keywords if kw.lower() not in english_stopwords]),
                'importance': 3  # 검색량을 중요도로 설정 (트렌드 키워드에 가중치 부여)
            })
            logging.info(f"Google 트렌드 새로운 트리 생성: {keyword} (국가: {country})")

    logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    # Google 트렌드 트리와 글로벌 뉴스 트리를 별도로 관리하여 대표 이슈를 추출
    # 트렌드 트리를 글로벌 트리와 병합하지 않고 별도의 소스로 관리
    # 이는 트렌드 키워드가 뉴스 키워드와 겹치지 않도록 하기 위함

    # 글로벌 뉴스 트리에서 대표 이슈 추출 (키워드 구문 생성)
    global_trees_info = extract_representative_info(global_news_trees, google_trends_g10, source='Global News')

    # Google 트렌드 트리에서 대표 이슈 추출 (키워드 구문 생성)
    trend_trees_info = extract_representative_info(trend_trees, google_trends_g10, source='Google Trends')

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
            if similarity >= 0.85 or edit_similarity >= 0.85:  # 코사인 유사도와 편집 거리 임계값 상향 조정
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

    # 데이터 저장 (선택 사항)
    # df = pd.DataFrame(unique_final_issues)
    # df.to_csv('top_10_multi_keywords.csv', index=False, encoding='utf-8-sig')
    # logging.info("상위 10개 멀티 키워드 데이터를 CSV 파일로 저장했습니다: top_10_multi_keywords.csv")

if __name__ == "__main__":
    main()
