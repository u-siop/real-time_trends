import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Hide FutureWarnings

import os
import re
import time
import logging
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from bs4 import BeautifulSoup
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from pytrends.request import TrendReq
from googletrans import Translator

import spacy
import pytextrank

# Initialize spaCy model with PyTextRank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# 웹페이지 스크래핑 함수
def scrape_webpage(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 예시: 네이버 뉴스 본문 추출
        article_body = soup.select_one('#dic_area')  # 네이버 뉴스 예시
        if article_body:
            article_text = article_body.get_text(separator=' ', strip=True)
            return article_text
        else:
            # 다른 뉴스 사이트의 본문 추출 로직 추가
            # 예: BBC, CNN, FOX NEWS 등
            if 'bbc.co.uk' in url:
                article_body = soup.select_one('.ssrcss-uf6wea-RichTextComponentWrapper')  # BBC 예시
            elif 'cnn.com' in url:
                article_body = soup.select_one('.l-container .zn-body__paragraph')  # CNN 예시
            elif 'foxnews.com' in url:
                article_body = soup.select_one('.article-body')  # FOX NEWS 예시
            elif 'nytimes.com' in url:
                article_body = soup.select_one('.css-53u6y8')  # New York Times 예시
            elif 'nhk.or.jp' in url:
                article_body = soup.select_one('.p-article-body__text')  # NHK 예시
            elif 'cctv.com' in url:
                article_body = soup.select_one('.article-content')  # CCTV 예시
            else:
                article_body = None

            if article_body:
                article_text = article_body.get_text(separator=' ', strip=True)
                return article_text
            else:
                logging.warning(f"본문을 찾을 수 없습니다: {url}")
                return ""
    except Exception as e:
        logging.error(f"웹페이지 스크래핑 오류 ({url}): {e}")
        return ""

# Google 트렌드 키워드 수집 함수 (G10 국가) - 최적화됨
def get_google_trends_g10():
    try:
        pytrends = TrendReq(hl='en', tz=360)
        countries = {
            'united_states': 'US',
            'united_kingdom': 'GB',
            'japan': 'JP',
            'china': 'CN',  # Note: pytrends may have limitations with China
            'germany': 'DE',
            'brazil': 'BR',
            'france': 'FR',
            'italy': 'IT',
            'canada': 'CA',
            'russia': 'RU'
        }

        all_trends = {}

        for country_name, country_code in countries.items():
            logging.info(f"Collecting trends for {country_name}")
            try:
                df_trending = pytrends.trending_searches(pn=country_name)
                trending_keywords = df_trending[0].tolist()

                # Use the trending keywords directly without fetching additional search volumes
                all_trends[country_name] = trending_keywords[:20]  # Limit to top 20

                logging.info(f"{country_name} trends collected: {len(all_trends[country_name])} keywords")
                # Optional: Implement caching here if needed

            except Exception as e:
                logging.error(f"{country_name} 트렌드 수집 중 오류 발생: {e}")

        return all_trends
    except Exception as e:
        logging.error(f"Google 트렌드 수집 중 오류 발생: {e}")
        return {}

# 뉴스 RSS 피드 수집 함수
def get_rss_news(rss_url):
    try:
        response = requests.get(rss_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        return [{'title': item.title.text, 'link': item.link.text} for item in items[:20]]
    except Exception as e:
        logging.error(f"RSS 뉴스 수집 중 오류 발생 ({rss_url}): {e}")
        return []

# 추가 뉴스 소스 함수
def get_newyorktimes_trending_news():
    rss_url = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
    return get_rss_news(rss_url)

def get_cctv_trending_news():
    rss_url = "http://news.cctv.com/rss/english.xml"  # 실제 RSS 피드 URL 필요
    return get_rss_news(rss_url)

# 텍스트 전처리 함수
def preprocess_text(text):
    if not text or not isinstance(text, str) or not text.strip():
        logging.warning("유효하지 않은 입력 텍스트.")
        return ""
    # 특수 문자 제거
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

# 대표 키워드 선정 함수 (Google 트렌드 우선, 불용어 필터링 추가)
def select_representative_keyword(top_keywords, used_keywords, google_trends_keywords):
    for kw in top_keywords:
        if (kw in google_trends_keywords) and (kw not in used_keywords) and (kw not in english_stopwords):
            return kw
    for kw in top_keywords:
        if (kw not in used_keywords) and (kw not in english_stopwords):
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
            if 'keywords' in news:
                keyword_counter.update([kw for kw in news.get('keywords', []) if kw not in english_stopwords])
            else:
                continue
        top_keywords = [word for word, freq in keyword_counter.most_common(5)]
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
            # Exclude phrases that contain stopwords
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
    # Google 트렌드 키워드 수집 (G10 국가)
    google_trends_g10 = get_google_trends_g10()
    logging.info("G10 국가 구글 트렌드 수집 완료")

    # 뉴스 소스별 RSS 피드 URL
    news_sources = {
        'BBC': "http://feeds.bbci.co.uk/news/rss.xml",
        'CNN': "http://rss.cnn.com/rss/edition.rss",
        'FOX NEWS': "http://feeds.foxnews.com/foxnews/latest",
        'NHK': "https://www3.nhk.or.jp/rss/news/cat0.xml",
        'New York Times': "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        'CCTV': "http://news.cctv.com/rss/english.xml"  # 실제 RSS 피드 URL 필요
    }

    # 뉴스 소스별 뉴스 수집
    all_news = {}
    for source, rss_url in news_sources.items():
        logging.info(f"Collecting news from {source}")
        news = get_rss_news(rss_url)
        all_news[source] = news
        logging.info(f"{source} 뉴스 수집 완료: {len(news)}개")

    # 기사별 텍스트 수집 (멀티스레딩)
    articles_texts = []
    articles_metadata = []

    def fetch_article_text(news, source):
        text = scrape_webpage(news['link'])
        if text:
            full_text = preprocess_text(news['title'] + ' ' + text)
            return full_text, news, source
        return None, news, source

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_article_text, news, source): (news, source)
            for source, source_news in all_news.items()
            for news in source_news
        }
        for future in as_completed(futures):
            full_text, news, source = future.result()
            if full_text:
                articles_texts.append(full_text)
                articles_metadata.append({'title': news['title'], 'link': news['link'], 'source': source})
            else:
                logging.warning(f"기사 본문이 없습니다: {news['title']}")

    logging.info(f"수집된 유효한 기사 수: {len(articles_texts)}")

    if not articles_texts:
        logging.error("유효한 기사가 하나도 없습니다. 프로그램을 종료합니다.")
        return

    # TextRank 기반 키워드 추출
    articles_keywords_list = []
    for idx, text in enumerate(articles_texts):
        keywords = extract_keywords_textrank(text, english_stopwords, top_n=10)
        if keywords:
            articles_keywords_list.append(keywords)
        else:
            articles_keywords_list.append([])
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
                tree_info['all_keywords'].update([kw for kw in keywords if kw not in english_stopwords])  # 집합으로 업데이트
                # 트리에 Google 트렌드 키워드 포함 여부 갱신
                trends = google_trends_g10.get(news['source'], [])
                tree_info['contains_trend'] = tree_info['contains_trend'] or any(word in trends for word in keywords)
                # 트리의 중요도 업데이트 (기사 수)
                tree_info['importance'] += 1
                merged = True
                logging.info(f"글로벌 트리 병합 완료: {news['title']} (유사도: {similarity:.2f})")
                break
        if not merged:
            # 새로운 글로벌 트리 생성
            trends = google_trends_g10.get(news['source'], [])
            contains_trend = any(word in trends for word in keywords)
            global_news_trees.append({
                'articles': [news],
                'all_keywords': set([kw for kw in keywords if kw not in english_stopwords]),  # 집합으로 초기화
                'contains_trend': contains_trend,
                'importance': 1  # 기사 수 초기화
            })
            logging.info(f"글로벌 새로운 트리 생성: {news['title']} (트렌드 포함: {'예' if contains_trend else '아니오'})")

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
                'all_keywords': set([kw for kw in keywords if kw not in english_stopwords]),
                'importance': 1  # 검색량을 중요도로 설정 (여기서는 단순히 1로 설정)
            })
            logging.info(f"Google 트렌드 새로운 트리 생성: {keyword} (국가: {country})")

    logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    # 글로벌 뉴스 트리와 Google 트렌드 트리에서 대표 이슈 추출 (키워드 구문 생성)
    global_trees_info = extract_representative_info(global_news_trees, google_trends_g10, source='Global News')
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

    # 중복 이슈 제거는 하지 않음 (각 소스에서 독립적으로 선택)
    # 최종 이슈 리스트 생성
    final_global_issues = top_global_issues
    final_trend_issues = top_trend_issues

    # 상위 6개 글로벌 뉴스 이슈와 상위 4개 트렌드 이슈를 합침
    final_issues = final_global_issues + final_trend_issues

    # 최종 이슈가 10개 미만일 경우, 부족한 만큼 글로벌 뉴스 전용이나 트렌드 전용에서 추가
    if len(final_issues) < 10:
        # 글로벌 뉴스에서 추가
        for item in sorted_global_trees_info[6:]:
            final_issues.append(item)
            if len(final_issues) >= 10:
                break
        # 트렌드에서 추가
        for item in sorted_trend_trees_info[4:]:
            final_issues.append(item)
            if len(final_issues) >= 10:
                break

    # 최종 이슈가 10개 미만일 경우, 추가로 채우기
    final_issues = final_issues[:10]

    # 상위 6개 글로벌 뉴스 이슈와 상위 4개 트렌드 이슈를 별도로 출력
    print("\n실시간 이슈:")
    print("\n글로벌 뉴스 상위 6개 이슈:")
    for rank, item in enumerate(top_global_issues, 1):
        phrase = item['phrase']
        print(f"{rank}. {phrase}")

    print("\nGoogle 트렌드 상위 4개 이슈:")
    for rank, item in enumerate(top_trend_issues, 1):
        phrase = item['phrase']
        print(f"{rank}. {phrase}")

    # 전체 10개 이슈를 함께 출력
    print("\n전체 실시간 이슈 (글로벌 뉴스 상위 6개 + Google 트렌드 상위 4개):")
    for rank, item in enumerate(final_issues, 1):
        phrase = item['phrase']
        print(f"{rank}. {phrase}")

    # 데이터 저장 (선택 사항)
    # df = pd.DataFrame(final_issues)
    # df.to_csv('real_time_issues.csv', index=False, encoding='utf-8-sig')
    # logging.info("실시간 이슈 데이터를 CSV 파일로 저장했습니다: real_time_issues.csv")

if __name__ == "__main__":
    main()
