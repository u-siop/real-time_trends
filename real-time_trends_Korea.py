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
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

from konlpy.tag import Okt
from soynlp.word import WordExtractor
from pytrends.request import TrendReq

import numpy as np
from tqdm import tqdm  # 진행 표시기 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 형태소 분석기 초기화 (Okt)
okt = Okt()

# 불용어 리스트를 파일에서 읽어오기
def load_stopwords(file_path):
    default_stopwords = set([
        # 기존 불용어에 조사, 접속사 등 추가
        '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과',
        '도', '를', '으로', '자', '에', '와', '한', '하다', '것', '말',
        '그', '저', '및', '더', '등', '데', '때', '년', '월', '일',
        '그리고', '하지만', '그러나', '그래서', '또는', '즉', '만약', '아니면',
        '때문에', '그런데', '그러므로', '따라서', '뿐만 아니라',
        '이런', '저런', '합니다', '있습니다', '했습니다', '있다', '없다'
    ])
    if not os.path.exists(file_path):
        logging.warning(f"불용어 파일을 찾을 수 없습니다: {file_path}. 기본 불용어를 사용합니다.")
        return default_stopwords
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        stopwords.update(default_stopwords)
        # 추가 불용어
        additional_stopwords = {'대표', 'A씨', 'B씨', '것으로', '등등'}
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

# 텍스트 밀도를 이용하여 뉴스 본문만을 스크랩하여 반환하는 함수
def scrape_webpage(url):
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
    text = re.sub(r'[^가-힣a-zA-Z\s]', ' ', text)
    # 여러 개의 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# N-그램 기반 키워드 추출 함수 (멀티 워드 지원)
def extract_ngrams(tokens, n=2):
    if len(tokens) < n:
        return []
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(sorted(ngram)) for ngram in ngrams]  # 단어 정렬

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
    # 키워드의 단어 순서를 정렬하여 일관성 유지
    sorted_top_keywords = [' '.join(sorted(kw.split())) for kw in top_keywords]
    for sorted_kw, original_kw in zip(sorted_top_keywords, top_keywords):
        if (original_kw in google_trends_keywords) and (original_kw not in used_keywords) and (original_kw not in stopwords):
            return sorted_kw  # 정렬된 키워드 반환
    for sorted_kw, original_kw in zip(sorted_top_keywords, top_keywords):
        if (original_kw not in used_keywords) and (original_kw not in stopwords):
            return sorted_kw
    return None

# 트리 별로 대표 이슈 추출 함수 (키워드 구문 생성)
def extract_representative_info(trees, source='Naver'):
    trees_info = []
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
        # 대표 키워드 선정 시 Google 트렌드 키워드 우선
        rep_keyword = select_representative_keyword(top_keywords, set(), tree_info.get('google_trends_keywords', []))
        if not rep_keyword:
            rep_keyword = top_keywords[0] if top_keywords else None
        if not rep_keyword:
            continue  # 대표 키워드가 없으면 스킵
        # 대표 키워드 제외 상위 키워드
        top_other_keywords = [kw for kw in top_keywords if kw != rep_keyword]
        if top_other_keywords:
            # 키워드의 단어 순서를 정렬하여 일관성 유지
            sorted_other_kw = ' '.join(top_other_keywords)
            # 이슈 구문은 2~3단어로 제한
            phrase = f"{rep_keyword} {sorted_other_kw}"
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

        # 가장 적절한 링크 선택 (첫 번째 기사 링크 사용)
        representative_link = articles[0]['link'] if articles else 'No Link'

        combined_info = {
            'phrase': phrase,
            'importance': importance,
            'source': source,  # 트리의 출처 추가
            'representative_link': representative_link  # 대표 링크 추가
        }
        trees_info.append(combined_info)
        logging.info(f"대표 이슈 추가됨: {phrase} - 중요도: {importance} - 출처: {source}")
    return trees_info

# 키워드 추출 함수 (soynlp 이용 + N-그램 + 멀티 워드)
def extract_keywords(text, stopwords, top_n=10):
    try:
        # konlpy를 이용한 명사 추출
        tokens = okt.nouns(text)
        # 불용어 및 1단어, 4단어 이상 단어 필터링
        filtered_tokens = sorted([word for word in tokens if 2 <= len(word) <= 3 and word not in stopwords])
        # 단어 빈도수 계산
        token_counts = Counter(filtered_tokens)
        top_keywords = [word for word, count in token_counts.most_common(top_n)]

        # 멀티 워드 추출 (2-그램)
        bigrams = extract_ngrams(filtered_tokens, n=2)

        # 빅그램 빈도수 계산
        bigram_counts = Counter([bg for bg in bigrams if bg not in stopwords and len(bg.replace(' ', '')) >= 2])
        sorted_bigrams = [bg for bg, cnt in bigram_counts.most_common(top_n)]

        # 결합된 키워드 목록
        combined_keywords = top_keywords + sorted_bigrams

        # 중복 제거 및 상위 N개 선택
        unique_keywords = []
        for kw in combined_keywords:
            if kw not in unique_keywords:
                unique_keywords.append(kw)
            if len(unique_keywords) >= top_n:
                break

        return unique_keywords[:top_n]
    except Exception as e:
        logging.error(f"키워드 추출 중 오류 발생: {e}")
        return []

# Selenium Manager 클래스 정의
class SeleniumManager:
    def __init__(self):
        options = Options()
        options.add_argument("--headless")  # 브라우저를 표시하지 않음
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.wait = WebDriverWait(self.driver, 10)  # 10초 타임아웃

    def get(self, url):
        self.driver.get(url)

    def find_element(self, by, value):
        return self.wait.until(EC.presence_of_element_located((by, value)))

    def find_elements(self, by, value):
        return self.wait.until(EC.presence_of_all_elements_located((by, value)))

    def click_element(self, by, value):
        element = self.wait.until(EC.element_to_be_clickable((by, value)))
        element.click()

    def quit(self):
        self.driver.quit()

# 네이버 뉴스 검색 및 상위 3개 키워드 추출 함수 (Selenium 사용)
def search_naver_news_with_keyword(keyword, stopwords, selenium_manager):
    try:
        # 네이버 뉴스 검색 URL
        search_url = f"https://search.naver.com/search.naver?&where=news&query={requests.utils.quote(keyword)}"
        selenium_manager.get(search_url)
        
        # 뉴스 검색 결과에서 제목과 링크 추출
        news_elements = selenium_manager.find_elements(By.CSS_SELECTOR, '.list_news .news_area')
        news_items = []
        for elem in news_elements:
            try:
                title_elem = elem.find_element(By.CSS_SELECTOR, '.news_tit')
                title = title_elem.text.strip()
                link = title_elem.get_attribute('href')
                news_items.append({'title': title, 'link': link})
            except NoSuchElementException:
                continue

        if not news_items:
            logging.warning(f"네이버 뉴스 검색 결과가 없습니다: {keyword}")
            return []

        # 상위 10개 뉴스 기사에서 텍스트 추출
        articles_texts = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(scrape_webpage, news['link']) for news in news_items[:10]]
            for future in as_completed(futures):
                article_text = future.result()
                if article_text:
                    full_text = preprocess_text(article_text)
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

# 메인 함수
def main():
    # Google 트렌드 키워드 수집
    trending_keywords = get_google_trends_keywords()
    logging.info(f"Google 트렌드 키워드 수집 완료: {len(trending_keywords)}개")
    
    keyword_volume = []
    trend_trees = []
    
    logging.info("Google 트렌드 검색량 수집 시작")
    
    # Selenium Manager 초기화
    selenium_manager = SeleniumManager()
    
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
        
        # 키워드 전처리 및 키워드 추출
        full_text = preprocess_text(keyword)
        keywords = extract_keywords(full_text, stopwords, top_n=10)
        
        # 네이버 뉴스에서 추가 키워드 검색
        top3_news_keywords = search_naver_news_with_keyword(keyword, stopwords, selenium_manager)
        
        # Google 트렌드 키워드와 네이버 뉴스 상위 키워드를 합침 (원본 트렌드 키워드도 포함)
        combined_keywords = keywords + top3_news_keywords + [keyword]  # 원본 트렌드 키워드 추가
        combined_keywords = list(set(combined_keywords) - stopwords)  # 불용어 제거 및 중복 제거
        
        # 트렌드 트리 생성
        trend_tree = {
            'articles': [{
                'title': keyword,
                'link': 'https://trends.google.com/trends/trendingsearches/daily?geo=KR',
                'keywords': combined_keywords
            }],
            'all_keywords': set(combined_keywords),
            'importance': volume,  # 검색량을 중요도로 설정
            'google_trends_keywords': [keyword]  # 원본 트렌드 키워드 저장
        }
        
        trend_trees.append(trend_tree)
        logging.info(f"Google 트렌드 및 네이버 뉴스 통합 트리 생성: {keyword} (검색량: {volume})")

    logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    # 네이버 뉴스 처리
    site_url = "https://news.naver.com/main/ranking/popularDay.naver"

    try:
        selenium_manager.get(site_url)

        # "다른 언론사 랭킹 더보기" 버튼 클릭 반복
        while True:
            try:
                more_button = selenium_manager.find_element(By.CSS_SELECTOR, '.button_rankingnews_more')
                if more_button.is_displayed():
                    selenium_manager.click_element(By.CSS_SELECTOR, '.button_rankingnews_more')
                    logging.info("더보기 버튼 클릭")
                else:
                    break
            except TimeoutException:
                logging.info("더보기 버튼이 더 이상 존재하지 않습니다.")
                break
            except Exception as e:
                logging.error(f"더보기 버튼 클릭 중 오류: {e}")
                break

        # 모든 뉴스 기사 요소 수집
        news_elements = selenium_manager.find_elements(By.CSS_SELECTOR, '.rankingnews_list .list_title')
        news_items = [
            {'title': elem.text, 'link': elem.get_attribute('href')}
            for elem in news_elements 
        ]
        logging.info(f"수집된 뉴스 기사 수: {len(news_items)}")
    finally:
        # Selenium 드라이버 종료
        selenium_manager.quit()

    # 기사별 텍스트 수집
    articles_texts = []
    articles_metadata = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_news = {executor.submit(scrape_webpage, news['link']): news for news in news_items}
        # tqdm을 이용한 진행 표시기 추가
        for future in tqdm(as_completed(future_to_news), total=len(future_to_news), desc="뉴스 기사 스크래핑"):
            news = future_to_news[future]
            try:
                article_text = future.result()
                if article_text:
                    # 텍스트를 합침 (제목 + 본문)
                    full_text = preprocess_text(news['title'] + ' ' + article_text)
                    if not full_text:
                        continue
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
            if similarity >= 0.3:  # 유사도 임계값 상향 조정
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
                'importance': 1  # 기사 수 초기화
            })
            logging.info(f"네이버 새로운 트리 생성: {news['title']} (트렌드 포함: {'예' if contains_trend else '아니오'})")

    logging.info(f"네이버 트리의 개수: {len(naver_trees)}")
    logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    # 네이버 트리와 Google 트렌드 트리에서 대표 이슈 추출 (키워드 구문 생성)
    naver_trees_info = extract_representative_info(naver_trees, source='Naver')
    trend_trees_info = extract_representative_info(trend_trees, source='Google Trends')

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
    top_naver_issues = sorted_naver_trees_info[:6]

    # 상위 4개 트렌드 이슈 선택
    top_trend_issues = sorted_trend_trees_info[:4]

    # 최종 이슈 리스트 생성
    final_naver_issues = top_naver_issues
    final_trend_issues = top_trend_issues

    # 상위 6개 네이버 이슈와 상위 4개 트렌드 이슈, 그리고 검색 기반 상위 15개 키워드를 합침
    final_issues = final_naver_issues + final_trend_issues

    # 최종 이슈가 10개 미만일 경우, 부족한 만큼 네이버 전용이나 트렌드 전용에서 추가
    if len(final_issues) < 10:
        # 네이버에서 추가
        for item in sorted_naver_trees_info[6:]:
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

    # 상위 6개 네이버 이슈와 상위 4개 트렌드 이슈, 그리고 검색 기반 상위 15개 키워드를 별도로 출력
    print("\n실시간 이슈:")
    print("\n네이버 뉴스 상위 6개 이슈:")
    for rank, item in enumerate(top_naver_issues, 1):
        phrase = item['phrase']
        link = item['representative_link']
        print(f"{rank}. {phrase} - 링크: {link}")

    print("\nGoogle 트렌드 상위 4개 이슈:")
    for rank, item in enumerate(top_trend_issues, 1):
        phrase = item['phrase']
        link = item['representative_link']
        print(f"{rank}. {phrase} - 링크: {link}")

    # 필요시 전체 10개 이슈를 함께 출력
    print("\n전체 실시간 이슈 (네이버 상위 6개 + Google 트렌드 상위 4개):")
    for rank, item in enumerate(final_issues, 1):
        phrase = item['phrase']
        link = item['representative_link']
        print(f"{rank}. {phrase} - 링크: {link}")

if __name__ == "__main__":
    main()
