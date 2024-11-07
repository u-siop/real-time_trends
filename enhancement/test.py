import os
import re
import time
import math
import logging
import requests
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from konlpy.tag import Komoran
from soynlp.word import WordExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pytrends.request import TrendReq
import numpy as np
import schedule

# FutureWarning 무시
warnings.simplefilter(action='ignore', category=FutureWarning)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 싱글톤 패턴을 적용한 MongoDB 연결 클래스
class MongoDBConnection:
    _instance = None

    def __new__(cls, uri="mongodb://localhost:27017/"):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
            cls._instance.client = MongoClient(uri)
        return cls._instance

    def get_database(self, db_name):
        return self.client[db_name]

# 불용어 로더 클래스
class StopwordLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.stopwords = self.load_stopwords()

    def load_stopwords(self):
        default_stopwords = set([
            '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과',
            '도', '를', '으로', '자', '에', '와', '한', '하다', '것', '말',
            '그', '저', '및', '더', '등', '데', '때', '년', '월', '일',
            '그리고', '하지만', '그러나', '그래서', '또는', '즉', '만약', '아니면',
            '때문에', '그런데', '그러므로', '따라서', '뿐만 아니라',
            '이런', '저런', '합니다', '있습니다', '했습니다', '있다', '없다',
            '됩니다', '되다', '이다', '대표', 'A씨', 'B씨', '것으로', '등등', '있다', '없다'
        ])
        if not os.path.exists(self.file_path):
            logging.warning(f"불용어 파일을 찾을 수 없습니다: {self.file_path}. 기본 불용어를 사용합니다.")
            return default_stopwords
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                stopwords = set(line.strip() for line in f if line.strip())
            stopwords.update(default_stopwords)
            logging.info(f"불용어 {len(stopwords)}개를 로드했습니다.")
            return stopwords
        except Exception as e:
            logging.error(f"불용어 파일 로드 중 오류 발생: {e}")
            return default_stopwords

# 웹 스크래핑 클래스 (Facade 패턴)
class WebScraper:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    def scrape_webpage(self, url):
        try:
            response = requests.get(url, timeout=10, headers=self.headers)
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

    def scrape_webpage_for_google_search(self, url):
        try:
            response = requests.get(url, timeout=10, headers=self.headers)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            html = response.text

            soup = BeautifulSoup(html, 'html.parser')

            # 불필요한 요소 제거
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
                density = self.calculate_text_density(block)
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

    def calculate_text_density(self, html_element):
        text_length = len(html_element.get_text(strip=True))
        tag_length = len(str(html_element))
        return text_length / max(tag_length, 1)

# Google 트렌드 키워드 관리 클래스
class GoogleTrendsManager:
    def __init__(self):
        self.pytrends = TrendReq(hl='ko', tz=540)

    def get_trending_keywords(self):
        try:
            df_trending = self.pytrends.trending_searches(pn='south_korea')
            trending_keywords = df_trending[0].tolist()
            return trending_keywords
        except Exception as e:
            logging.error(f"Google 트렌드 키워드 수집 중 오류 발생: {e}")
            return []

    def get_keyword_volume(self, keyword):
        try:
            self.pytrends.build_payload([keyword], cat=0, timeframe='now 1-d', geo='KR')
            interest_over_time_df = self.pytrends.interest_over_time()
            if not interest_over_time_df.empty:
                avg_volume = interest_over_time_df[keyword].mean()
                return avg_volume
            else:
                return 0
        except Exception as e:
            logging.error(f"키워드 '{keyword}' 검색량 가져오는 중 오류 발생: {e}")
            return 0

# 형태소 분석기 클래스 (Strategy 패턴)
class KeywordExtractor:
    def __init__(self, stopwords):
        self.stopwords = stopwords
        self.komoran = Komoran()

    def preprocess_text(self, text):
        if not text or not isinstance(text, str) or not text.strip():
            logging.warning("유효하지 않은 입력 텍스트.")
            return ""
        text = re.sub(r'[^0-9가-힣a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_keywords(self, text, top_n=10):
        try:
            tokens = self.komoran.nouns(text)
            filtered_tokens = [word for word in tokens if len(word) >= 2 and word not in self.stopwords]
            token_counts = Counter(filtered_tokens)
            top_keywords = [word for word, count in token_counts.most_common(top_n)]

            # 멀티 워드 추출
            combined_ngrams = []
            for n in range(2, 6):
                ngrams = self.extract_ngrams(filtered_tokens, n)
                combined_ngrams.extend(ngrams)

            ngram_counts = Counter([ng for ng in combined_ngrams if ng not in self.stopwords and len(ng.replace(' ', '')) >= 2])
            sorted_ngrams = [ng for ng, cnt in ngram_counts.most_common(top_n)]

            combined_keywords = top_keywords + sorted_ngrams
            unique_keywords = []
            combined_keywords_sorted = sorted(combined_keywords, key=lambda x: len(x), reverse=True)
            for kw in combined_keywords_sorted:
                if kw in unique_keywords:
                    continue
                if not any((kw != other and kw in other) for other in unique_keywords):
                    unique_keywords.append(kw)
                if len(unique_keywords) >= top_n:
                    break

            unique_keywords = self.remove_invalid_keywords(unique_keywords)
            return unique_keywords[:top_n]
        except Exception as e:
            logging.error(f"키워드 추출 중 오류 발생: {e}")
            return []

    def extract_ngrams(self, tokens, n=2):
        if len(tokens) < n:
            return []
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return [' '.join(ngram) for ngram in ngrams]

    def remove_invalid_keywords(self, keywords):
        valid_keywords = [kw for kw in keywords if re.match(r'^[0-9a-zA-Z가-힣\s]{2,}$', kw)]
        return valid_keywords

    def remove_substrings_and_duplicates(self, keywords):
        unique_keywords = []
        sorted_keywords = sorted(keywords, key=lambda x: len(x), reverse=True)
        for kw in sorted_keywords:
            if kw not in unique_keywords:
                unique_keywords.append(kw)
            if len(unique_keywords) >= 10:
                break
        return unique_keywords

# 뉴스 검색 클래스
class NaverNewsSearcher:
    def __init__(self, keyword_extractor):
        self.keyword_extractor = keyword_extractor
        self.web_scraper = WebScraper()

    def search_news_with_keyword(self, keyword):
        try:
            search_url = f"https://search.naver.com/search.naver?&where=news&query={requests.utils.quote(keyword)}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 뉴스 기사 요소 선택자 수정
            news_elements = soup.select('ul.list_news > li.bx')
            news_items = []
            for elem in news_elements:
                # 제목 및 링크 추출
                title_elem = elem.select_one('a.news_tit')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href')
                    # 뉴스 요약 본문 추출
                    desc_elem = elem.select_one('div.dsc_wrap')
                    if desc_elem:
                        description = desc_elem.get_text(separator=' ', strip=True)
                    else:
                        description = ''
                    news_items.append({'title': title, 'link': link, 'description': description})

            articles_texts = []
            for news in news_items[:10]:
                # 기사 본문 스크래핑
                article_text = self.web_scraper.scrape_webpage_for_google_search(news['link'])
                if article_text:
                    full_text = self.keyword_extractor.preprocess_text(news['title'] + ' ' + news['description'] + ' ' + article_text)
                    if full_text:
                        articles_texts.append(full_text)
                else:
                    # 기사 본문을 가져오지 못한 경우에도 제목과 요약을 사용
                    full_text = self.keyword_extractor.preprocess_text(news['title'] + ' ' + news['description'])
                    if full_text:
                        articles_texts.append(full_text)

            if not articles_texts:
                logging.warning(f"네이버 뉴스 검색 결과에서 텍스트를 추출할 수 없습니다: {keyword}")
                return []

            # 모든 기사 텍스트를 합쳐서 키워드 추출
            combined_text = ' '.join(articles_texts)
            keywords = self.keyword_extractor.extract_keywords(combined_text, top_n=5)
            top_keywords = keywords[:5] if len(keywords) >= 5 else keywords

            logging.info(f"키워드 '{keyword}'에 대한 상위 키워드: {top_keywords}")
            return top_keywords

        except Exception as e:
            logging.error(f"네이버 뉴스 검색 중 오류 발생 ({keyword}): {e}")
            return []

# 이슈 병합 클래스
class IssueMerger:
    def __init__(self):
        pass

    def merge_similar_issues(self, issues, similarity_threshold=0.2):
        if not issues:
            return []

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

        # 키워드 집합 준비
        keywords_list = [issue['all_keywords'] for issue in issues]

        # 유사도 계산 및 병합
        for i in range(n):
            for j in range(i + 1, n):
                intersection = keywords_list[i] & keywords_list[j]
                union_set = keywords_list[i] | keywords_list[j]
                similarity = len(intersection) / len(union_set) if union_set else 0
                if similarity >= similarity_threshold:
                    union(i, j)

        # 그룹화
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
            combined_importance = sum(issues[idx]['importance'] for idx in group)
            combined_sources = set(issues[idx]['source'] for idx in group)
            combined_keywords = set()
            for idx in group:
                combined_keywords.update(issues[idx]['all_keywords'])

            # 키워드 중요도 기반 정렬
            keyword_counter = Counter()
            for idx in group:
                keyword_counter.update(issues[idx]['all_keywords'])
            sorted_keywords = [kw for kw, _ in keyword_counter.most_common()]

            # 대표 키워드 선택
            phrase_keywords = self.select_representative_keywords(sorted_keywords, max_words=5)

            if not phrase_keywords:
                continue

            phrase = ', '.join(phrase_keywords)

            merged_issues.append({
                'phrase': phrase,
                'importance': combined_importance,
                'source': ', '.join(combined_sources),
                'all_keywords': combined_keywords
            })
            logging.info(f"이슈 병합: {[issues[idx]['phrase'] for idx in group]} -> {phrase}")

        return merged_issues

    def select_representative_keywords(self, keywords, max_words=5):
        # 키워드 길이와 빈도를 고려하여 대표 키워드 선택
        unique_keywords = []
        for kw in keywords:
            if kw not in unique_keywords:
                unique_keywords.append(kw)
            if len(unique_keywords) >= max_words:
                break
        return unique_keywords

class TreeBuilder:
    def __init__(self, articles_metadata, articles_keywords_list):
        self.articles_metadata = articles_metadata
        self.articles_keywords_list = articles_keywords_list

    def build_trees(self, similarity_threshold=0.2):
        naver_trees = []
        for idx, news in enumerate(tqdm(self.articles_metadata, desc="네이버 트리 생성")):
            keywords = self.articles_keywords_list[idx]
            if not keywords:
                continue
            merged = False
            for tree_info in naver_trees:
                similarity = self.calculate_jaccard_similarity(
                    keywords, 
                    tree_info['all_keywords']
                )
                if similarity >= similarity_threshold:
                    tree_info['articles'].append(news)
                    tree_info['all_keywords'].update(keywords)
                    tree_info['importance'] += 1
                    merged = True
                    logging.info(f"네이버 트리 병합 완료: {news['title']} (유사도: {similarity:.2f})")
                    break
            if not merged:
                naver_trees.append({
                    'articles': [news],
                    'all_keywords': set(keywords),
                    'importance': 1
                })
                logging.info(f"네이버 새로운 트리 생성: {news['title']}")

        return naver_trees

    def calculate_jaccard_similarity(self, keywords1, keywords2):
        set1 = set(keywords1)
        set2 = set(keywords2)
        intersection = set1 & set2
        union = set1 | set2
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def calculate_hierarchical_similarity(self, tree1, tree2):
        # 각 계층의 키워드 리스트
        keywords1_root = [tree1['root']]
        keywords1_depth1 = tree1['depth1']
        keywords1_depth2 = tree1['depth2']

        keywords2_root = [tree2['root']]
        keywords2_depth1 = tree2['depth1']
        keywords2_depth2 = tree2['depth2']

        # 계층별 유사도 계산 (Jaccard 유사도 사용)
        sim_root = self.calculate_jaccard_similarity(keywords1_root, keywords2_root)
        sim_depth1 = self.calculate_jaccard_similarity(keywords1_depth1, keywords2_depth1)
        sim_depth2 = self.calculate_jaccard_similarity(keywords1_depth2, keywords2_depth2)

        # 가중치 적용 (root: 0.5, depth1: 0.3, depth2: 0.2)
        total_similarity = (sim_root * 0.5) + (sim_depth1 * 0.3) + (sim_depth2 * 0.2)

        return total_similarity

    def extract_tree_keywords(self, naver_trees):
        # 각 트리에서 중요 키워드를 추출하여 트리의 계층 구조 생성
        tree_structures = []
        for tree in naver_trees:
            keyword_counter = Counter()
            for article in tree['articles']:
                keyword_counter.update(article['keywords'])
            sorted_keywords = [kw for kw, _ in keyword_counter.most_common()]
            if not sorted_keywords:
                continue
            root_keyword = sorted_keywords[0]
            depth1_keywords = sorted_keywords[1:4]  # 상위 3개 키워드
            depth2_keywords = sorted_keywords[4:7]  # 다음 3개 키워드
            tree_structures.append({
                'root': root_keyword,
                'depth1': depth1_keywords,
                'depth2': depth2_keywords,
                'importance': tree['importance'],
                'timestamp': datetime.now()
            })
        return tree_structures

    # 트리 병합 메서드 수정
    def merge_trees(self, tree_structures, similarity_threshold=0.3):
        merged_trees = []
        for tree in tree_structures:
            merged = False
            for m_tree in merged_trees:
                # 계층적 유사도 계산
                similarity = self.calculate_hierarchical_similarity(tree, m_tree)
                if similarity >= similarity_threshold:
                    # 트리 병합
                    m_tree['importance'] += tree['importance']
                    m_tree['timestamp'] = max(m_tree['timestamp'], tree['timestamp'])
                    # 키워드 병합
                    combined_keywords = [m_tree['root'], tree['root']] + m_tree['depth1'] + tree['depth1'] + m_tree['depth2'] + tree['depth2']
                    keyword_counter = Counter(combined_keywords)
                    sorted_keywords = [kw for kw, _ in keyword_counter.most_common()]
                    m_tree['root'] = sorted_keywords[0]
                    m_tree['depth1'] = sorted_keywords[1:4]
                    m_tree['depth2'] = sorted_keywords[4:7]
                    merged = True
                    logging.info(f"트리 병합 완료: {tree['root']} -> {m_tree['root']} (유사도: {similarity:.2f})")
                    break
            if not merged:
                merged_trees.append(tree)
        return merged_trees

# 메인 실행 클래스
class MainExecutor:
    def __init__(self):
        self.stopwords = StopwordLoader('stopwords-ko.txt').stopwords
        self.mongo_client = MongoDBConnection().client
        self.db = self.mongo_client['news_db']
        self.news_collection = self.db['news_articles']
        self.trees_collection = self.db['news_trees']
        self.final_issues_collection = self.db['final_issues']
        # 트리 구조를 저장할 새로운 컬렉션 생성
        self.tree_structures_collection = self.db['tree_structures']
        self.keyword_extractor = KeywordExtractor(self.stopwords)
        self.web_scraper = WebScraper()
        self.google_trends_manager = GoogleTrendsManager()
        self.naver_news_searcher = NaverNewsSearcher(self.keyword_extractor)
        self.issue_merger = IssueMerger()

    def run(self):
        self.current_time = datetime.now()
        importance_reset_interval = 24  # 중요도 초기화 간격 (시간 단위)

        # Google 트렌드 키워드 수집
        trending_keywords = self.google_trends_manager.get_trending_keywords()
        logging.info(f"Google 트렌드 키워드 수집 완료: {len(trending_keywords)}개")

        keyword_volume = []
        trend_trees = []

        logging.info("Google 트렌드 검색량 수집 시작")

        for keyword in tqdm(trending_keywords, desc="Google 트렌드 키워드 처리"):
            if keyword in self.stopwords:
                logging.info(f"불용어 키워드 스킵: {keyword}")
                continue
            volume = self.google_trends_manager.get_keyword_volume(keyword)
            keyword_volume.append((keyword, volume))

        sorted_keywords = sorted(keyword_volume, key=lambda x: -x[1])[:20]  # 상위 20개로 변경
        logging.info("검색량 기준으로 정렬된 상위 20개 트렌드 키워드:")
        for i, (kw, vol) in enumerate(sorted_keywords, 1):
            logging.info(f"{i}. {kw} - 검색량: {vol}")

        # 상위 10개 트렌드 키워드를 네이버 뉴스에서 검색하여 키워드 추출
        trend_top_keywords = []
        logging.info("Google 트렌드 키워드를 네이버 뉴스에서 검색하여 키워드 추출 시작")
        for keyword, _ in tqdm(sorted_keywords[:10], desc="Google 트렌드 키워드 네이버 뉴스 검색"):
            top_keywords = self.naver_news_searcher.search_news_with_keyword(keyword)
            if top_keywords:
                trend_top_keywords.extend(top_keywords)

        # 네이버 뉴스 처리
        site_url = "https://news.naver.com/main/ranking/popularDay.naver"
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) as driver:
            driver.get(site_url)
            time.sleep(3)
            while True:
                try:
                    more_button = driver.find_element(By.CSS_SELECTOR, '.button_rankingnews_more')
                    if more_button.is_displayed():
                        driver.execute_script("arguments[0].click();", more_button)
                        logging.info("더보기 버튼 클릭")
                        time.sleep(2)
                    else:
                        break
                except NoSuchElementException:
                    logging.info("더보기 버튼이 더 이상 존재하지 않습니다.")
                    break
                except Exception as e:
                    logging.error(f"더보기 버튼 클릭 중 오류: {e}")
                    break

            news_elements = driver.find_elements(By.CSS_SELECTOR, '.rankingnews_list .list_title')
            news_items = [
                {'title': elem.text, 'link': elem.get_attribute('href')}
                for elem in news_elements 
            ]
            logging.info(f"수집된 뉴스 기사 수: {len(news_items)}")

        articles_texts = []
        articles_metadata = []

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_news = {executor.submit(self.web_scraper.scrape_webpage, news['link']): news for news in news_items}
            for future in tqdm(as_completed(future_to_news), total=len(future_to_news), desc="뉴스 기사 스크래핑"):
                news = future_to_news[future]
                try:
                    article_text = future.result()
                    if article_text:
                        full_text = self.keyword_extractor.preprocess_text(news['title'] + ' ' + article_text)
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
            keywords = self.keyword_extractor.extract_keywords(text, top_n=10)
            if keywords:
                articles_keywords_list.append(keywords)
            else:
                articles_keywords_list.append([])
            articles_metadata[idx]['keywords'] = keywords

        # 트리 생성
        tree_builder = TreeBuilder(articles_metadata, articles_keywords_list)
        naver_trees = tree_builder.build_trees(similarity_threshold=0.2)

        # 트리의 키워드 계층 구조 추출
        tree_structures = tree_builder.extract_tree_keywords(naver_trees)

        logging.info(f"생성된 네이버 뉴스 트리의 개수: {len(tree_structures)}")

        # 구글 트렌드 기반 기사 수집 및 트리 생성
        trend_articles_texts = []
        trend_articles_metadata = []

        logging.info("구글 트렌드 키워드를 사용하여 기사 수집 및 트리 생성 시작")
        for keyword, _ in tqdm(sorted_keywords[:10], desc="구글 트렌드 키워드로 기사 수집"):
            news_items = []
            try:
                # 네이버 뉴스 검색
                search_url = f"https://search.naver.com/search.naver?&where=news&query={requests.utils.quote(keyword)}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(search_url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                news_elements = soup.select('ul.list_news > li.bx')
                for elem in news_elements[:5]:  # 상위 5개 기사만 수집
                    title_elem = elem.select_one('a.news_tit')
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        link = title_elem.get('href')
                        news_items.append({'title': title, 'link': link})
            except Exception as e:
                logging.error(f"네이버 뉴스 검색 중 오류 발생 ({keyword}): {e}")
                continue

            # 기사 본문 스크래핑
            for news in news_items:
                article_text = self.web_scraper.scrape_webpage(news['link'])
                if article_text:
                    full_text = self.keyword_extractor.preprocess_text(news['title'] + ' ' + article_text)
                    if full_text:
                        trend_articles_texts.append(full_text)
                        trend_articles_metadata.append({'title': news['title'], 'link': news['link']})
                else:
                    logging.warning(f"기사 본문이 없습니다: {news['title']}")

        # 키워드 추출
        trend_articles_keywords_list = []
        for idx, text in enumerate(trend_articles_texts):
            keywords = self.keyword_extractor.extract_keywords(text, top_n=10)
            if keywords:
                trend_articles_keywords_list.append(keywords)
            else:
                trend_articles_keywords_list.append([])
            trend_articles_metadata[idx]['keywords'] = keywords

        # 트리 생성
        trend_tree_builder = TreeBuilder(trend_articles_metadata, trend_articles_keywords_list)
        trend_trees = trend_tree_builder.build_trees(similarity_threshold=0.2)

        # 트리의 키워드 계층 구조 추출
        trend_tree_structures = trend_tree_builder.extract_tree_keywords(trend_trees)

        logging.info(f"생성된 구글 트렌드 트리의 개수: {len(trend_tree_structures)}")

        # 구글 트렌드 기반 트리의 중요도에 가중치 부여
        for tree in trend_tree_structures:
            tree['importance'] *= 0.8  # 가중치 조정 (1.5는 예시값)

        # 두 트리 리스트를 합침
        combined_tree_structures = tree_structures + trend_tree_structures

        # 트리 병합 수행
        combined_tree_structures = tree_builder.merge_trees(combined_tree_structures, similarity_threshold=0.3)
        logging.info(f"트리 병합 후 트리의 개수: {len(combined_tree_structures)}")

        # 데이터베이스에 저장하기 전에 기존 DB와 비교하여 중복 처리
        for tree in combined_tree_structures:
            existing_tree = self.tree_structures_collection.find_one({
                'root': tree['root'],
                'depth1': tree['depth1'],
                'depth2': tree['depth2']
            })
            if existing_tree:
                # 중요도 업데이트
                last_update_time = existing_tree.get('last_update_time', existing_tree['timestamp'])
                time_diff = (self.current_time - last_update_time).total_seconds() / 3600  # 시간 단위

                # 중요도 초기화 여부 확인
                if time_diff >= importance_reset_interval:
                    # 중요도 초기화
                    new_importance = tree['importance']
                    logging.info(f"중요도 초기화: ROOT - {tree['root']}")
                else:
                    # 기존 중요도에 현재 중요도를 합산
                    new_importance = existing_tree['importance'] + tree['importance']

                # 트리 업데이트
                self.tree_structures_collection.update_one(
                    {'_id': existing_tree['_id']},
                    {'$set': {
                        'importance': new_importance,
                        'timestamp': self.current_time,
                        'last_update_time': self.current_time
                    }}
                )
                logging.info(f"트리 업데이트: ROOT - {tree['root']} (중요도 갱신)")
            else:
                # 새로운 트리 삽입
                tree_doc = {
                    'root': tree['root'],
                    'depth1': tree['depth1'],
                    'depth2': tree['depth2'],
                    'importance': tree['importance'],
                    'timestamp': self.current_time,
                    'last_update_time': self.current_time
                }
                try:
                    self.tree_structures_collection.insert_one(tree_doc)
                    logging.info(f"새로운 트리 저장 완료: ROOT - {tree['root']}")
                except Exception as e:
                    logging.error(f"트리 저장 중 오류 발생: {e}")

        # 트리 구조 출력
        top_trees = list(self.tree_structures_collection.find().sort('importance', -1).limit(10))
        logging.info("최종 상위 10개 트리:")
        for idx, tree in enumerate(top_trees, 1):
            logging.info(f"트리 {idx}:")
            logging.info(f"  ROOT: {tree['root']}")
            logging.info(f"  Depth 1: {', '.join(tree['depth1'])}")
            logging.info(f"  Depth 2: {', '.join(tree['depth2'])}")
            logging.info(f"  Importance: {tree['importance']}")

def run_main_executor():
    executor = MainExecutor()
    executor.run()

if __name__ == "__main__":
    # 5분마다 실행되도록 스케줄링 설정
    schedule.every(5).minutes.do(run_main_executor)

    # 프로그램 시작 시 한 번 실행
    run_main_executor()

    # 메인 쓰레드에서 스케줄러를 계속 실행
    while True:
        schedule.run_pending()
        time.sleep(1)