import os
import re
import time
import logging
import requests
import hashlib
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from pymongo import ReturnDocument
from bs4 import BeautifulSoup
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
import schedule
import sys

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

# 설정을 관리하는 클래스
class SettingsManager:
    _instance = None

    def __new__(cls, db):
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
            cls._instance.db = db
        return cls._instance

    def get_last_reset_time(self):
        settings = self.db['settings'].find_one({'setting': 'last_reset_time'})
        if settings:
            return settings['value']
        else:
            return None

    def update_last_reset_time(self, new_time):
        self.db['settings'].update_one(
            {'setting': 'last_reset_time'},
            {'$set': {'value': new_time}},
            upsert=True
        )

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

# 사전 기반 매핑 사전 정의
alias_mapping = {
    "도널드 트럼프 대통령": "트럼프 대통령",
    "트럼프 대통령": "트럼프 대통령",
    "도널드 트럼프": "트럼프 대통령",
    "트럼프": "트럼프 대통령",
    # 추가적인 매핑을 여기에 추가하세요
}

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
            unique_keywords = self.normalize_keywords(unique_keywords)  # 정규화 추가
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

    def normalize_keywords(self, keywords):
        normalized = []
        for kw in keywords:
            normalized_kw = alias_mapping.get(kw, kw)  # 매핑 사전에서 변환
            normalized.append(normalized_kw)
        return normalized

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
            articles_metadata = []

            for news in news_items[:10]:
                # 기사 본문 스크래핑
                # 네이버 뉴스라면 scrape_webpage, 아니면 scrape_webpage_for_google_search 사용
                if 'news.naver.com' in news['link']:
                    article_text = self.web_scraper.scrape_webpage(news['link'])
                else:
                    article_text = self.web_scraper.scrape_webpage_for_google_search(news['link'])
                
                if article_text:
                    full_text = self.keyword_extractor.preprocess_text(news['title'] + ' ' + news['description'] + ' ' + article_text)
                    if full_text:
                        articles_texts.append(full_text)
                        articles_metadata.append({'title': news['title'], 'link': news['link'], 'keywords': []})
                else:
                    # 기사 본문을 가져오지 못한 경우에도 제목과 요약을 사용
                    full_text = self.keyword_extractor.preprocess_text(news['title'] + ' ' + news.get('description', ''))
                    if full_text:
                        articles_texts.append(full_text)
                        articles_metadata.append({'title': news['title'], 'link': news['link'], 'keywords': []})
                    logging.warning(f"기사 본문이 없습니다: {news['title']}")

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

# 트리 생성 및 관리 클래스
class TreeBuilder:
    def __init__(self, articles_metadata, articles_keywords_list):
        self.articles_metadata = articles_metadata
        self.articles_keywords_list = articles_keywords_list

    def build_trees(self, similarity_threshold=0.2):
        trees = []
        for idx, news in enumerate(tqdm(self.articles_metadata, desc="트리 생성")):
            keywords = self.articles_keywords_list[idx]
            if not keywords:
                continue
            merged = False
            for tree_info in trees:
                similarity = self.calculate_jaccard_similarity(
                    keywords, 
                    tree_info['all_keywords']
                )
                if similarity >= similarity_threshold:
                    tree_info['articles'].append(news)
                    tree_info['all_keywords'].update(keywords)
                    tree_info['importance'] += 1
                    merged = True
                    logging.info(f"트리 병합 완료: {news['title']} (유사도: {similarity:.2f})")
                    break
            if not merged:
                trees.append({
                    'articles': [news],
                    'all_keywords': set(keywords),
                    'importance': 1
                })
                logging.info(f"새로운 트리 생성: {news['title']}")

        return trees

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

    def extract_tree_keywords(self, trees):
        # 각 트리에서 중요 키워드를 추출하여 트리의 계층 구조 생성
        tree_structures = []
        for tree in trees:
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

    # 트리 병합 메서드
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

# 트리 병합 및 저장 로직이 포함된 MainExecutor 클래스
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
        # 설정을 저장할 컬렉션
        self.settings_collection = self.db['settings']
        # 설정 관리자 초기화
        self.settings_manager = SettingsManager(self.db)
        self.keyword_extractor = KeywordExtractor(self.stopwords)
        self.web_scraper = WebScraper()
        self.google_trends_manager = GoogleTrendsManager()
        self.naver_news_searcher = NaverNewsSearcher(self.keyword_extractor)
        self.issue_merger = IssueMerger()

        # 고유 인덱스 생성 (기존 인덱스 삭제 후 재생성)
        self.create_unique_index()

    def create_unique_index(self):
        try:
            # 기존 고유 인덱스 삭제
            self.tree_structures_collection.drop_index('unique_tree_key')
            logging.info("기존 고유 인덱스 'unique_tree_key' 삭제 완료.")
        except Exception as e:
            if "not found" in str(e):
                logging.info("기존 고유 인덱스 'unique_tree_key'가 존재하지 않습니다.")
            else:
                logging.error(f"기존 고유 인덱스 삭제 중 오류 발생: {e}")

        try:
            # 새 고유 인덱스 생성
            self.tree_structures_collection.create_index(
                [('unique_key', ASCENDING)],
                unique=True,
                name='unique_tree_key'
            )
            logging.info("고유 인덱스 'unique_tree_key' 생성 완료.")
        except Exception as e:
            logging.error(f"고유 인덱스 생성 중 오류 발생: {e}")

    def generate_unique_key(self, root):
        # 키워드 정규화: 소문자 변환, 공백 및 특수문자 제거
        root_normalized = re.sub(r'[^0-9가-힣a-z]', '', root.strip().lower())
        
        # 해시 생성 (SHA256 사용)
        unique_key = hashlib.sha256(root_normalized.encode('utf-8')).hexdigest()
        
        logging.debug(f"생성된 unique_key: {unique_key} (원본: {root_normalized})")
        return unique_key

    def insert_or_update_tree(self, tree_doc, delta_importance_dict):
        try:
            # find_one_and_update를 사용하여 이전 importance 값을 가져오고 업데이트
            updated_document = self.tree_structures_collection.find_one_and_update(
                {'unique_key': tree_doc['unique_key']},
                {
                    '$inc': {'importance': tree_doc['importance']},
                    '$set': {
                        'timestamp': tree_doc['timestamp'],
                        'last_update_time': tree_doc['last_update_time']
                    },
                    '$setOnInsert': {
                        'root': tree_doc['root']
                    },
                    '$addToSet': {  # 중복 없이 depth1과 depth2 키워드 추가
                        'depth1': {'$each': tree_doc['depth1']},
                        'depth2': {'$each': tree_doc['depth2']}
                    }
                },
                upsert=True,
                return_document=ReturnDocument.BEFORE  # 이전 문서를 반환
            )

            if updated_document:
                previous_importance = updated_document.get('importance', 0)
                delta = tree_doc['importance']
                delta_importance_dict[tree_doc['unique_key']] = delta
                logging.info(f"트리 업데이트 완료: ROOT - {tree_doc['root']} (추가된 Importance: {delta})")
            else:
                # 새로 삽입된 경우
                delta_importance_dict[tree_doc['unique_key']] = tree_doc['importance']
                logging.info(f"새로운 트리 저장 완료: ROOT - {tree_doc['root']} (Importance: {tree_doc['importance']})")
        except DuplicateKeyError:
            logging.error(f"DuplicateKeyError 발생: unique_key={tree_doc['unique_key']}")
        except Exception as e:
            logging.error(f"트리 저장 중 오류 발생: {e}")

    def reset_database_if_needed(self, importance_reset_interval):
        self.current_time = datetime.now()
        last_reset_time = self.settings_manager.get_last_reset_time()
        if last_reset_time:
            logging.info(f"마지막 초기화 시간: {last_reset_time}")
        else:
            last_reset_time = self.current_time
            self.settings_manager.update_last_reset_time(last_reset_time)
            logging.info("초기화 시간 설정 완료.")

        time_diff = (self.current_time - last_reset_time).total_seconds() / 3600  # 시간 단위
        logging.info(f"현재 시간: {self.current_time}, 마지막 초기화 이후 {time_diff:.2f}시간 경과.")

        if time_diff >= importance_reset_interval:
            # DB 초기화 (tree_structures_collection 삭제)
            self.tree_structures_collection.delete_many({})
            logging.info("tree_structures_collection 초기화 완료.")

            # 마지막 초기화 시간 업데이트
            self.settings_manager.update_last_reset_time(self.current_time)
            logging.info("마지막 초기화 시간 업데이트 완료.")
            return True
        else:
            logging.info(f"초기화 필요 없음. 마지막 초기화 이후 {time_diff:.2f}시간 경과.")
            return False

    # 키워드 요약 함수 (OpenAI API 이용)
    def summarize_keywords(self, content):
        prompt = f"""
[요약하는 방법]
1. 아래에 나열된 여러 개의 키워드를 기반으로 핵심 키워드 10개를 2어절 이내로 요약해줘.
2. 유사하거나 중복된 내용의 키워드는 하나의 항목으로 병합해줘.
- 예: "대통령 직무, 부정 평가, 긍정 평가" -> "대통령 직무 평가"
3. 정치에 관한 내용은 최대 2개로 제한하며, 비슷한 내용은 합치지 말고 제외해줘.
4. 각 요약 항목은 명확하고 간결해야 하며, 핵심 주제를 반영해야 해.
5. 동일한 키워드가 여러 번 등장하면 한 번만 포함시켜줘.
6. 최신의 실제 이슈를 반영하여 현재 상황에 맞는 용어와 표현을 사용해줘.
7. 넘버링은 하지마.

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
대통령 직무 평가
불법 영업 논란
국정 감사 및 여론
아버지 둔기로 살해
소녀상 모욕 사건
23기 정숙, 출연자 검증 논란
GD와 베이비 몬스터
기아타이거즈 세일
테슬라, 김예지
북한군 교전 주장

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
14. 사이버트럭, 테슬라 전기 픽업 트럭, 사이버 트럭 지드래곤, 테슬라 사이버트럭
15. 김하온, 블록 토너먼트, 오리지널 예능, 티빙 오리지널
16. 원내대표 대통령, 국민 원내대표, 추경호 국민, 원내 대책, 대책 회의
17. 아버지 시신 냉동, 이혼 소송 진행, 아버지 상태, 아버지 사망, 경찰
18. 최고 위원, 국민 담화, 여사 판단, 판단 변화, 변화 작동
19. 추경호 원내대표, 국민 소통 기회, 기자회견 최종, 대통령 국민, 말씀
20. 온라인 커뮤니티, 커뮤니티 갈무리, 사고 당시 모습, 추돌 사고, 여성

요약된 실시간 이슈:
미국 대선 후보
북한, 탄도 미사일 시험 발사
통영 해산물 축제
강남 추돌 사고 
윤석열 대통령 기자회견
한국 부채춤 중국 논란
국민의 힘, 원내대표 대책 회의
아버지 시신 냉동 사건
금투세 금융 투자 소득세
지드래곤, 테슬라 사이버트럭

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

    def run(self):
        self.current_time = datetime.now()
        importance_reset_interval = 12  # 중요도 초기화 간격 (시간 단위)

        # 초기화 여부 확인 및 수행
        self.reset_database_if_needed(importance_reset_interval)

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
                            articles_metadata.append({'title': news['title'], 'link': news['link'], 'keywords': []})
                    else:
                        # 기사 본문을 가져오지 못한 경우에도 제목과 요약을 사용
                        full_text = self.keyword_extractor.preprocess_text(news['title'] + ' ' + news.get('description', ''))
                        if full_text:
                            articles_texts.append(full_text)
                            articles_metadata.append({'title': news['title'], 'link': news['link'], 'keywords': []})
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
        for idx, text in enumerate(tqdm(articles_texts, desc="뉴스 기사 키워드 추출")):
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
                # 네이버 뉴스라면 scrape_webpage, 아니면 scrape_webpage_for_google_search 사용
                if 'news.naver.com' in news['link']:
                    article_text = self.web_scraper.scrape_webpage(news['link'])
                else:
                    article_text = self.web_scraper.scrape_webpage_for_google_search(news['link'])

                if article_text:
                    full_text = self.keyword_extractor.preprocess_text(news['title'] + ' ' + article_text)
                    if full_text:
                        trend_articles_texts.append(full_text)
                        trend_articles_metadata.append({'title': news['title'], 'link': news['link'], 'keywords': []})
                else:
                    # 기사 본문을 가져오지 못한 경우에도 제목을 사용
                    full_text = self.keyword_extractor.preprocess_text(news['title'])
                    if full_text:
                        trend_articles_texts.append(full_text)
                        trend_articles_metadata.append({'title': news['title'], 'link': news['link'], 'keywords': []})
                    logging.warning(f"기사 본문이 없습니다: {news['title']}")

        # 키워드 추출
        trend_articles_keywords_list = []
        for idx, text in enumerate(tqdm(trend_articles_texts, desc="구글 트렌드 기사 키워드 추출")):
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
            tree['importance'] *= 0.7  # 가중치 조정

        # 두 트리 리스트를 합침
        combined_tree_structures = tree_structures + trend_tree_structures

        # 사전 병합: 동일한 unique_key를 가진 트리를 합산
        merged_tree_dict = {}
        for tree in combined_tree_structures:
            unique_key = self.generate_unique_key(tree['root'])  # root만 전달
            if unique_key in merged_tree_dict:
                merged_tree_dict[unique_key]['importance'] += tree['importance']
                # depth1과 depth2 키워드 병합
                merged_tree_dict[unique_key]['depth1'].extend(tree['depth1'])
                merged_tree_dict[unique_key]['depth2'].extend(tree['depth2'])
            else:
                merged_tree_dict[unique_key] = tree

        logging.info(f"사전 병합 후 고유 트리의 개수: {len(merged_tree_dict)}")

        # 트리 삽입/업데이트 및 delta_importance_dict 생성
        delta_importance_dict = {}
        for unique_key, tree in merged_tree_dict.items():
            tree_doc = {
                'root': tree['root'],
                'depth1': sorted(list(set([kw.strip() for kw in tree['depth1']]))),
                'depth2': sorted(list(set([kw.strip() for kw in tree['depth2']]))),
                'importance': tree['importance'],
                'timestamp': tree['timestamp'],
                'last_update_time': tree['timestamp'],
                'unique_key': unique_key  # 고유 키 추가
            }

            self.insert_or_update_tree(tree_doc, delta_importance_dict)

        # 현재 중요도 기준 상위 16개 트리
        top_16_trees = list(self.tree_structures_collection.find().sort('importance', -1).limit(16))

        # delta_importance_dict을 기반으로 급상승한 상위 4개 트리 선택
        sorted_delta = sorted(delta_importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_4_rising_keys = [item[0] for item in sorted_delta[:4]]

        # 상위 4개 급상승 트리 가져오기
        top_4_rising_trees = list(self.tree_structures_collection.find({'unique_key': {'$in': top_4_rising_keys}}))

        # 중복 제거: 상위 16개에 이미 포함된 트리 제외
        top_16_keys = set(tree['unique_key'] for tree in top_16_trees)
        top_4_rising_trees = [tree for tree in top_4_rising_trees if tree['unique_key'] not in top_16_keys]

        # 최종 상위 20개 트리: 상위 16개 + 급상승 4개
        final_top_trees = top_16_trees + top_4_rising_trees[:4]  # 최대 4개 추가

        logging.info("최종 상위 20개 트리 선정 완료.")

        # 키워드 요약 호출
        summarized_keywords = self.summarize_keywords(final_top_trees)
        print("\n요약된 실시간 이슈:")
        print(summarized_keywords)

        current_time_str = datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f"한국_실시간_이슈_{current_time_str}.txt"

        # 글로벌 실시간 이슈를 텍스트 파일로 저장
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write("한국 실시간 이슈 :\n")
                f.write(f"{summarized_keywords}\n")
            logging.info(f"한국 실시간 이슈가 {file_name} 파일에 저장되었습니다.")
            print(f"\n한국 실시간 이슈가 {file_name} 파일에 저장되었습니다.")
        except Exception as e:
            logging.error(f"한국 실시간 이슈 저장 중 오류 발생: {e}")
            print("\n한국 실시간 이슈 저장에 실패했습니다.")

# 트리 병합 및 저장 로직이 포함된 MainExecutor 클래스
def run_main_executor():
    executor = MainExecutor()
    executor.run()

# 데이터 클린업 스크립트
def cleanup_duplicates():
    mongo_client = MongoDBConnection().client
    db = mongo_client['news_db']
    collection = db['tree_structures']

    pipeline = [
        {"$group": {
            "_id": "$unique_key",
            "ids": {"$addToSet": "$_id"},
            "count": {"$sum": 1}
        }},
        {"$match": {
            "count": {"$gt": 1}
        }}
    ]

    duplicates = list(collection.aggregate(pipeline))
    logging.info(f"중복된 unique_key 수: {len(duplicates)}")

    for dup in duplicates:
        # 첫 번째 문서를 제외한 나머지 삭제
        ids_to_delete = dup['ids'][1:]
        result = collection.delete_many({"_id": {"$in": ids_to_delete}})
        logging.info(f"unique_key={dup['_id']}인 중복 문서 {result.deleted_count}개 삭제 완료.")

    logging.info("중복 데이터 클린업 완료.")

# 프로그램 실행
if __name__ == "__main__":
    # 먼저 데이터 클린업을 수행하여 기존 중복 데이터를 제거합니다.
    cleanup_duplicates()

    # 스케줄러 설정: 1시간마다 실행
    schedule.every(1).hour.do(run_main_executor)

    # 프로그램 시작 시 한 번 실행
    run_main_executor()

    # 메인 쓰레드에서 스케줄러를 계속 실행
    while True:
        schedule.run_pending()
        time.sleep(1)
