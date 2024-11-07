import os
import re
import time
import logging
import requests
import warnings  # FutureWarning 무시를 위해 추가
from collections import Counter
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
import schedule  # 스케줄링을 위해 추가

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
        valid_keywords = [kw for kw in keywords if re.match(r'^[0-9a-zA-Z가-힣]{2,}(?: [0-9a-zA-Z가-힣]{2,})*$', kw)]
        return valid_keywords

    def remove_substrings_and_duplicates(self, keywords):
        unique_keywords = []
        sorted_keywords = sorted(keywords, key=lambda x: len(x), reverse=True)
        for kw in sorted_keywords:
            if kw in unique_keywords:
                continue
            if not any((kw != other and kw in other) for other in unique_keywords):
                unique_keywords.append(kw)
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
            keywords = self.keyword_extractor.extract_keywords(combined_text, top_n=10)
            top_3_keywords = keywords[:3] if len(keywords) >= 3 else keywords

            logging.info(f"키워드 '{keyword}'에 대한 상위 3개 키워드: {top_3_keywords}")
            return top_3_keywords
        
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
    
# 대표 이슈 추출 클래스
class RepresentativeIssueExtractor:
    def __init__(self, stopwords, trending_keywords):
        self.stopwords = stopwords
        self.trending_keywords = trending_keywords
        self.keyword_extractor = KeywordExtractor(stopwords)

    def extract_representative_info(self, trees, source='Naver', max_words=5):
        trees_info = []
        used_keywords = set()
        for tree_info in trees:
            articles = tree_info['articles']
            importance = tree_info.get('importance', len(articles))
            keyword_counter = Counter()
            for news in articles:
                if 'keywords' in news:
                    keyword_counter.update(news['keywords'])
            if not keyword_counter:
                continue

            # 키워드 중요도 기반 정렬
            sorted_keywords = [kw for kw, _ in keyword_counter.most_common() if kw not in self.stopwords]

            # 대표 키워드 선택
            phrase_keywords = self.select_representative_keywords(sorted_keywords, used_keywords, max_words)
            if not phrase_keywords:
                continue

            phrase = ', '.join(phrase_keywords)
            used_keywords.update(phrase_keywords)

            combined_info = {
                'phrase': phrase,
                'importance': importance,
                'source': source,
                'all_keywords': set(phrase_keywords)
            }
            trees_info.append(combined_info)
            logging.info(f"Representative issue added: {phrase} - Importance: {importance} - Source: {source}")
        return trees_info

    def select_representative_keywords(self, sorted_keywords, used_keywords, max_words=5):
        phrase_keywords = []
        for kw in sorted_keywords:
            if kw not in used_keywords:
                phrase_keywords.append(kw)
            if len(phrase_keywords) >= max_words:
                break
        return phrase_keywords

# 메인 실행 클래스
class MainExecutor:
    def __init__(self):
        self.stopwords = StopwordLoader('stopwords-ko.txt').stopwords
        self.mongo_client = MongoDBConnection().client
        self.db = self.mongo_client['news_db']
        self.news_collection = self.db['news_articles']
        self.trees_collection = self.db['news_trees']
        self.final_issues_collection = self.db['final_issues']
        self.keyword_extractor = KeywordExtractor(self.stopwords)
        self.web_scraper = WebScraper()
        self.google_trends_manager = GoogleTrendsManager()
        self.naver_news_searcher = NaverNewsSearcher(self.keyword_extractor)
        self.issue_merger = IssueMerger()

    def run(self):
        self.current_time = datetime.now()
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

        sorted_keywords = sorted(keyword_volume, key=lambda x: -x[1])[:5]
        logging.info("검색량 기준으로 정렬된 상위 5개 트렌드 키워드:")
        for i, (kw, vol) in enumerate(sorted_keywords, 1):
            logging.info(f"{i}. {kw} - 검색량: {vol}")

        for keyword, volume in tqdm(sorted_keywords, desc="Google 트렌드 트리 생성"):
            if keyword in self.stopwords or volume <= 0:
                continue
            full_text = self.keyword_extractor.preprocess_text(keyword)
            keywords = self.keyword_extractor.extract_keywords(full_text, top_n=10)
            if not keywords:
                continue
            trend_trees.append({
                'articles': [{
                    'title': keyword,
                    'link': 'https://trends.google.com/trends/trendingsearches/daily?geo=KR',
                    'keywords': keywords
                }],
                'all_keywords': set([kw for kw in keywords if kw not in self.stopwords]),
                'google_trends_keywords': [keyword],
                'importance': volume
            })
            logging.info(f"Google 트렌드 새로운 트리 생성: {keyword} (검색량: {volume})")

        trend_top3_keywords = []
        logging.info("Google 트렌드 키워드를 네이버 뉴스에서 검색하여 상위 3개 키워드 추출 시작")
        for keyword, _ in tqdm(sorted_keywords, desc="Google 트렌드 키워드 네이버 뉴스 검색"):
            top3 = self.naver_news_searcher.search_news_with_keyword(keyword)
            if top3:
                trend_top3_keywords.append({
                    'trend_keyword': keyword,
                    'naver_keywords': top3,
                    'source': 'Google Trends (Naver Search)',
                    'importance': next((vol for kw, vol in keyword_volume if kw == keyword), 1)
                })

        combined_phrases = []
        for item in trend_top3_keywords:
            trend_kw = item['trend_keyword']
            naver_kws = item['naver_keywords']
            importance = item['importance']
            combined_set = [trend_kw] + naver_kws[:3]
            phrase = ', '.join(combined_set)
            phrase_keywords = self.keyword_extractor.remove_invalid_keywords(phrase.split(', '))
            phrase = ', '.join(phrase_keywords)
            if not phrase:
                continue
            combined_phrases.append({
                'phrase': phrase,
                'importance': importance,
                'source': 'Google Trends + Naver Search',
                'all_keywords': set(phrase_keywords)  # 'all_keywords' 키 추가
            })
            logging.info(f"조합된 이슈 구문 추가: {phrase}")

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

        word_extractor = WordExtractor()
        logging.info("WordExtractor 훈련 시작")
        word_extractor.train(articles_texts)
        logging.info("WordExtractor 훈련 완료")

        words = word_extractor.extract()
        logging.info(f"추출된 단어 수: {len(words)}")

        articles_keywords_list = []
        for idx, text in tqdm(enumerate(articles_texts), total=len(articles_texts), desc="뉴스 기사 키워드 추출"):
            keywords = self.keyword_extractor.extract_keywords(text, top_n=10)
            if keywords:
                articles_keywords_list.append(keywords)
            else:
                articles_keywords_list.append([])
            articles_metadata[idx]['keywords'] = keywords

        naver_trees = []
        for idx, news in enumerate(tqdm(articles_metadata, desc="네이버 트리 생성")):
            keywords = articles_keywords_list[idx]
            if not keywords:
                continue
            merged = False
            for tree_info in naver_trees:
                similarity = self.calculate_jaccard_similarity(
                    keywords, 
                    tree_info['all_keywords']
                )
                if similarity >= 0.2:
                    tree_info['articles'].append(news)
                    tree_info['all_keywords'].update([kw for kw in keywords if kw not in self.stopwords])
                    tree_info['contains_trend'] = tree_info['contains_trend'] or any(word in trending_keywords for word in keywords)
                    tree_info['importance'] += 1
                    merged = True
                    logging.info(f"네이버 트리 병합 완료: {news['title']} (유사도: {similarity:.2f})")
                    break
            if not merged:
                contains_trend = any(word in trending_keywords for word in keywords)
                naver_trees.append({
                    'articles': [news],
                    'all_keywords': set([kw for kw in keywords if kw not in self.stopwords]),
                    'contains_trend': contains_trend,
                    'importance': 1
                })
                logging.info(f"네이버 새로운 트리 생성: {news['title']} (트렌드 포함: {'예' if contains_trend else '아니오'})")

        logging.info(f"네이버 트리의 개수: {len(naver_trees)}")
        logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

        # 기존 트리 데이터베이스에 저장
        for tree_info in naver_trees:
            tree_doc = {
                'articles': tree_info['articles'],
                'all_keywords': list(tree_info['all_keywords']),
                'contains_trend': tree_info['contains_trend'],
                'importance': tree_info['importance'],
                'timestamp': self.current_time
            }
            self.trees_collection.insert_one(tree_doc)

        # 최근 24시간 내의 트리 정보 가져오기
        recent_trees = list(self.trees_collection.find({
            'timestamp': {'$gte': self.current_time - pd.Timedelta(hours=24)}
        }))

        representative_issue_extractor = RepresentativeIssueExtractor(self.stopwords, trending_keywords)
        naver_trees_info = representative_issue_extractor.extract_representative_info(recent_trees, source='Naver', max_words=5)
        trend_trees_info = representative_issue_extractor.extract_representative_info(trend_trees, source='Google Trends', max_words=5)

        sorted_naver_trees_info = sorted(
            naver_trees_info,
            key=lambda x: -x['importance']
        )

        sorted_trend_trees_info = sorted(
            trend_trees_info,
            key=lambda x: -x['importance']
        )

        # 상위 트렌드 이슈 선택
        top_trend_issues = sorted_trend_trees_info[:5]

        # 최종 이슈 리스트 생성: 네이버 이슈와 트렌드 이슈, 조합된 구문을 합침
        final_issues = sorted_naver_trees_info + top_trend_issues + combined_phrases

        # 기존의 최종 이슈들을 데이터베이스에서 가져오기 (삭제 없이 모든 이슈를 유지)
        existing_final_issues = list(self.final_issues_collection.find())

        # 기존 이슈들과 현재 이슈들을 병합
        existing_issues = [
            {
                'phrase': issue['phrase'],
                'importance': issue['importance'],
                'source': issue['source'],
                'all_keywords': set(issue.get('all_keywords', []))  # 리스트를 set으로 변환
            }
            for issue in existing_final_issues
        ]

        all_issues = existing_issues + final_issues

        # 유사도 기반 이슈 병합 (연결 요소 기반 병합)
        all_issues_merged = self.issue_merger.merge_similar_issues(all_issues, similarity_threshold=0.2)

        # 중요도 기준으로 상위 10개 이슈 추출
        top_10_issues = sorted(all_issues_merged, key=lambda x: -x['importance'])[:10]
        logging.info("상위 10개 이슈:")
        for idx, issue in enumerate(top_10_issues, 1):
            logging.info(f"{idx}. {issue['phrase']} (중요도: {issue['importance']})")

        # 키워드 중복이 많은 상위 10개 이슈 추출
        issue_keyword_counts = []
        for issue in all_issues_merged:
            overlap_count = sum(
                len(set(issue.get('all_keywords', set())) & set(other_issue.get('all_keywords', set())))
                for other_issue in all_issues_merged if issue != other_issue
            )
            issue_keyword_counts.append((issue, overlap_count))

        top_10_overlap_issues = sorted(issue_keyword_counts, key=lambda x: -x[1])[:10]
        logging.info("키워드 중복이 많은 상위 10개 이슈:")
        for idx, (issue, count) in enumerate(top_10_overlap_issues, 1):
            logging.info(f"{idx}. {issue['phrase']} (키워드 중복 수: {count})")

        # 키워드 중복이 많은 상위 10개 이슈 추출
        issue_keyword_counts = []
        for issue in all_issues_merged:
            overlap_count = sum(
                len(issue.get('all_keywords', set()) & other_issue.get('all_keywords', set()))
                for other_issue in all_issues_merged if issue != other_issue
            )
            issue_keyword_counts.append((issue, overlap_count))

        top_10_overlap_issues = sorted(issue_keyword_counts, key=lambda x: -x[1])[:10]
        logging.info("키워드 중복이 많은 상위 10개 이슈:")
        for idx, (issue, count) in enumerate(top_10_overlap_issues, 1):
            logging.info(f"{idx}. {issue['phrase']} (키워드 중복 수: {count})")

        # 새로운 이슈들을 데이터베이스에 추가 (삭제 없이)
        for issue in final_issues:
            try:
                issue_doc = {
                    'phrase': issue['phrase'],
                    'importance': issue['importance'],
                    'source': issue['source'],
                    'timestamp': datetime.now(),
                    'all_keywords': list(issue.get('all_keywords', set()))
                }
                # 중복 방지를 위해 upsert 사용
                self.final_issues_collection.update_one(
                    {'phrase': issue['phrase']},
                    {'$set': issue_doc},
                    upsert=True
                )
            except Exception as e:
                logging.error(f"이슈 저장 중 오류 발생: {e}")

    def calculate_jaccard_similarity(self, keywords1, keywords2):
        set1 = set(keywords1)
        set2 = set(keywords2)
        intersection = set1 & set2
        union = set1 | set2
        if not union:
            return 0.0
        return len(intersection) / len(union)

def run_main_executor():
    executor = MainExecutor()
    executor.run()

if __name__ == "__main__":
    # 20분마다 실행되도록 스케줄링 설정
    schedule.every(20).minutes.do(run_main_executor)

    # 프로그램 시작 시 한 번 실행
    run_main_executor()

    # 메인 쓰레드에서 스케줄러를 계속 실행
    while True:
        schedule.run_pending()
        time.sleep(1)
