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

from openai import OpenAI

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 형태소 분석기 초기화 (Komoran)
komoran = Komoran()

# 불용어 리스트를 파일에서 읽어오기
def load_stopwords(file_path):
    default_stopwords = set([
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

# Google 트렌드 키워드 수집 함수
def get_google_trends_keywords():
    try:
        pytrends = TrendReq(hl='ko', tz=540)
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
    text = re.sub(r'[^0-9가-힣a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# N-그램 기반 키워드 추출 함수
def extract_ngrams(tokens, n=2):
    if len(tokens) < n:
        return []
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

# 대표 키워드 선정 함수
def select_representative_keyword(top_keywords, used_keywords, google_trends_keywords):
    for kw in top_keywords:
        if kw in google_trends_keywords and kw not in used_keywords and kw not in stopwords:
            return kw
    for kw in top_keywords:
        if kw not in used_keywords and kw not in stopwords:
            return kw
    return None

# 중복 키워드 제거 함수
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
    valid_keywords = [kw for kw in keywords if re.match(r'^[0-9a-zA-Z가-힣]{2,}(?: [0-9a-zA-Z가-힣]{2,})*$', kw)]
    return valid_keywords

# 트리 별로 대표 이슈 추출 함수
def extract_representative_info(trees, source='Naver', max_words=3):
    trees_info = []
    used_keywords = set()
    for tree_info in trees:
        articles = tree_info['articles']
        importance = tree_info.get('importance', len(articles))
        keyword_counter = Counter()
        for news in articles:
            if 'keywords' in news:
                keyword_counter.update([kw for kw in news.get('keywords', []) if kw not in stopwords])
            else:
                continue
        top_keywords = [word for word, freq in keyword_counter.most_common(5)]
        if not top_keywords:
            continue
        rep_keyword = select_representative_keyword(top_keywords, used_keywords, tree_info.get('google_trends_keywords', []))
        if not rep_keyword:
            rep_keyword = top_keywords[0] if top_keywords else None
        if not rep_keyword:
            continue
        used_keywords.add(rep_keyword)
        top_other_keywords = [kw for kw in top_keywords if kw != rep_keyword]
        if top_other_keywords:
            phrase_keywords = [rep_keyword] + top_other_keywords[:max_words-1]
            phrase_keywords = remove_substrings_and_duplicates(phrase_keywords)
            phrase_keywords = remove_invalid_keywords(phrase_keywords)
            if len(phrase_keywords) > max_words:
                phrase_keywords = phrase_keywords[:max_words]
            if len(phrase_keywords) < 2:
                phrase_keywords = [rep_keyword]
            phrase = ', '.join(phrase_keywords)
        else:
            phrase = rep_keyword

        combined_info = {
            'phrase': phrase,
            'importance': importance,
            'source': source
        }
        trees_info.append(combined_info)
        logging.info(f"Representative issue added: {phrase} - Importance: {importance} - Source: {source}")
    return trees_info

# 키워드 추출 함수
def extract_keywords(text, stopwords, top_n=10):
    try:
        tokens = komoran.nouns(text)
        filtered_tokens = [word for word in tokens if 2 <= len(word) <= 3 and word not in stopwords]
        token_counts = Counter(filtered_tokens)
        top_keywords = [word for word, count in token_counts.most_common(top_n)]
        bigrams = extract_ngrams(filtered_tokens, n=2)
        bigram_counts = Counter([bg for bg in bigrams if bg not in stopwords and len(bg.replace(' ', '')) >= 2])
        sorted_bigrams = [bg for bg, cnt in bigram_counts.most_common(top_n)]
        combined_keywords = top_keywords + sorted_bigrams
        unique_keywords = []
        combined_keywords_sorted = sorted(combined_keywords, key=lambda x: len(x), reverse=True)
        for kw in combined_keywords_sorted:
            if kw in unique_keywords:
                continue
            if not any((kw != other and kw in other) for other in unique_keywords):
                unique_keywords.append(kw)
            if len(unique_keywords) >= top_n:
                break
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

        news_elements = soup.select('.list_news .news_area')
        news_items = []
        for elem in news_elements:
            title_elem = elem.select_one('.news_tit')
            if title_elem:
                title = title_elem.get_text(strip=True)
                link = title_elem.get('href')
                news_items.append({'title': title, 'link': link})

        articles_texts = []
        for news in news_items[:10]:
            article_text = scrape_webpage(news['link'])
            if article_text:
                full_text = preprocess_text(news['title'] + ' ' + article_text)
                if full_text:
                    articles_texts.append(full_text)

        if not articles_texts:
            logging.warning(f"네이버 뉴스 검색 결과에서 텍스트를 추출할 수 없습니다: {keyword}")
            return []

        keywords = extract_keywords(' '.join(articles_texts), stopwords, top_n=10)
        top_3_keywords = keywords[:3] if len(keywords) >= 3 else keywords

        logging.info(f"키워드 '{keyword}'에 대한 상위 3개 키워드: {top_3_keywords}")
        return top_3_keywords
    except Exception as e:
        logging.error(f"네이버 뉴스 검색 중 오류 발생 ({keyword}): {e}")
        return []

# 연결 요소 기반 이슈 병합 함수
def merge_similar_issues(issues, similarity_threshold=0.3):
    if not issues:
        return []

    phrases = [issue['phrase'] for issue in issues]
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '), lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(phrases)

    similarity_matrix = cosine_similarity(tfidf_matrix)

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

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= similarity_threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    merged_issues = []
    for group in groups.values():
        if not group:
            continue
        if len(group) == 1:
            merged_issues.append(issues[group[0]])
        else:
            combined_phrases = [issues[idx]['phrase'] for idx in group]
            combined_importance = sum([issues[idx]['importance'] if issues[idx]['importance'] else 0 for idx in group])
            combined_sources = set([issues[idx]['source'] for idx in group])
            words = set(', '.join(combined_phrases).split(', '))
            words = remove_substrings_and_duplicates(words)
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

# 메인 함수
def main():
    trending_keywords = get_google_trends_keywords()
    logging.info(f"Google 트렌드 키워드 수집 완료: {len(trending_keywords)}개")

    keyword_volume = []
    trend_trees = []

    logging.info("Google 트렌드 검색량 수집 시작")
    for keyword in tqdm(trending_keywords, desc="Google 트렌드 키워드 처리"):
        if keyword in stopwords:
            logging.info(f"불용어 키워드 스킵: {keyword}")
            continue
        try:
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

    sorted_keywords = sorted(keyword_volume, key=lambda x: x[1], reverse=True)[:5]
    logging.info("검색량 기준으로 정렬된 상위 5개 트렌드 키워드:")
    for i, (kw, vol) in enumerate(sorted_keywords, 1):
        logging.info(f"{i}. {kw} - 검색량: {vol}")

    for keyword, volume in tqdm(sorted_keywords, desc="Google 트렌드 트리 생성"):
        if keyword in stopwords or volume <= 0:
            continue
        full_text = preprocess_text(keyword)
        keywords = extract_keywords(full_text, stopwords, top_n=10)
        if not keywords:
            continue
        trend_trees.append({
            'articles': [{
                'title': keyword,
                'link': 'https://trends.google.com/trends/trendingsearches/daily?geo=KR',
                'keywords': keywords
            }],
            'all_keywords': set([kw for kw in keywords if kw not in stopwords]),
            'google_trends_keywords': [keyword],
            'importance': volume
        })
        logging.info(f"Google 트렌드 새로운 트리 생성: {keyword} (검색량: {volume})")

    logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    trend_top3_keywords = []
    logging.info("Google 트렌드 키워드를 네이버 뉴스에서 검색하여 상위 3개 키워드 추출 시작")
    for keyword, _ in tqdm(sorted_keywords, desc="Google 트렌드 키워드 네이버 뉴스 검색"):
        top3 = search_naver_news_with_keyword(keyword, stopwords)
        if top3:
            trend_top3_keywords.append({
                'trend_keyword': keyword,
                'naver_keywords': top3,
                'source': 'Google Trends (Naver Search)',
                'importance': next((vol for kw, vol in keyword_volume if kw == keyword), 1)
            })

    logging.info(f"Google 트렌드 키워드 기반 추출된 상위 3개 키워드의 총 개수: {len(trend_top3_keywords) * 3}")

    combined_phrases = []
    for item in trend_top3_keywords:
        trend_kw = item['trend_keyword']
        naver_kws = item['naver_keywords']
        importance = item['importance']
        combined_set = [trend_kw] + naver_kws[:3]
        phrase = ', '.join(combined_set)
        phrase_keywords = remove_invalid_keywords(phrase.split(', '))
        phrase = ', '.join(phrase_keywords)
        if not phrase:
            continue
        combined_phrases.append({
            'phrase': phrase,
            'importance': importance,
            'source': 'Google Trends + Naver Search'
        })
        logging.info(f"조합된 이슈 구문 추가: {phrase}")

    site_url = "https://news.naver.com/main/ranking/popularDay.naver"

    # 브라우저 옵션 설정
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
        news_items = [{'title': elem.text, 'link': elem.get_attribute('href')} for elem in news_elements]
        logging.info(f"수집된 뉴스 기사 수: {len(news_items)}")

    articles_texts = []
    articles_metadata = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_news = {executor.submit(scrape_webpage, news['link']): news for news in news_items}
        for future in tqdm(as_completed(future_to_news), total=len(future_to_news), desc="뉴스 기사 스크래핑"):
            news = future_to_news[future]
            try:
                article_text = future.result()
                if article_text:
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

    word_extractor = WordExtractor()
    logging.info("WordExtractor 훈련 시작")
    word_extractor.train(articles_texts)
    logging.info("WordExtractor 훈련 완료")

    words = word_extractor.extract()
    logging.info(f"추출된 단어 수: {len(words)}")

    articles_keywords_list = []
    for idx, text in tqdm(enumerate(articles_texts), total=len(articles_texts), desc="뉴스 기사 키워드 추출"):
        keywords = extract_keywords(text, stopwords, top_n=10)
        if keywords:
            articles_keywords_list.append(keywords)
        else:
            articles_keywords_list.append([])
        articles_metadata[idx]['keywords'] = keywords

    naver_trees = []  # 네이버 뉴스 트리 정보의 리스트

    for idx, news in enumerate(tqdm(articles_metadata, desc="네이버 트리 생성")):
        keywords = articles_keywords_list[idx]
        if not keywords:
            continue
        merged = False
        for tree_info in naver_trees:
            similarity = calculate_jaccard_similarity(keywords, tree_info['all_keywords'])
            logging.debug(f"Comparing '{news['title']}' with tree keywords: {tree_info['all_keywords']} | Similarity: {similarity}")
            if similarity >= 0.2:
                tree_info['articles'].append(news)
                tree_info['all_keywords'].update([kw for kw in keywords if kw not in stopwords])
                tree_info['importance'] += 1
                merged = True
                logging.info(f"네이버 트리 병합 완료: {news['title']} (유사도: {similarity:.2f})")
                break
        if not merged:
            contains_trend = any(word in trending_keywords for word in keywords)
            naver_trees.append({
                'articles': [news],
                'all_keywords': set([kw for kw in keywords if kw not in stopwords]),
                'contains_trend': contains_trend,
                'importance': 1
            })
            logging.info(f"네이버 새로운 트리 생성: {news['title']} (트렌드 포함: {'예' if contains_trend else '아니오'})")

    logging.info(f"네이버 트리의 개수: {len(naver_trees)}")
    logging.info(f"Google 트렌드 트리의 개수: {len(trend_trees)}")

    naver_trees_info = extract_representative_info(naver_trees, source='Naver', max_words=3)
    trend_trees_info = extract_representative_info(trend_trees, source='Google Trends', max_words=3)

    sorted_naver_trees_info = sorted(naver_trees_info, key=lambda x: -x['importance'])
    sorted_trend_trees_info = sorted(trend_trees_info, key=lambda x: -x['importance'])

    top_naver_issues = sorted_naver_trees_info[:6]
    final_issues = top_naver_issues + combined_phrases

    final_issues = merge_similar_issues(final_issues, similarity_threshold=0.3)
    final_issues = merge_similar_issues(final_issues, similarity_threshold=0.2)

    if len(final_issues) < 10:
        for item in sorted_naver_trees_info[6:]:
            final_issues.append(item)
            if len(final_issues) >= 10:
                break

    final_issues = final_issues[:10]

    # 최종 이슈 출력
    print("\n실시간 이슈:")
    print("\n네이버 뉴스 상위 6개 이슈:")
    for rank, item in enumerate(top_naver_issues, 1):
        print(f"{rank}. {item['phrase']}")

    print("\nGoogle 트렌드 키워드 기반 네이버 뉴스 상위 3개 키워드의 조합:")
    for rank, item in enumerate(combined_phrases, 1):
        print(f"{rank}. {item['phrase']}")

    print("\n전체 실시간 이슈 (Google 트렌드 + 네이버 상위 6개 + 조합된 구문):")
    for rank, item in enumerate(final_issues, 1):
        print(f"{rank}. {item['phrase']}")

# OpenAI API 키 설정
client = OpenAI()

def summarize_article_content(content):
    prompt = f"""
    
    1. 다음 10개의 키워드를 보고 3~5어절의 키워드로 요약해줘
    2. 여러가지의 키워드가 합쳐져 있으면 두 개의 키워드로 분리해도 돼
    예시 ) 국정 감사, 국회 운영, 대통령 관저, 대통령 다혜, 대통령 명태, 명태균, 문재인 대통령, 여론 조사, 윤석열 대통령, 정진석 대통령, 참고인 조사
    --> 1. 국정 감사 및 여론
        2. 문재인 전 대통령, 다혜, 정진석

예시)

1. 대통령 직무, 부정 평가, 긍정 평가
2. 불법 영업, 사생활 논란, 음식점 운영, 트리플 스타, 트리플스타, 흑백 요리사
3. 국정 감사, 국회 운영, 대통령 관저, 대통령 다혜, 대통령 명태, 명태균, 문재인 대통령, 여론 조사, 윤석열 대통령, 정진석 대통령, 참고인 조사
4. 아버지 살해, 아버지 둔기, 30대 남성
5. 소말리, 소녀상 모욕, 편의점 난동, 조니 말리
6. 23기 정숙, 출연자 검증, 논란 제작진, 유튜브 채널
7. GD, 베이비 몬스터, 더블 타이틀, 몬스터 정규
8. 기아 타이, 타이 거즈, 기아 세일
9. 테슬라 코리아, 김예지 국내, 최초 테슬라
10. 북한군 교전, 북한군 추정, 주장 북한군

1. 대통령 직무 평가
2. 흑백 요리사, 불법 영업 논란
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful news letter artist that summarizes keywords to news keywords for SNS."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.15,
        )

        summary_text = response.choices[0].message.content.strip()
        return summary_text

    except Exception as e:
        print(f"요약 생성 중 오류 발생: {e}")
        return {
            'headline': "Failed to summarize.",
            'summary': "Failed to summarize.",
            'hashtags': [],
            'title_ideas': []
        }

if __name__ == "__main__":
    main()
