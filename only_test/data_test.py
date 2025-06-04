import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set

class ImprovedConsultationKeywordExtractor:
    def __init__(self):
        self.topic_change_patterns = []
        self.topic_keywords = {}
        self.consulting_categories = set()
        
        # 상담 공통어 (제외할 단어들)
        self.common_consultation_words = {
            # 인사말/예의
            '안녕하세요', '감사합니다', '죄송합니다', '수고하셨습니다', '안녕히계세요',
            # 시간 표현
            '지금', '오늘', '내일', '어제', '이제', '나중에', '잠시만', '잠깐',
            # 확인/응답
            '네', '예', '아니요', '맞습니다', '알겠습니다', '그렇습니다', '됩니다',
            # 접속사/전환어
            '그런데', '그러면', '그럼', '그리고', '혹시', '또', '그러니까', '근데',
            # 지시어
            '이것', '그것', '저것', '이게', '그게', '저게', '여기', '거기', '저기',
            # 일반적 표현
            '본인', '고객님', '손님', '상담원', '저희', '우리', '제가', '저는',
            '입니다', '있습니다', '없습니다', '하겠습니다', '드리겠습니다',
            '어떻게', '뭐', '무엇', '언제', '어디서', '왜', '어떤',
            # 숫자/단위
            '원', '개', '번', '회', '일', '월', '년', '시', '분',
            # 동작 관련 일반어
            '보다', '하다', '되다', '있다', '없다', '말하다', '이야기하다',
            # 대기/처리 관련
            '기다려', '기다려주시겠습니까', '처리', '진행', '확인',
            # 기타 불용어
            '거예요', '거죠', '거고요', '이었습니다', '였습니다', '주세요', '드립니다'
        }
        
        # 카테고리별 특화 키워드 사전 (실제 업무 용어들)
        self.category_specific_terms = {
            '도난/분실': ['도난', '분실', '잃어버렸', '없어졌', '털렸', '사라졌', '찾을수없', '훔쳐갔'],
            '이용내역': ['사용내역', '이용내역', '거래내역', '결제내역', '사용처', '가맹점', '승인번호'],
            '결제대금': ['청구서', '청구금액', '납부', '결제일', '출금일', '연체', '미납', '잔액부족'],
            '오토할부': ['오토할부', '자동할부', '무이자할부', '할부개월', '할부수수료'],
            '캐시백': ['캐시백', '적립', '포인트', '리워드', '혜택금액'],
            '장기카드대출': ['카드론', '현금서비스', '대출한도', '이자율', '상환', '원리금'],
            '선결제': ['선결제', '선납', '미리결제', '사전결제'],
            '즉시출금': ['즉시출금', '실시간출금', '바로출금', '당일출금'],
            '바우처': ['바우처', '상품권', '쿠폰', '할인권', '이용권'],
            '매출구분': ['매출구분', '일시불', '할부', '개월변경', '결제방법변경'],
            '입금': ['입금', '송금', '계좌이체', '무통장입금', '입금확인'],
            '포인트': ['포인트', '마일리지', '적립금', '리워드포인트'],
            '마일리지': ['마일리지', '항공마일', '마일적립', '마일사용'],
            '가상계좌': ['가상계좌', '입금계좌', '전용계좌', '개인별계좌'],
            '이벤트': ['이벤트', '프로모션', '특가', '할인행사', '혜택이벤트'],
            '교육비': ['교육비', '학비', '등록금', '수강료', '교육기관'],
            '승인취소': ['승인취소', '결제취소', '거래취소', '매출취소'],
            '매출취소': ['매출취소', '거래취소', '결제무효', '승인취소'],
            '결제일': ['결제일', '출금일', '납부일', '청구일', '결제예정일'],
            '한도': ['한도', '이용한도', '결제한도', '현금서비스한도', '신용한도'],
            '한도상향': ['한도상향', '한도증액', '한도늘리기', '한도올리기', '한도증대'],
            '단기카드대출': ['단기대출', '긴급대출', '임시대출', '소액대출'],
            '긴급배송': ['긴급배송', '당일배송', '익일배송', '빠른배송', '응급배송'],
            '이월약정': ['이월약정', '리볼빙', '부분결제', '최소결제', '이월결제'],
            '연체': ['연체', '미납', '체납', '지연납부', '연체료'],
            '해지': ['해지', '탈퇴', '중도해지', '약정해지', '서비스해지'],
            '심사': ['심사', '승인심사', '신용심사', '한도심사', '발급심사'],
            '안심클릭': ['안심클릭', '인증서비스', '보안결제', '추가인증'],
            '페이': ['페이', '간편결제', '모바일결제', '디지털결제']
        }
    
    def extract_category_keywords(self, category_texts: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """개선된 카테고리별 특성 키워드 추출"""
        print("개선된 카테고리별 키워드 추출 중...")
        
        category_keywords = {}
        
        # 전체 텍스트에서 공통 키워드 추출 (제외용)
        all_text_keywords = Counter()
        for texts in category_texts.values():
            for text in texts:
                keywords = self.extract_meaningful_keywords(text)
                all_text_keywords.update(keywords)
        
        # 너무 자주 나오는 키워드들 (전체 카테고리의 70% 이상에서 나오는 것들)
        common_across_categories = set()
        total_categories = len(category_texts)
        
        for keyword, count in all_text_keywords.items():
            if count > total_categories * 0.7:  # 70% 이상 카테고리에서 등장
                common_across_categories.add(keyword)
        
        for category, texts in category_texts.items():
            category_keyword_counts = Counter()
            
            # 해당 카테고리의 모든 텍스트에서 키워드 추출
            for text in texts:
                keywords = self.extract_meaningful_keywords(text)
                category_keyword_counts.update(keywords)
            
            # 카테고리별 특화 키워드 필터링
            specific_keywords = []
            
            for keyword, count in category_keyword_counts.most_common():
                if (count >= 3 and  # 최소 3번 이상 등장
                    keyword not in self.common_consultation_words and  # 공통어 제외
                    keyword not in common_across_categories and  # 범용 키워드 제외
                    self.is_category_relevant(keyword, category) and  # 카테고리 관련성 확인
                    len(keyword) >= 2):  # 최소 2글자 이상
                    
                    specific_keywords.append(keyword)
                
                if len(specific_keywords) >= 15:  # 상위 15개로 제한
                    break
            
            # 카테고리 특화 용어 추가
            category_specific = self.get_category_specific_terms(category)
            for term in category_specific:
                if term not in specific_keywords:
                    # 실제 텍스트에서 해당 용어가 사용되었는지 확인
                    term_count = sum(text.count(term) for text in texts)
                    if term_count > 0:
                        specific_keywords.insert(0, term)  # 우선순위로 앞에 추가
            
            category_keywords[category] = specific_keywords[:20]  # 최대 20개
            print(f"{category}: {len(specific_keywords)}개 특화 키워드")
        
        return category_keywords
    
    def extract_meaningful_keywords(self, text: str) -> List[str]:
        """의미있는 키워드만 추출"""
        # 한글 명사 패턴 (2글자 이상)
        korean_nouns = re.findall(r'[가-힣]{2,}', text)
        
        # 숫자+단위, 특수문자 포함 키워드 제거
        filtered_keywords = []
        for word in korean_nouns:
            if (not re.search(r'\d', word) and  # 숫자 포함 제외
                not re.search(r'[ㄱ-ㅎㅏ-ㅣ]', word) and  # 자음/모음만 있는 것 제외
                len(word) <= 10 and  # 너무 긴 단어 제외
                word not in self.common_consultation_words):  # 공통어 제외
                filtered_keywords.append(word)
        
        return filtered_keywords
    
    def is_category_relevant(self, keyword: str, category: str) -> bool:
        """키워드가 해당 카테고리와 관련성이 있는지 판단"""
        # 카테고리 이름에서 핵심 키워드 추출
        category_core_terms = set()
        
        # 슬래시로 구분된 용어들 추출
        terms = category.split('/')
        for term in terms:
            # 괄호 내용 제거하고 공백 정리
            clean_term = re.sub(r'\([^)]*\)', '', term).strip()
            if len(clean_term) >= 2:
                category_core_terms.add(clean_term)
        
        # 키워드가 카테고리 핵심 용어와 연관성이 있는지 확인
        for core_term in category_core_terms:
            if (keyword in core_term or 
                core_term in keyword or
                self.are_semantically_related(keyword, core_term)):
                return True
        
        return True  # 일단 모든 키워드를 허용하고, 다른 필터에서 걸러내기
    
    def are_semantically_related(self, word1: str, word2: str) -> bool:
        """두 단어가 의미적으로 연관되어 있는지 간단히 판단"""
        # 간단한 연관성 규칙
        related_pairs = {
            ('카드', '결제'), ('카드', '승인'), ('카드', '이용'),
            ('분실', '도난'), ('분실', '재발급'),
            ('할부', '개월'), ('할부', '무이자'),
            ('포인트', '적립'), ('포인트', '마일리지'),
            ('취소', '환불'), ('취소', '매출'),
            ('한도', '상향'), ('한도', '증액'),
            ('대출', '상환'), ('대출', '이자'),
            ('바우처', '상품권'), ('바우처', '쿠폰')
        }
        
        return (word1, word2) in related_pairs or (word2, word1) in related_pairs
    
    def get_category_specific_terms(self, category: str) -> List[str]:
        """카테고리별 특화 용어 반환"""
        category_lower = category.lower()
        specific_terms = []
        
        for key, terms in self.category_specific_terms.items():
            if key in category_lower or any(term in category_lower for term in key.split('/')):
                specific_terms.extend(terms)
        
        return specific_terms
    
    def extract_keywords_from_folder(self, folder_path: str) -> Dict:
        """폴더 내 모든 JSON 파일에서 키워드 자동 추출"""
        print("개선된 키워드 추출 시작...")
        
        all_texts = []
        category_texts = defaultdict(list)
        
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        total_files = len(json_files)
        print(f"총 {total_files}개의 JSON 파일을 처리합니다.")
        
        for i, filename in enumerate(json_files):  # 모든 파일 처리
            if i % 50 == 0:
                print(f"파일 처리 진행률: {i}/{total_files}")
                
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    consulting_content = item['consulting_content']
                    category = item.get('consulting_category', '기타')
                    
                    all_texts.append(consulting_content)
                    category_texts[category].append(consulting_content)
                    self.consulting_categories.add(category)
                    
            except Exception as e:
                print(f"파일 {filename} 처리 오류: {e}")
                continue
        
        # 1. 토픽 변화 키워드 추출 (기존 로직 유지)
        topic_change_keywords = self.extract_topic_change_keywords(all_texts)
        
        # 2. 개선된 카테고리별 주요 키워드 추출
        category_keywords = self.extract_category_keywords(category_texts)
        
        # 3. 일반 토픽 키워드 추출 (기존 로직 유지)
        general_keywords = self.extract_general_keywords(all_texts)
        
        return {
            'topic_change_keywords': topic_change_keywords,
            'category_keywords': category_keywords,
            'general_keywords': general_keywords,
            'categories': list(self.consulting_categories)
        }
    
    def extract_topic_change_keywords(self, texts: List[str]) -> List[str]:
        """토픽 변화를 나타내는 키워드 추출 (기존 로직)"""
        print("토픽 변화 키워드 추출 중...")
        
        topic_change_patterns = [
            r'\b(그런데|근데)\b',
            r'\b(혹시|만약|그럼|그러면)\b',
            r'\b(추가로|또|또한|그리고)\b',
            r'\b(그러니까|즉|다시말해)\b',
            r'\b(잠깐|잠시만)\b',
            r'\b(별도로|따로|개별적으로)\b'
        ]
        
        keyword_counts = Counter()
        
        for text in texts:
            lines = text.split('\n')
            for line in lines:
                if line.startswith('손님:') or line.startswith('상담사:'):
                    content = line[3:].strip() if line.startswith('손님:') else line[4:].strip()
                    
                    for pattern in topic_change_patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            if isinstance(match, tuple):
                                keyword_counts[' '.join(match)] += 1
                            else:
                                keyword_counts[match] += 1
        
        frequent_keywords = [keyword for keyword, count in keyword_counts.most_common() if count >= 5]
        print(f"추출된 토픽 변화 키워드: {len(frequent_keywords)}개")
        return frequent_keywords[:20]  # 상위 20개
    
    def extract_general_keywords(self, texts: List[str]) -> Dict[str, str]:
        """일반적인 상담 키워드 추출 및 분류 (기존 로직)"""
        print("일반 토픽 키워드 추출 중...")
        
        domain_keywords = {
            '결제', '청구', '납부', '출금', '이체', '승인', '거래',
            '카드', '신용카드', '체크카드', '발급', '재발급',
            '바우처', '할인', '적립', '포인트', '혜택', '쿠폰', '상품권',
            '취소', '환불', '분실', '도난', '정지', '해제',
            '변경', '수정', '등록', '해지', '연장',
            '확인', '조회', '내역', '명세서', '잔액',
            '가입', '신청', '연결', '해지', '이용'
        }
        
        keyword_counts = Counter()
        
        for text in texts:
            for keyword in domain_keywords:
                count = text.count(keyword)
                if count > 0:
                    keyword_counts[keyword] += count
        
        classified_keywords = {}
        for keyword, count in keyword_counts.most_common():
            if count >= 10:
                category = self.classify_keyword(keyword)
                classified_keywords[keyword] = category
        
        return classified_keywords
    
    def classify_keyword(self, keyword: str) -> str:
        """키워드를 적절한 카테고리로 분류 (기존 로직)"""
        classifications = {
            '결제 관련': ['결제', '청구', '납부', '출금', '이체', '승인', '거래'],
            '카드 관련': ['카드', '발급', '재발급', '신용카드', '체크카드'],
            '혜택 관련': ['바우처', '할인', '적립', '포인트', '혜택', '쿠폰', '상품권'],
            '취소/환불': ['취소', '환불', '분실', '도난', '정지', '해제'],
            '정보 변경': ['변경', '수정', '등록', '해지', '연장'],
            '내역 확인': ['확인', '조회', '내역', '명세서', '잔액'],
            '서비스 이용': ['가입', '신청', '연결', '해지', '이용']
        }
        
        for category, keywords in classifications.items():
            if keyword in keywords:
                return category
        
        return '일반 상담'
    
    def generate_keyword_code(self, extracted_keywords: Dict) -> str:
        """추출된 키워드를 코드로 생성"""
        code = "# 개선된 데이터 기반 자동 추출된 키워드들\n\n"
        
        # 토픽 변화 키워드
        code += "topic_change_keywords = [\n"
        for keyword in extracted_keywords['topic_change_keywords']:
            code += f"    '{keyword}',\n"
        code += "]\n\n"
        
        # 카테고리별 특화 키워드
        code += "category_specific_keywords = {\n"
        for category, keywords in extracted_keywords['category_keywords'].items():
            code += f"    '{category}': {keywords},\n"
        code += "}\n\n"
        
        # 일반 토픽 키워드
        code += "topic_keywords = {\n"
        for keyword, category in extracted_keywords['general_keywords'].items():
            code += f"    '{keyword}': '{category}',\n"
        code += "}\n"
        
        return code

# 사용 예시
def extract_improved_keywords(folder_path: str):
    """개선된 키워드 추출 함수"""
    extractor = ImprovedConsultationKeywordExtractor()
    
    # 키워드 추출
    keywords = extractor.extract_keywords_from_folder(folder_path)
    
    # 결과 출력
    print("\n=== 개선된 추출 결과 ===")
    print(f"토픽 변화 키워드: {len(keywords['topic_change_keywords'])}개")
    print(f"발견된 카테고리: {len(keywords['categories'])}개")
    print(f"일반 키워드: {len(keywords['general_keywords'])}개")
    
    # 카테고리별 키워드 미리보기
    print("\n=== 카테고리별 키워드 미리보기 ===")
    for category, keyword_list in keywords['category_keywords'].items():
        print(f"{category}: {keyword_list[:10]}...")  # 상위 10개만 미리보기
    
    # 코드 생성
    code = extractor.generate_keyword_code(keywords)
    
    # 파일로 저장
    with open('improved_extracted_keywords.py', 'w', encoding='utf-8') as f:
        f.write(code)
    
    print("\n개선된 키워드 추출 완료! improved_extracted_keywords.py 파일 확인하세요.")
    
    return keywords

if __name__ == "__main__":
    # 실제 사용
    keywords = extract_improved_keywords('C:/Users/kimdu/Desktop/LLM_Pre_trained_Instruction_Tuning_Data/3opendata/1_data/Training/02_labelingdata/요약')
    print("개선된 키워드 추출기 준비 완료!")