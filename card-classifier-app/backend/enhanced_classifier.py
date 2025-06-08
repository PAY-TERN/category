#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고도화된 점진적 재학습 카드 분류기 (지속적 모델 업데이트)
- ML 분류 디버깅 개선
- 카드사용 월별 배치 분할
- 하나의 모델 지속적 업데이트
"""

import pandas as pd
import subprocess
import json
import re
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
from collections import Counter
import warnings
import logging
import time
import pickle
from datetime import datetime
import traceback

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class PersistentIncrementalCategorizer:
    """지속적 모델 업데이트 분류기 (월별 배치 처리)"""
    
    CATEGORIES = [
        "식비", "카페/간식", "편의점/마트/잡화", "술/유흥", "쇼핑", 
        "영화/OTT", "취미/여가", "의료/건강", "주거/통신/공과금", 
        "보험/세금/기타금융", "미용/뷰티", "교통/대중교통", 
        "자동차/주유소", "여행/숙박", "항공", "교육", "생활", "기타","간편결제"
    ]
    
    # 카테고리별 키워드 패턴
    CATEGORY_KEYWORDS = {
        "식비": ["맛집", "식당", "횟집", "한식", "중식", "일식", "양식", "분식", "치킨", "피자", "햄버거", "김밥", "국밥", "찌개", "곰탕", "삼겹살", "갈비", "닭갈비", "족발", "보쌈", "냉면", "비빔밥", "돈까스", "라면", "우동", "짬뽕", "짜장면"],
        "카페/간식": ["스타벅스", "이디야", "커피빈", "투썸", "디저트", "베이커리", "빵집", "도넛", "아이스크림", "빙수", "떡집", "과자", "케이크", "마카롱", "와플", "크로플", "bubble tea", "버블티", "smoothie", "frappuccino"],
        "편의점/마트/잡화": ["gs25", "cu", "세븐일레븐", "미니스톱", "emart", "롯데마트", "홈플러스", "코스트코", "다이소", "아트박스", "무인양품", "올리브영", "왓슨스", "부츠", "마켓컬리", "쿠팡", "편의점", "마트", "슈퍼"],
        "술/유흥": ["호프", "술집", "맥주", "소주", "와인", "칵테일", "노래방", "클럽", "펜션", "룸", "가라오케", "포차", "izakaya", "pub", "bar", "beer", "wine", "whiskey", "cocktail"],
        "쇼핑": ["백화점", "아울렛", "온라인", "쇼핑몰", "의류", "신발", "가방", "액세서리", "브랜드", "패션", "유니클로", "자라", "h&m", "gap", "나이키", "아디다스", "명품", "luxury"],
        "영화/OTT": ["cgv", "롯데시네마", "메가박스", "영화", "넷플릭스", "디즈니", "왓챠", "웨이브", "cinema", "movie", "netflix", "disney+", "youtube", "spotify", "apple music"],
        "취미/여가": ["pc방", "게임", "볼링", "당구", "골프", "수영", "헬스", "요가", "독서실", "스포츠", "찜질방", "사우나", "spa", "마사지", "게임센터", "오락실", "노래방", "escape room"],
        "의료/건강": ["병원", "약국", "치과", "한의원", "안과", "피부과", "정형외과", "내과", "검진", "의료", "pharmacy", "hospital", "clinic", "dental", "medical", "health", "medicine"],
        "주거/통신/공과금": ["전기요금", "가스요금", "수도요금", "관리비", "통신비", "인터넷", "휴대폰", "전화요금", "임대료", "월세", "전세", "kt", "skt", "lg유플러스", "utility", "rent"],
        "보험/세금/기타금융": ["보험", "세금", "국세청", "시청", "구청", "은행", "카드", "대출", "적금", "펀드", "투자", "주식", "insurance", "tax", "bank", "loan", "investment", "stock"],
        "미용/뷰티": ["미용실", "헤어샵", "네일샵", "피부관리", "마사지", "화장품", "에스테틱", "스파", "beauty", "salon", "nail", "cosmetic", "skincare", "makeup", "perfume"],
        "교통/대중교통": ["지하철", "버스", "택시", "카카오택시", "우버", "기차", "고속버스", "시외버스", "교통카드", "subway", "bus", "taxi", "train", "uber", "grab", "transport"],
        "자동차/주유소": ["주유소", "gs칼텍스", "sk에너지", "현대오일뱅크", "s-oil", "정비소", "세차", "타이어", "자동차", "gas station", "oil", "car wash", "tire", "auto", "vehicle"],
        "여행/숙박": ["호텔", "모텔", "펜션", "리조트", "에어비앤비", "게스트하우스", "여행사", "항공권", "기차표", "hotel", "motel", "resort", "airbnb", "travel", "booking", "agoda"],
        "항공": ["대한항공", "아시아나", "제주항공", "진에어", "티웨이", "이스타항공", "공항", "항공", "airline", "airport", "flight", "aviation", "korean air", "asiana"],
        "교육": ["학원", "과외", "온라인강의", "교육비", "학비", "교재", "문구점", "학용품", "유치원", "어학원", "academy", "education", "school", "university", "course", "tuition"],
        "생활": ["세탁소", "수선집", "열쇠", "인쇄", "복사", "택배", "우체국", "동사무소", "민원", "생활용품", "laundry", "post office", "copy", "print", "delivery", "repair"]
    }
    
    def __init__(self, merchant_column: str = '가맹점명', date_column: str = '이용일자'):
        """초기화 - 하나의 모델만 지속적으로 업데이트"""
        self.merchant_column = merchant_column
        self.date_column = date_column
        
        # 모델명은 고정 - 하나의 모델만 사용
        self.model_name = "unified_model"
        
        # 모델 저장 디렉토리
        self.model_dir = Path('models')
        self.model_dir.mkdir(exist_ok=True)
        
        # 데이터 저장 디렉토리
        self.data_dir = Path('training_data')
        self.data_dir.mkdir(exist_ok=True)
        
        # 앙상블 모델 구성
        self.ensemble_classifier = None
        self.vectorizer = None
        self.char_vectorizer = None
        self.label_encoder = None
        self.scaler = StandardScaler()
        
        # 학습 데이터 관리
        self.all_training_data = pd.DataFrame()
        self.performance_history = []
        self.current_batch = 0
        
        # 신뢰도 임계값 동적 조정 (초기값을 낮게 설정)
        self.confidence_threshold = 0.4  # 0.7에서 0.5로 낮춤
        self.min_confidence_threshold = 0.3
        
        # 키워드 기반 분류기
        self.keyword_patterns = self._compile_keyword_patterns()
        
        # Ollama 설정
        self.ollama_model = "llama3"
        self.ollama_available = self._check_ollama_connection()
        
        # 초기화 시 기존 모델 자동 로드 시도
        self.auto_load_existing_model()
    
    def create_monthly_batches(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """월별로 배치 생성"""
        logger.info("📦 월별 배치 생성 중...")
        
        # 날짜 파싱
        df = self.parse_date_column(df)
        
        # 유효한 날짜가 있는 데이터만 사용
        valid_df = df[df['년월'].notna()].copy()
        
        if len(valid_df) == 0:
            logger.error("❌ 유효한 날짜 데이터가 없습니다.")
            return []
        
        # 월별 그룹화
        monthly_groups = valid_df.groupby('년월')
        batches = []
        
        # 월별로 정렬해서 배치 생성
        sorted_months = sorted(monthly_groups.groups.keys())
        
        for month in sorted_months:
            month_data = monthly_groups.get_group(month)
            batches.append(month_data.copy())
            logger.info(f"   📅 {month}: {len(month_data)}개 거래")
        
        logger.info(f"📦 총 {len(batches)}개 월별 배치 생성")
        return batches
    
    def load_excel_data(self, excel_path: str) -> pd.DataFrame:
        """엑셀 데이터 로드"""
        logger.info(f"📁 엑셀 파일 로드: {excel_path}")
        df = pd.read_excel(excel_path)
        logger.info(f"📊 전체 데이터: {len(df)}개 행")
        
        df_filtered = df[df[self.merchant_column].notna()].copy()
        logger.info(f"📊 유효한 데이터: {len(df_filtered)}개 행")
        
        return df_filtered
    
    def _compile_keyword_patterns(self) -> Dict[str, List[re.Pattern]]:
        """키워드 패턴 컴파일"""
        patterns = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            patterns[category] = [re.compile(keyword, re.IGNORECASE) for keyword in keywords]
        return patterns
    
    def _check_ollama_connection(self) -> bool:
        """Ollama 연결 상태 확인"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Ollama 연결 확인 실패: {e}")
            return False
    
    def auto_load_existing_model(self):
        """초기화 시 기존 모델 자동 로드"""
        logger.info("🔍 기존 학습된 모델 검색 중...")
        
        # 최신 모델 파일 확인
        latest_model_path = self.model_dir / f'{self.model_name}_latest.pkl'
        
        if latest_model_path.exists():
            try:
                logger.info(f"✅ 기존 모델 발견: {latest_model_path}")
                self.load_complete_model()
                logger.info(f"🎯 기존 모델 로드 완료! 누적 학습 데이터: {len(self.all_training_data)}개")
                return True
            except Exception as e:
                logger.warning(f"⚠️ 기존 모델 로드 실패: {e}")
        
        logger.info("📝 기존 모델 없음. 새로운 모델로 시작합니다.")
        return False
    def save_complete_model(self, batch_num: int = None):
        """모델과 모든 관련 데이터 완전 저장 - 완전 수정됨"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # performance_history 완전 정제
        cleaned_performance_history = []
        for i, perf in enumerate(self.performance_history):
            if isinstance(perf, dict):
                # 필수 필드가 모두 있는지 확인하고 정제
                cleaned_perf = {
                    'batch': int(perf.get('batch', i + 1)),
                    'train_accuracy': float(perf.get('train_accuracy', 0.0)),
                    'cv_mean': float(perf.get('cv_mean', 0.0)),
                    'cv_std': float(perf.get('cv_std', 0.0)),
                    'training_samples': int(perf.get('training_samples', 0)),
                    'categories': int(perf.get('categories', 0)),
                    'confidence_threshold': float(perf.get('confidence_threshold', 0.5)),
                    'timestamp': str(perf.get('timestamp', datetime.now().isoformat()))
                }
                
                # DataFrame이나 복잡한 객체가 없는지 확인
                valid_entry = True
                for key, value in cleaned_perf.items():
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        logger.warning(f"⚠️ 성능 데이터에서 pandas 객체 감지, 해당 항목 제외")
                        valid_entry = False
                        break
                
                if valid_entry:
                    cleaned_performance_history.append(cleaned_perf)
        
        logger.info(f"📊 성능 히스토리 정제: {len(self.performance_history)} -> {len(cleaned_performance_history)}개")
        
        # 모델 상태 딕셔너리 (정제된 데이터만 포함)
        model_state = {
            'ensemble_classifier': self.ensemble_classifier,
            'vectorizer': self.vectorizer,
            'char_vectorizer': self.char_vectorizer,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'confidence_threshold': float(self.confidence_threshold),
            'min_confidence_threshold': float(self.min_confidence_threshold),
            'performance_history': cleaned_performance_history,  # 완전 정제된 히스토리
            'current_batch': int(self.current_batch),
            'ollama_model': str(self.ollama_model),
            'model_name': str(self.model_name),
            'categories': list(self.CATEGORIES),
            'saved_at': timestamp,
            'merchant_column': str(self.merchant_column),
            'date_column': str(self.date_column)
        }
        
        # 최신 모델 저장
        latest_model_path = self.model_dir / f'{self.model_name}_latest.pkl'
        with open(latest_model_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        # 배치별 백업 저장
        if batch_num is not None and isinstance(batch_num, int):
            backup_model_path = self.model_dir / f'{self.model_name}_batch_{batch_num}.pkl'
            with open(backup_model_path, 'wb') as f:
                pickle.dump(model_state, f)
            
            logger.info(f"💾 모델 저장 완료: 배치 {batch_num}")
        
        # 학습 데이터 저장
        if len(self.all_training_data) > 0:
            latest_data_path = self.data_dir / f'{self.model_name}_training_data.xlsx'
            self.all_training_data.to_excel(latest_data_path, index=False)
        
        logger.info(f"✅ 완전 저장 완료: {latest_model_path}")
        
        # 정제된 데이터로 메모리상 performance_history 업데이트
        self.performance_history = cleaned_performance_history
        
    def load_complete_model(self, batch_num: int = None):
            """저장된 모델과 모든 관련 데이터 완전 로드 - 완전 수정됨"""
            try:
                # 로드할 모델 파일 결정
                if batch_num:
                    model_path = self.model_dir / f'{self.model_name}_batch_{batch_num}.pkl'
                else:
                    model_path = self.model_dir / f'{self.model_name}_latest.pkl'
                
                # 모델 상태 로드
                if not model_path.exists():
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
                
                with open(model_path, 'rb') as f:
                    model_state = pickle.load(f)
                
                # 모델 상태 복원
                self.ensemble_classifier = model_state['ensemble_classifier']
                self.vectorizer = model_state['vectorizer']
                self.char_vectorizer = model_state['char_vectorizer']
                self.label_encoder = model_state['label_encoder']
                self.scaler = model_state['scaler']
                self.confidence_threshold = model_state.get('confidence_threshold', 0.5)

                # 임계값 상한 제한
                if self.confidence_threshold > 0.7:
                    logger.warning(f"⚠️ 불합리하게 높은 신뢰도 임계값 감지: {self.confidence_threshold} → 0.7로 조정")
                    self.confidence_threshold = 0.7
                        
                self.min_confidence_threshold = model_state.get('min_confidence_threshold', 0.3)
                
                # performance_history 완전 정제 및 재구성
                raw_history = model_state.get('performance_history', [])
                self.performance_history = []
                
                for i, perf in enumerate(raw_history):
                    if isinstance(perf, dict):
                        # 새로운 정제된 성능 딕셔너리 생성
                        cleaned_perf = {
                            'batch': perf.get('batch', i + 1),  # batch가 없으면 순서대로 번호 부여
                            'train_accuracy': float(perf.get('train_accuracy', 0.0)),
                            'cv_mean': float(perf.get('cv_mean', 0.0)),
                            'cv_std': float(perf.get('cv_std', 0.0)),
                            'training_samples': int(perf.get('training_samples', 0)),
                            'categories': int(perf.get('categories', 0)),
                            'confidence_threshold': float(perf.get('confidence_threshold', 0.5)),
                            'timestamp': perf.get('timestamp', datetime.now().isoformat())
                        }
                        
                        # DataFrame이나 기타 복잡한 객체는 제외
                        valid_perf = True
                        for key, value in cleaned_perf.items():
                            if isinstance(value, pd.DataFrame):
                                logger.warning(f"   ⚠️ 성능 데이터에서 DataFrame 발견, 항목 제외: {i}")
                                valid_perf = False
                                break
                        
                        if valid_perf:
                            self.performance_history.append(cleaned_perf)
                            logger.debug(f"   ✅ 성능 데이터 정제 완료: 배치 {cleaned_perf['batch']}")
                
                self.current_batch = model_state.get('current_batch', 0)
                
                # 학습 데이터 로드
                data_path = self.data_dir / f'{self.model_name}_training_data.xlsx'
                if data_path.exists():
                    self.all_training_data = pd.read_excel(data_path)
                    logger.info(f"📊 학습 데이터 로드: {len(self.all_training_data)}개")
                else:
                    self.all_training_data = pd.DataFrame()
                
                # 정제된 데이터로 즉시 재저장 (향후 문제 방지)
                logger.info("🔧 정제된 데이터로 모델 재저장 중...")
                self.save_complete_model()
                
                logger.info(f"✅ 모델 로드 완료!")
                logger.info(f"   - 총 배치: {len(self.performance_history)}개")
                logger.info(f"   - 누적 샘플: {len(self.all_training_data)}개")
                logger.info(f"   - 현재 정확도: {self.performance_history[-1]['train_accuracy']:.3f}" if self.performance_history else "   - 정확도: 아직 없음")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ 모델 로드 실패: {e}")
                logger.error(f"   상세 오류: {traceback.format_exc()}")
                return False
    # enhanced_classifier.py의 parse_date_column 함수를 다음과 같이 수정

    def parse_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """날짜 컬럼 파싱 (다양한 형식 지원) - 디버깅 강화"""
        df = df.copy()
        
        if self.date_column not in df.columns:
            raise ValueError(f"날짜 컬럼 '{self.date_column}'을 찾을 수 없습니다.")
        
        logger.info(f"🔍 날짜 컬럼 '{self.date_column}' 원본 데이터 샘플:")
        sample_data = df[self.date_column].dropna().head(10)
        for i, value in enumerate(sample_data):
            logger.info(f"   {i+1}. {repr(value)} (타입: {type(value).__name__})")
        
        try:
            # 1차: pandas 기본 파싱 시도
            logger.info("📅 1차: pandas 기본 날짜 파싱 시도...")
            df_temp = df.copy()
            df_temp[self.date_column] = pd.to_datetime(df_temp[self.date_column], errors='coerce')
            
            # 성공적으로 파싱된 데이터 확인
            valid_dates = df_temp[self.date_column].dropna()
            if len(valid_dates) > 0:
                logger.info(f"✅ pandas 파싱 성공: {len(valid_dates)}개 / {len(df)}개")
                logger.info(f"   날짜 범위: {valid_dates.min()} ~ {valid_dates.max()}")
                
                # 1970년대 데이터가 많으면 엑셀 숫자 날짜일 가능성
                dates_1970 = valid_dates[valid_dates.dt.year == 1970]
                if len(dates_1970) > len(valid_dates) * 0.5:
                    logger.warning(f"⚠️ 1970년 날짜가 {len(dates_1970)}개 감지 - 엑셀 숫자 날짜 변환 시도")
                    # 엑셀 숫자 날짜 변환 시도
                    df = self._convert_excel_serial_dates(df)
                else:
                    df = df_temp
            else:
                logger.warning("⚠️ pandas 기본 파싱 실패 - 수동 변환 시도")
                df = self._manual_date_conversion(df)
                
        except Exception as e:
            logger.error(f"❌ 날짜 파싱 오류: {e}")
            df = self._manual_date_conversion(df)
        
        # 최종 결과 확인
        valid_count = df[self.date_column].notna().sum()
        failed_count = len(df) - valid_count
        
        if failed_count > 0:
            logger.warning(f"⚠️ 날짜 파싱 실패: {failed_count}개 행")
        
        # 연-월 컬럼 추가
        df['년월'] = df[self.date_column].dt.strftime('%Y-%m')
        
        # 월별 분포 확인
        month_distribution = df['년월'].value_counts().sort_index()
        logger.info(f"📊 월별 분포:")
        for month, count in month_distribution.head(10).items():
            logger.info(f"   {month}: {count}개")
        
        logger.info(f"📅 날짜 파싱 완료: {valid_count}개 행")
        
        return df

    def _convert_excel_serial_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """엑셀 시리얼 날짜 변환"""
        logger.info("🔄 엑셀 시리얼 날짜 변환 시도...")
        df_temp = df.copy()
        
        def convert_excel_serial(value):
            """엑셀 시리얼 날짜를 datetime으로 변환"""
            try:
                if pd.isna(value):
                    return None
                
                # 숫자인 경우 엑셀 시리얼 날짜로 간주
                if isinstance(value, (int, float)):
                    # 엑셀 기준: 1900년 1월 1일 = 1
                    # pandas 기준: 1899년 12월 30일 = 0
                    excel_epoch = pd.Timestamp('1899-12-30')
                    days_to_add = pd.Timedelta(days=int(value))
                    return excel_epoch + days_to_add
                
                # 문자열인 경우 그대로 파싱 시도
                return pd.to_datetime(value, errors='coerce')
                
            except Exception:
                return None
        
        df_temp[self.date_column] = df_temp[self.date_column].apply(convert_excel_serial)
        
        # 변환 결과 확인
        valid_converted = df_temp[self.date_column].dropna()
        if len(valid_converted) > 0:
            logger.info(f"✅ 엑셀 변환 성공: {len(valid_converted)}개")
            logger.info(f"   변환 후 날짜 범위: {valid_converted.min()} ~ {valid_converted.max()}")
            return df_temp
        else:
            logger.warning("⚠️ 엑셀 변환 실패")
            return df

    def _manual_date_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """수동 날짜 변환 (다양한 형식 지원)"""
        logger.info("🔧 수동 날짜 변환 시도...")
        df_temp = df.copy()
        
        def parse_date_manual(value):
            """다양한 날짜 형식 수동 파싱"""
            if pd.isna(value):
                return None
            
            value_str = str(value).strip()
            
            # 패턴들 시도
            patterns = [
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD, YYYY/MM/DD
                r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # MM/DD/YYYY, DD/MM/YYYY
                r'(\d{4})(\d{2})(\d{2})',              # YYYYMMDD
                r'(\d{4})\.(\d{1,2})\.(\d{1,2})',      # YYYY.MM.DD
            ]
            
            for pattern in patterns:
                match = re.match(pattern, value_str)
                if match:
                    groups = match.groups()
                    try:
                        if len(groups[0]) == 4:  # 첫 번째가 연도
                            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        else:  # 첫 번째가 월 또는 일
                            month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                        
                        return pd.Timestamp(year=year, month=month, day=day)
                    except ValueError:
                        continue
            
            # 모든 패턴 실패 시 None 반환
            return None
        
        df_temp[self.date_column] = df_temp[self.date_column].apply(parse_date_manual)
        
        valid_manual = df_temp[self.date_column].dropna()
        if len(valid_manual) > 0:
            logger.info(f"✅ 수동 변환 성공: {len(valid_manual)}개")
            logger.info(f"   수동 변환 후 날짜 범위: {valid_manual.min()} ~ {valid_manual.max()}")
            return df_temp
        else:
            logger.error("❌ 모든 날짜 변환 방법 실패")
            return df

    # 또한 백엔드의 전처리 함수도 수정
    def convert_excel_date(value):
        """엑셀 시리얼 날짜를 실제 날짜로 변환 (개선된 버전)"""
        if pd.isna(value):
            return None
        
        # 이미 datetime 객체인 경우
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')
        
        # 숫자인 경우 - 엑셀 시리얼 날짜 처리
        if isinstance(value, (int, float)):
            try:
                # 합리적인 범위 확인 (1900년~2100년 사이)
                if 1 <= value <= 73048:  # 1900-01-01 ~ 2099-12-31
                    # 엑셀 기준점: 1899-12-30
                    excel_epoch = datetime(1899, 12, 30)
                    converted_date = excel_epoch + timedelta(days=int(value))
                    return converted_date.strftime('%Y-%m-%d')
                else:
                    return str(value)  # 범위 밖이면 문자열로 유지
            except Exception as e:
                logger.warning(f"숫자 날짜 변환 실패: {value} -> {e}")
                return str(value)
        
        # 문자열인 경우
        if isinstance(value, str):
            # 이미 올바른 형식인 경우
            if re.match(r'\d{4}-\d{2}-\d{2}', value):
                return value
            
            # 다양한 형식 시도
            try:
                parsed_date = pd.to_datetime(value, errors='coerce')
                if not pd.isna(parsed_date):
                    return parsed_date.strftime('%Y-%m-%d')
            except:
                pass
        
        return str(value)

    # 날짜 컬럼 감지 함수도 개선
    def is_likely_excel_date(series, threshold=0.3):
        """시리즈가 엑셀 날짜 컬럼인지 판단 (개선된 버전)"""
        if series.dtype not in ['int64', 'float64', 'object']:
            return False
            
        numeric_values = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric_values) == 0:
            return False
        
        # 엑셀 날짜 범위 확인 (1900년~2030년)
        excel_range_values = numeric_values[(numeric_values >= 1) & (numeric_values <= 47483)]
        recent_date_values = numeric_values[(numeric_values >= 36526) & (numeric_values <= 47483)]  # 2000-2030
        
        recent_ratio = len(recent_date_values) / len(numeric_values)
        total_ratio = len(excel_range_values) / len(numeric_values)
        
        logger.info(f"   날짜 판단: 전체비율={total_ratio:.2f}, 최근비율={recent_ratio:.2f}")
        
        return recent_ratio >= threshold or total_ratio >= 0.7
    
    def create_monthly_batches(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """월별로 배치 생성"""
        logger.info("📦 월별 배치 생성 중...")
        
        # 날짜 파싱
        df = self.parse_date_column(df)
        
        # 유효한 날짜가 있는 데이터만 사용
        valid_df = df[df['년월'].notna()].copy()
        
        if len(valid_df) == 0:
            logger.error("❌ 유효한 날짜 데이터가 없습니다.")
            return []
        
        # 월별 그룹화
        monthly_groups = valid_df.groupby('년월')
        batches = []
        
        # 월별로 정렬해서 배치 생성
        sorted_months = sorted(monthly_groups.groups.keys())
        
        for month in sorted_months:
            month_data = monthly_groups.get_group(month)
            batches.append(month_data.copy())
            logger.info(f"   📅 {month}: {len(month_data)}개 거래")
        
        logger.info(f"📦 총 {len(batches)}개 월별 배치 생성")
        return batches
    
    def preprocess_merchant_name(self, name: str) -> str:
        """고도화된 가맹점명 전처리"""
        if pd.isna(name):
            return ""
        
        name = str(name)
        
        # 특수문자 정규화
        name = re.sub(r'[()[\]{}]', ' ', name)
        name = re.sub(r'[^\w가-힣\s]', ' ', name)
        
        # 불필요한 접미사 제거
        suffixes = ['점', '지점', '본점', '매장', '대리점', '프라자', '센터', '몰', '마트']
        for suffix in suffixes:
            name = re.sub(f'{suffix}$', '', name)
        
        # 공백 정규화
        name = re.sub(r'\s+', ' ', name)
        
        return name.strip().lower()
    
    def extract_enhanced_features(self, merchants: List[str]) -> np.ndarray:
        """고도화된 특성 추출"""
        # 기본 TF-IDF 특성
        word_features = self.vectorizer.transform(merchants)
        
        # 문자 단위 TF-IDF 특성
        char_features = self.char_vectorizer.transform(merchants)
        
        # 수작업 특성
        manual_features = []
        for merchant in merchants:
            features = []
            
            # 길이 특성
            features.append(len(merchant))
            features.append(len(merchant.split()))
            
            # 숫자 포함 여부
            features.append(1 if re.search(r'\d', merchant) else 0)
            
            # 영어 포함 여부
            features.append(1 if re.search(r'[a-zA-Z]', merchant) else 0)
            
            # 특수문자 개수
            features.append(len(re.findall(r'[^\w가-힣\s]', merchant)))
            
            # 키워드 매칭 점수
            keyword_scores = [0] * len(self.CATEGORIES)
            for i, category in enumerate(self.CATEGORIES):
                if category in self.keyword_patterns:
                    score = sum(1 for pattern in self.keyword_patterns[category] 
                              if pattern.search(merchant))
                    keyword_scores[i] = score
            
            features.extend(keyword_scores)
            manual_features.append(features)
        
        manual_features = np.array(manual_features)
        
        # 모든 특성 결합
        combined_features = np.hstack([
            word_features.toarray(),
            char_features.toarray(),
            manual_features
        ])
        
        return combined_features
    
    def keyword_based_classification(self, merchant: str) -> Optional[str]:
        """키워드 기반 분류 (규칙 기반)"""
        merchant_lower = str(merchant).lower()

        category_scores = {}
        
        for category, patterns in self.keyword_patterns.items():
            score = sum(1 for pattern in patterns if pattern.search(merchant_lower))
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] >= 1:  # 최소 1개 키워드 매칭
                return best_category[0]
        
        return None
    
    def create_ensemble_model(self) -> VotingClassifier:
        """앙상블 모델 생성"""
        # 개별 모델들
        rf = RandomForestClassifier(
            n_estimators=200,  # 300에서 200으로 줄여서 속도 향상
            max_depth=20,      # 25에서 20으로 줄임
            min_samples_split=3,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        lr = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        svm = SVC(
            C=1.0,
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        # 소프트 투표 앙상블
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lr', lr),
                ('svm', svm)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def _adjust_confidence_threshold(self, accuracy: float):
        """개선된 신뢰도 임계값 동적 조정"""
        data_size = len(self.all_training_data)
        
        # 데이터 크기에 따른 기본 임계값 설정
        if data_size < 50:
            base_threshold = 0.3
        elif data_size < 100:
            base_threshold = 0.4
        elif data_size < 200:
            base_threshold = 0.5
        else:
            base_threshold = 0.6
        
        # 정확도에 따른 조정 - 더 보수적으로 수정
        if accuracy > 0.95:
            self.confidence_threshold = min(base_threshold + 0.1, 0.6)  # 상한을 0.6으로 제한
        elif accuracy > 0.9:
            self.confidence_threshold = min(base_threshold + 0.05, 0.55)
        elif accuracy > 0.8:
            self.confidence_threshold = min(base_threshold + 0.02, 0.5)
        else:
            self.confidence_threshold = max(base_threshold, self.min_confidence_threshold)

        
        logger.info(f"   🎯 임계값 조정: {self.confidence_threshold:.3f} (데이터: {data_size}개, 정확도: {accuracy:.3f})")
    def retrain_enhanced_model(self, batch_num: int):
        """고도화된 모델 재학습 (기존 모델 업데이트) - 완전 수정됨"""
        logger.info(f"\n🧠 배치 {batch_num} - 기존 모델 업데이트")
        
        # 유효한 학습 데이터 선별
        valid_data = self.all_training_data[
            (self.all_training_data[self.merchant_column].notna()) & 
            (self.all_training_data['카테고리'].notna()) &
            (self.all_training_data['카테고리'].isin(self.CATEGORIES))
        ].copy()
        
        if len(valid_data) < 10:
            logger.warning("   ⚠️ 학습 데이터가 부족합니다.")
            # 빈 성능 딕셔너리라도 필수 필드는 포함
            empty_performance = {
                'batch': int(batch_num),
                'train_accuracy': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'training_samples': int(len(valid_data)),
                'categories': 0,
                'confidence_threshold': float(self.confidence_threshold),
                'timestamp': datetime.now().isoformat()
            }
            return empty_performance
        
        logger.info(f"   전체 누적 학습 데이터: {len(valid_data)}개")
        
        # 특성 추출 준비
        X_raw = valid_data[self.merchant_column].apply(self.preprocess_merchant_name)
        y = valid_data['카테고리']
        
        # 벡터라이저 초기화/업데이트
        if self.vectorizer is None:
            logger.info("   🔧 벡터라이저 초기화")
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.9,
                analyzer='word',
                token_pattern=r'\b\w+\b'
            )
            
            self.char_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(2, 4),
                min_df=1,
                max_df=0.9,
                analyzer='char_wb'
            )
        else:
            logger.info("   🔄 기존 벡터라이저 업데이트")
        
        # 전체 데이터로 벡터라이저 재학습
        self.vectorizer.fit(X_raw)
        self.char_vectorizer.fit(X_raw)
        
        # 고도화된 특성 추출
        X_enhanced = self.extract_enhanced_features(X_raw.tolist())
        X_scaled = self.scaler.fit_transform(X_enhanced)
        
        logger.info(f"   특성 수: {X_scaled.shape[1]}개")
        
        # 라벨 인코딩
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            logger.info("   🔧 라벨 인코더 초기화")
        else:
            logger.info("   🔄 기존 라벨 인코더 업데이트")
        
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f"   카테고리 수: {len(self.label_encoder.classes_)}개")
        
        # 앙상블 모델 재학습
        logger.info("   🤖 앙상블 모델 재학습")
        self.ensemble_classifier = self.create_ensemble_model()
        self.ensemble_classifier.fit(X_scaled, y_encoded)
        
        # 교차 검증으로 성능 평가
        try:
            cv_scores = cross_val_score(self.ensemble_classifier, X_scaled, y_encoded, cv=min(5, len(set(y))))
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception as e:
            logger.warning(f"   ⚠️ 교차검증 실패: {e}")
            cv_mean, cv_std = 0.0, 0.0
        
        # 훈련 정확도
        y_pred = self.ensemble_classifier.predict(X_scaled)
        train_accuracy = float(accuracy_score(y_encoded, y_pred))
        
        # 신뢰도 임계값 동적 조정
        self._adjust_confidence_threshold(train_accuracy)
        
        # 성능 기록 - 모든 필드를 명시적으로 기본 타입으로 생성
        performance = {
            'batch': int(batch_num),
            'train_accuracy': train_accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_samples': int(len(valid_data)),
            'categories': int(len(set(y))),
            'confidence_threshold': float(self.confidence_threshold),
            'timestamp': datetime.now().isoformat()
        }
        
        # performance_history에 추가 전에 검증
        logger.info(f"   📊 성능 기록 추가: batch={performance['batch']}, accuracy={performance['train_accuracy']:.3f}")
        self.performance_history.append(performance)
        
        # 모델 자동 저장
        self.save_complete_model(batch_num)
        
        logger.info(f"   ✅ 모델 업데이트 완료")
        logger.info(f"   훈련 정확도: {train_accuracy:.3f}")
        logger.info(f"   교차검증 평균: {cv_mean:.3f} (±{cv_std:.3f})")
        logger.info(f"   신뢰도 임계값: {self.confidence_threshold:.3f}")
        
        return performance
    def classify_batch_with_enhanced_model(self, batch_df: pd.DataFrame, batch_num: int, month: str) -> pd.DataFrame:
        """고도화된 모델로 배치 분류 (ML → 키워드 → LLM 순서로 변경)"""
        logger.info(f"\n🔀 배치 {batch_num} ({month}) - 하이브리드 분류 (ML 우선)")
        
        unique_merchants = batch_df[self.merchant_column].unique().tolist()
        logger.info(f"   고유 가맹점: {len(unique_merchants)}개")
        
        # 1단계: ML 분류 우선 수행
        ml_results = {}
        remaining_candidates = []
        
        if self.ensemble_classifier is not None:
            try:
                logger.info(f"   🤖 ML 모델 우선 분류 - 신뢰도 임계값: {self.confidence_threshold}")
                
                # 전처리
                processed_names = [self.preprocess_merchant_name(m) for m in unique_merchants]
                logger.info(f"   📝 전처리 완료: {len(processed_names)}개")
                
                # 특성 추출
                X = self.extract_enhanced_features(processed_names)
                logger.info(f"   🔧 특성 추출 완료: {X.shape}")
                
                # 스케일링 시도 (차원 불일치 처리 포함)
                try:
                    X_scaled = self.scaler.transform(X)
                    logger.info(f"   📏 스케일링 완료: {X_scaled.shape}")
                    
                except ValueError as scale_error:
                    # 특성 차원 불일치 오류 감지
                    if "features" in str(scale_error) and "expecting" in str(scale_error):
                        logger.warning(f"   ⚠️ 특성 차원 불일치 감지!")
                        logger.warning(f"   오류 메시지: {scale_error}")
                        logger.warning(f"   🔄 새로운 특성에 맞춰 모델 재학습을 시작합니다...")
                        
                        # 현재 배치 데이터를 임시 카테고리로 설정 (키워드 분류 먼저 시도)
                        temp_batch_data = batch_df.copy()
                        temp_keyword_results = {}
                        for merchant in unique_merchants:
                            keyword_result = self.keyword_based_classification(merchant)
                            temp_keyword_results[merchant] = keyword_result if keyword_result else '기타'
                        
                        temp_batch_data['카테고리'] = temp_batch_data[self.merchant_column].map(temp_keyword_results)
                        temp_batch_data['카테고리'] = temp_batch_data['카테고리'].fillna('기타')
                        
                        # 임시로 학습 데이터에 추가
                        self.merge_with_existing_data(temp_batch_data)
                        
                        # 모델 재학습 (새로운 특성 차원으로)
                        self.retrain_enhanced_model(batch_num)
                        
                        # 재학습 후 다시 특성 추출 및 스케일링 시도
                        X = self.extract_enhanced_features(processed_names)
                        X_scaled = self.scaler.transform(X)
                        logger.info(f"   🔄 재학습 후 스케일링 완료: {X_scaled.shape}")
                    else:
                        # 다른 스케일링 오류는 재발생
                        raise scale_error
                
                # 예측
                probabilities = self.ensemble_classifier.predict_proba(X_scaled)
                logger.info(f"   🎯 예측 완료: {probabilities.shape}")
                
                # 신뢰도 분석
                max_confidences = probabilities.max(axis=1)
                high_confidence_count = (max_confidences >= self.confidence_threshold).sum()
                
                logger.info(f"   📊 신뢰도 분석:")
                logger.info(f"      - 평균 신뢰도: {max_confidences.mean():.3f}")
                logger.info(f"      - 최대 신뢰도: {max_confidences.max():.3f}")
                logger.info(f"      - 최소 신뢰도: {max_confidences.min():.3f}")
                logger.info(f"      - 임계값 이상: {high_confidence_count}개")
                
                # 임시로 낮은 임계값으로 테스트
                for temp_threshold in [0.3, 0.4, 0.5]:
                    temp_count = (max_confidences >= temp_threshold).sum()
                    logger.info(f"      - 임계값 {temp_threshold} 이상: {temp_count}개")
                
                # ML 분류 결과 적용
                for i, merchant in enumerate(unique_merchants):
                    probs = probabilities[i]
                    predicted_idx = np.argmax(probs)
                    confidence = probs[predicted_idx]
                    
                    if confidence >= self.confidence_threshold:
                        predicted_category = self.label_encoder.inverse_transform([predicted_idx])[0]
                        ml_results[merchant] = predicted_category
                        logger.debug(f"      ✅ ML 분류: {merchant} -> {predicted_category} (신뢰도: {confidence:.3f})")
                    else:
                        remaining_candidates.append(merchant)
                        
            except Exception as e:
                logger.error(f"   ⚠️ ML 분류 오류: {e}")
                logger.error(f"   상세 오류: {traceback.format_exc()}")
                # ML 실패 시 모든 항목을 다음 단계로
                remaining_candidates.extend(unique_merchants)
        else:
            logger.info("   ⚠️ ML 모델 없음")
            remaining_candidates.extend(unique_merchants)
        
        logger.info(f"   ML 분류: {len(ml_results)}개")
        logger.info(f"   남은 후보: {len(remaining_candidates)}개")
        
        # 2단계: 키워드 기반 분류 (ML에서 분류되지 않은 항목만)
        keyword_results = {}
        llm_candidates = []
        
        for merchant in remaining_candidates:
            keyword_result = self.keyword_based_classification(merchant)
            if keyword_result:
                keyword_results[merchant] = keyword_result
            else:
                llm_candidates.append(merchant)
        
        logger.info(f"   키워드 분류: {len(keyword_results)}개")
        logger.info(f"   LLM 후보: {len(llm_candidates)}개")
        
        # 3단계: LLM 분류 (ML과 키워드로 분류되지 않은 항목만)
        llm_results = {}
        if llm_candidates and self.ollama_available:
            try:
                llm_results = self.query_llm_for_batch(llm_candidates, batch_num)
            except Exception as e:
                logger.error(f"   ⚠️ LLM 분류 오류: {e}")
                # LLM 실패 시 키워드 기반으로 처리
                for merchant in llm_candidates:
                    keyword_result = self.keyword_based_classification(merchant)
                    llm_results[merchant] = keyword_result if keyword_result else '기타'
        else:
            # LLM 사용 불가 시 키워드 기반으로 처리
            for merchant in llm_candidates:
                keyword_result = self.keyword_based_classification(merchant)
                llm_results[merchant] = keyword_result if keyword_result else '기타'
        
        logger.info(f"   LLM 분류: {len(llm_candidates)}개")
        
        # 결과 통합 (ML → 키워드 → LLM 순서)
        all_results = {**ml_results, **keyword_results, **llm_results}
        batch_df['카테고리'] = batch_df[self.merchant_column].map(all_results)
        batch_df['카테고리'] = batch_df['카테고리'].fillna('기타')
        
        # 분류 방법별 통계
        logger.info(f"\n   📊 분류 방법별 결과 (ML 우선):")
        logger.info(f"      - ML: {len(ml_results)}개")
        logger.info(f"      - 키워드: {len(keyword_results)}개")
        logger.info(f"      - LLM: {len([k for k in llm_results.keys() if k in llm_candidates])}개")
        
        # 4단계: 누적 학습 및 저장
        try:
            logger.info(f"\n📚 분류 완료 후 누적 학습 시작 (배치 {batch_num}) - 데이터 수: {len(batch_df)}")
            
            # 먼저 학습 데이터를 업데이트
            self.update_training_data_with_corrections(batch_df, batch_num)
            
            # 그 다음 모델을 재학습
            self.retrain_enhanced_model(batch_num)
            
            logger.info(f"✅ 모델 누적 학습 및 저장 완료! 총 누적 샘플 수: {len(self.all_training_data)}")

        except Exception as e:
            logger.error(f"❌ 누적 학습 실패: {e}")
            logger.error(traceback.format_exc())

        return batch_df
    
    def load_excel_data(self, excel_path: str) -> pd.DataFrame:
        """엑셀 데이터 로드"""
        logger.info(f"📁 엑셀 파일 로드: {excel_path}")
        df = pd.read_excel(excel_path)
        logger.info(f"📊 전체 데이터: {len(df)}개 행")
        
        df_filtered = df[df[self.merchant_column].notna()].copy()
        logger.info(f"📊 유효한 데이터: {len(df_filtered)}개 행")
        
        return df_filtered
    
    def query_ollama(self, prompt: str, model: str = None, timeout: int = 60) -> str:
        """Ollama 호출 (실제 LLM 처리)"""
        if model is None:
            model = self.ollama_model
            
        try:
            logger.info(f"🤖 Ollama 호출 시작 (모델: {model})")
            start_time = time.time()
            
            result = subprocess.run(
                ['ollama', 'run', model],
                input=prompt.encode('utf-8'),
                capture_output=True,
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"🤖 Ollama 호출 완료 ({elapsed_time:.2f}초)")
            
            if result.returncode != 0:
                logger.error(f"Ollama 오류: {result.stderr.decode('utf-8')}")
                raise Exception(f"Ollama 실행 실패: {result.stderr.decode('utf-8')}")
            
            return result.stdout.decode('utf-8')
            
        except subprocess.TimeoutExpired:
            logger.error(f"Ollama 타임아웃 ({timeout}초)")
            raise Exception(f"Ollama 호출 타임아웃 ({timeout}초)")
        except Exception as e:
            logger.error(f"Ollama 호출 오류: {e}")
            raise e
    
    def build_enhanced_llm_prompt(self, merchants: List[str]) -> str:
        """고도화된 LLM 프롬프트"""
        categories_text = ", ".join(self.CATEGORIES)
        
        # 카테고리별 예시 추가
        examples = {
            "식비": "맛집, 한식당, 중국집, 일식집, 치킨집, 피자집",
            "카페/간식": "스타벅스, 이디야커피, 베이커리, 도넛, 아이스크림",
            "편의점/마트/잡화": "GS25, CU, 이마트, 다이소, 올리브영",
            "교통/대중교통": "지하철, 버스, 택시, 카카오택시",
            "자동차/주유소": "GS칼텍스, SK에너지, 주유소, 세차장",
            "쇼핑": "백화점, 아울렛, 의류매장, 신발가게",
            "의료/건강": "병원, 약국, 치과, 한의원"
        }
        
        prompt = f"""가맹점명을 정확하게 분류해주세요.

사용 가능한 카테고리:
{categories_text}

카테고리별 예시:
{chr(10).join([f"- {cat}: {ex}" for cat, ex in examples.items()])}

분류 규칙:
1. 가맹점명에서 핵심 키워드를 파악하세요
2. 가장 적절한 카테고리를 선택하세요
3. 확실하지 않으면 "기타"를 선택하세요
4. 반드시 JSON 형식으로만 응답하세요

JSON 형식 예시:
{{"스타벅스 강남점": "카페/간식", "GS25 서초점": "편의점/마트/잡화", "김밥천국": "식비"}}

분류할 가맹점 목록:
"""
        for i, merchant in enumerate(merchants, 1):
            prompt += f"{i}. {merchant}\n"
            
        prompt += "\nJSON 응답:"
        return prompt
    
    def parse_llm_response(self, response: str) -> Dict[str, str]:
        """LLM 응답 파싱 (개선된 버전)"""
        logger.info("🔍 LLM 응답 파싱 중...")
        
        # JSON 블록 찾기 (여러 패턴 시도)
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # 중첩 가능한 JSON
            r'\{.*?\}',  # 기본 JSON
            r'```json\s*(\{.*?\})\s*```',  # 마크다운 JSON 블록
            r'```\s*(\{.*?\})\s*```'  # 일반 코드 블록
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # 마크다운 블록에서 추출한 경우
                    if isinstance(match, tuple):
                        json_str = match[0] if match else match
                    else:
                        json_str = match
                    
                    # JSON 파싱 시도
                    result = json.loads(json_str)
                    
                    # 유효한 카테고리만 필터링
                    valid_result = {}
                    for key, value in result.items():
                        if value in self.CATEGORIES:
                            valid_result[key] = value
                        else:
                            logger.warning(f"잘못된 카테고리: {value} -> 기타로 변경")
                            valid_result[key] = "기타"
                    
                    logger.info(f"✅ JSON 파싱 성공: {len(valid_result)}개 항목")
                    return valid_result
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 실패: {e}")
                    continue
        
        # JSON 파싱 실패 시 텍스트 파싱 시도
        logger.warning("JSON 파싱 실패, 텍스트 파싱 시도...")
        return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> Dict[str, str]:
        """텍스트 응답 파싱 (JSON 실패 시 백업)"""
        result = {}
        lines = response.split('\n')
        
        for line in lines:
            # "가맹점명: 카테고리" 형식 찾기
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    merchant = parts[0].strip().strip('"\'')
                    category = parts[1].strip().strip('"\'')
                    
                    if category in self.CATEGORIES:
                        result[merchant] = category
            
            # "가맹점명 -> 카테고리" 형식 찾기
            elif '->' in line:
                parts = line.split('->', 1)
                if len(parts) == 2:
                    merchant = parts[0].strip().strip('"\'')
                    category = parts[1].strip().strip('"\'')
                    
                    if category in self.CATEGORIES:
                        result[merchant] = category
        
        return result
    
    def query_llm_for_batch(self, merchants: List[str], batch_num: int) -> Dict[str, str]:
        """LLM 분류 (배치 단위)"""
        if not merchants:
            return {}
        
        # 배치를 더 작은 청크로 나누기 (LLM 처리 안정성 향상)
        chunk_size = 10
        all_results = {}
        
        for i in range(0, len(merchants), chunk_size):
            chunk = merchants[i:i + chunk_size]
            logger.info(f"   청크 {i//chunk_size + 1}: {len(chunk)}개 처리 중...")
            
            prompt = self.build_enhanced_llm_prompt(chunk)
            
            try:
                response = self.query_ollama(prompt)
                results = self.parse_llm_response(response)
                all_results.update(results)
                
                # 처리되지 않은 항목들은 키워드 기반으로 분류
                for merchant in chunk:
                    if merchant not in results:
                        keyword_result = self.keyword_based_classification(merchant)
                        all_results[merchant] = keyword_result if keyword_result else '기타'
                
            except Exception as e:
                logger.error(f"   청크 처리 실패: {e}")
                # 실패한 청크는 키워드 기반으로 처리
                for merchant in chunk:
                    keyword_result = self.keyword_based_classification(merchant)
                    all_results[merchant] = keyword_result if keyword_result else '기타'
        
        return all_results
    
    def classify_batch_with_llm(self, batch_df: pd.DataFrame, batch_num: int, month: str) -> pd.DataFrame:
        """LLM으로 배치 분류"""
        logger.info(f"\n🤖 배치 {batch_num} ({month}) - LLM 분류")
        
        unique_merchants = batch_df[self.merchant_column].unique().tolist()
        logger.info(f"   고유 가맹점: {len(unique_merchants)}개")
        
        # Ollama 사용 가능 여부 확인
        if not self.ollama_available:
            logger.warning("   ⚠️ Ollama 사용 불가, 키워드 기반 분류 사용")
            return self._fallback_keyword_classification(batch_df, unique_merchants)
        
        try:
            llm_results = self.query_llm_for_batch(unique_merchants, batch_num)
        except Exception as e:
            logger.error(f"   ⚠️ LLM 분류 실패: {e}, 키워드 기반 분류로 전환")
            return self._fallback_keyword_classification(batch_df, unique_merchants)
        
        # 결과 적용
        batch_df['카테고리'] = batch_df[self.merchant_column].map(llm_results)
        batch_df['카테고리'] = batch_df['카테고리'].fillna('기타')
        
        return batch_df
    
    def _fallback_keyword_classification(self, batch_df: pd.DataFrame, unique_merchants: List[str]) -> pd.DataFrame:
        """키워드 기반 백업 분류"""
        keyword_results = {}
        
        for merchant in unique_merchants:
            keyword_result = self.keyword_based_classification(merchant)
            keyword_results[merchant] = keyword_result if keyword_result else '기타'
        
        batch_df['카테고리'] = batch_df[self.merchant_column].map(keyword_results)
        batch_df['카테고리'] = batch_df['카테고리'].fillna('기타')
        
        return batch_df
    
    
    
    def merge_with_existing_data(self, new_batch_data: pd.DataFrame):
        """새로운 배치 데이터를 기존 학습 데이터와 병합 - 수정됨"""
        if len(self.all_training_data) == 0:
            self.all_training_data = new_batch_data.copy()
            logger.info(f"📝 첫 번째 학습 데이터 추가: {len(new_batch_data)}개")
        else:
            # 기존 데이터와 새 데이터를 단순히 concat (중복 제거 안 함)
            self.all_training_data = pd.concat([self.all_training_data, new_batch_data], ignore_index=True)
            logger.info(f"📝 새로운 학습 데이터 추가: {len(new_batch_data)}개")
        
        logger.info(f"📊 총 누적 학습 데이터: {len(self.all_training_data)}개")

    def update_training_data_with_corrections(self, corrected_batch: pd.DataFrame, batch_num: int):
        """수정된 배치 데이터로 학습 데이터 업데이트 - 수정됨"""
        self.current_batch = batch_num
        
        # 기존 데이터와 병합 (항상 추가)
        self.merge_with_existing_data(corrected_batch)
        
        # 자동 저장
        self.save_complete_model(batch_num)
        
        logger.info(f"💾 배치 {batch_num} 학습 데이터 업데이트 및 저장 완료")
        
    def manual_correction_step(self, batch_df: pd.DataFrame, batch_num: int, month: str) -> pd.DataFrame:
        """수정 단계 (웹에서는 API로 대체)"""
        logger.info(f"\n✏️ 배치 {batch_num} ({month}) 수정 단계")
        
        batch_file = f'batch_{batch_num}_{month}_classified.xlsx'
        batch_df.to_excel(batch_file, index=False)
        
        category_counts = batch_df['카테고리'].value_counts()
        logger.info(f"   분류 결과:")
        for category, count in category_counts.head(8).items():
            logger.info(f"   - {category}: {count}개")
        
        logger.info(f"\n📝 '{batch_file}' 파일을 열어서 직접 수정해주세요.")
        input("✅ 수정 완료 후 Enter를 눌러주세요...")
        
        try:
            corrected_df = pd.read_excel(batch_file)
            logger.info(f"   ✅ 수정된 데이터 로드: {len(corrected_df)}개 행")
            
            return corrected_df
        except Exception as e:
            logger.error(f"   ⚠️ 파일 로드 오류: {e}")
            return batch_df
    
    def process_monthly_incremental_learning(self, excel_path: str):
        """월별 점진적 학습 실행 (기존 모델 업데이트)"""
        logger.info("🚀 지속적 모델 업데이트 시작! (월별 배치)")
        logger.info(f"💡 모델명: {self.model_name}")
        logger.info(f"💡 기존 학습 데이터: {len(self.all_training_data)}개")
        logger.info(f"💡 기존 배치: {len(self.performance_history)}개")
        
        # 데이터 로드 및 월별 배치 생성
        df = self.load_excel_data(excel_path)
        monthly_batches = self.create_monthly_batches(df)
        
        if not monthly_batches:
            logger.error("❌ 처리할 배치가 없습니다.")
            return
        
        total_batches = len(monthly_batches)
        start_batch = len(self.performance_history) + 1  # 기존 배치 이후부터 시작
        
        for i, batch_df in enumerate(monthly_batches):
            batch_num = start_batch + i
            month = batch_df['년월'].iloc[0] if '년월' in batch_df.columns else f"배치{batch_num}"
            
            logger.info(f"\n{'='*70}")
            logger.info(f"📦 배치 {batch_num} ({month}) 처리 중")
            logger.info(f"   해당 월 거래: {len(batch_df)}개")
            logger.info(f"   기존 누적 학습 데이터: {len(self.all_training_data)}개")
            logger.info(f"   총 진행: {i+1}/{total_batches}")
            logger.info(f"{'='*70}")
            
            # 1단계: 분류 (기존 모델이 있으면 하이브리드 사용)
            if self.ensemble_classifier is None:
                # 첫 번째 학습이거나 모델이 없는 경우
                classified_batch = self.classify_batch_with_llm(batch_df, batch_num, month)
            else:
                # 기존 모델이 있는 경우 하이브리드 분류
                classified_batch = self.classify_batch_with_enhanced_model(batch_df, batch_num, month)
            
            # 2단계: 수동 수정
            corrected_batch = self.manual_correction_step(classified_batch, batch_num, month)
            
            # 3단계: 기존 모델 업데이트
            performance = self.retrain_enhanced_model(batch_num)
            
            # 결과 출력
            if performance:
                logger.info(f"\n📊 배치 {batch_num} ({month}) 완료:")
                logger.info(f"   현재 배치: {len(corrected_batch)}개")
                logger.info(f"   누적 데이터: {performance['training_samples']}개")
                logger.info(f"   훈련 정확도: {performance['train_accuracy']:.3f}")
                logger.info(f"   교차검증: {performance['cv_mean']:.3f} (±{performance['cv_std']:.3f})")
                logger.info(f"   신뢰도 임계값: {performance['confidence_threshold']:.3f}")
                
                # 성능 개선 확인
                if len(self.performance_history) >= 2:
                    prev_acc = self.performance_history[-2]['train_accuracy']
                    current_acc = performance['train_accuracy']
                    improvement = current_acc - prev_acc
                    if improvement > 0:
                        logger.info(f"   📈 성능 향상: +{improvement:.3f}")
                    elif improvement < 0:
                        logger.info(f"   📉 성능 하락: {improvement:.3f}")
                    else:
                        logger.info(f"   ➡️ 성능 유지")
            
            # 다음 배치 진행 확인
            if i + 1 < total_batches:
                continue_process = input(f"\n🔄 다음 배치를 진행하시겠습니까? (y/n): ")
                if continue_process.lower() != 'y':
                    logger.info("🛑 처리를 중단합니다.")
                    break
        
        self.show_enhanced_final_summary()
    
    def show_enhanced_final_summary(self):
        """고도화된 최종 결과 요약"""
        logger.info(f"\n🎯 === 지속적 모델 업데이트 최종 결과 ===")
        logger.info(f"모델명: {self.model_name}")
        logger.info(f"처리된 배치: {len(self.performance_history)}개")
        logger.info(f"최종 학습 데이터: {len(self.all_training_data)}개")
        
        if self.performance_history:
            initial_performance = self.performance_history[0]
            final_performance = self.performance_history[-1]
            
            logger.info(f"\n📈 성능 변화:")
            logger.info(f"   초기 훈련 정확도: {initial_performance['train_accuracy']:.3f}")
            logger.info(f"   최종 훈련 정확도: {final_performance['train_accuracy']:.3f}")
            logger.info(f"   전체 개선: {final_performance['train_accuracy'] - initial_performance['train_accuracy']:+.3f}")
            
            logger.info(f"\n📊 교차검증 성능:")
            logger.info(f"   최종 CV 평균: {final_performance['cv_mean']:.3f}")
            logger.info(f"   최종 CV 표준편차: {final_performance['cv_std']:.3f}")
            
            logger.info(f"\n📊 최근 5개 배치 성능:")
            for perf in self.performance_history[-5:]:
                batch = perf['batch']
                train_acc = perf['train_accuracy']
                cv_mean = perf['cv_mean']
                samples = perf['training_samples']
                logger.info(f"   배치 {batch}: 훈련={train_acc:.3f}, CV={cv_mean:.3f} (누적 {samples}개)")
            
            # 최종 카테고리 분포
            if len(self.all_training_data) > 0:
                final_distribution = self.all_training_data['카테고리'].value_counts()
                logger.info(f"\n🏷️ 최종 카테고리 분포:")
                for category, count in final_distribution.head(12).items():
                    percentage = count / len(self.all_training_data) * 100
                    logger.info(f"   - {category}: {count}개 ({percentage:.1f}%)")
        
        # 모델 저장 상태
        logger.info(f"\n💾 모델 저장 상태:")
        logger.info(f"   최신 모델: {self.model_dir / f'{self.model_name}_latest.pkl'}")
        logger.info(f"   백업 모델: {len(list(self.model_dir.glob(f'{self.model_name}_batch_*.pkl')))}개")
        logger.info(f"   학습 데이터: {self.data_dir / f'{self.model_name}_training_data.xlsx'}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """현재 모델 상태 반환"""
        status = {
            'model_name': self.model_name,
            'model_trained': self.ensemble_classifier is not None,
            'total_training_samples': len(self.all_training_data),
            'total_batches': len(self.performance_history),
            'current_batch': self.current_batch,
            'confidence_threshold': self.confidence_threshold,
            'ollama_available': self.ollama_available,
            'categories_count': len(self.CATEGORIES),
            'keyword_patterns_loaded': len(self.keyword_patterns)
        }
        
        if self.performance_history:
            latest_perf = self.performance_history[-1]
            status.update({
                'latest_accuracy': latest_perf['train_accuracy'],
                'latest_cv_mean': latest_perf['cv_mean'],
                'latest_cv_std': latest_perf['cv_std']
            })
        
        return status
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """하나의 통합 모델 정보만 반환"""
        models = []
        
        # 통합 모델 메타데이터만 확인
        metadata_file = self.model_dir / f'{self.model_name}_metadata.json'
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                models.append(metadata)
            except Exception as e:
                logger.warning(f"메타데이터 로드 실패: {metadata_file}, {e}")
        
        return models
    
    def switch_model(self, model_name: str):
        """모델 전환 불가 - 하나의 통합 모델만 사용"""
        logger.warning("⚠️ 하나의 통합 모델만 사용됩니다. 모델 전환이 불가능합니다.")
        return False


def main():
    """메인 실행 함수"""
    print("🎯 월별 배치 분할 카드 분류기")
    print("💡 특징:")
    print("   - 기존 학습된 모델 자동 로드")
    print("   - 카드사용 월별 배치 분할")
    print("   - ML 분류 디버깅 강화")
    print("   - 하나의 모델 지속적 업데이트")
    print("   - 신뢰도 임계값 동적 조정")
    
    model_name = input("\n🏷️ 모델명 (기본값: default_model): ").strip()
    if not model_name:
        model_name = 'default_model'
    
    excel_file = input("📁 엑셀 파일 경로 (기본값: card.xlsx): ").strip()
    if not excel_file:
        excel_file = 'card.xlsx'
    
    date_column = input("📅 날짜 컬럼명 (기본값: 이용일자): ").strip()
    if not date_column:
        date_column = '이용일자'
    
    merchant_column = input("🏪 가맹점 컬럼명 (기본값: 가맹점명): ").strip()
    if not merchant_column:
        merchant_column = '가맹점명'
    
    # 분류기 초기화 (기존 모델 자동 로드)
    categorizer = PersistentIncrementalCategorizer(
        merchant_column=merchant_column,
        date_column=date_column,
        model_name=model_name
    )
    
    # 모델 상태 출력
    status = categorizer.get_model_status()
    print(f"\n📊 현재 모델 상태:")
    print(f"   - 모델 학습됨: {status['model_trained']}")
    print(f"   - 누적 샘플: {status['total_training_samples']}개")
    print(f"   - 처리된 배치: {status['total_batches']}개")
    if status['model_trained']:
        print(f"   - 현재 정확도: {status.get('latest_accuracy', 0):.3f}")
        print(f"   - 신뢰도 임계값: {status['confidence_threshold']:.3f}")
    
    # 모델 진단 실행
    if status['model_trained']:
        diagnose = input("\n🔍 모델 진단을 실행하시겠습니까? (y/n): ")
        if diagnose.lower() == 'y':
            categorizer.diagnose_ml_model()
    
    # 신뢰도 임계값 수동 조정 옵션
    adjust_threshold = input(f"\n🎯 신뢰도 임계값을 수동 조정하시겠습니까? 현재: {categorizer.confidence_threshold:.3f} (y/n): ")
    if adjust_threshold.lower() == 'y':
        new_threshold = float(input("새로운 임계값 (0.1-0.9): "))
        if 0.1 <= new_threshold <= 0.9:
            categorizer.confidence_threshold = new_threshold
            print(f"✅ 임계값 조정 완료: {new_threshold:.3f}")
        else:
            print("⚠️ 잘못된 값입니다. 기존 값 유지")
    
    # 월별 점진적 학습 실행
    print("\n🚀 월별 점진적 학습을 시작합니다...")
    categorizer.process_monthly_incremental_learning(excel_file)


def test_model_functionality():
    """모델 기능 테스트"""
    print("\n🧪 모델 기능 테스트:")
    
    # 테스트 모델 생성
    test_classifier = PersistentIncrementalCategorizer(model_name="test_model")
    
    # 상태 확인
    status = test_classifier.get_model_status()
    print(f"✅ 테스트 모델 상태: {status}")
    
    # 키워드 분류 테스트
    test_merchants = ["스타벅스 강남점", "GS25 서초점", "김밥천국", "현대오일뱅크"]
    print(f"\n🔤 키워드 분류 테스트:")
    for merchant in test_merchants:
        result = test_classifier.keyword_based_classification(merchant)
        print(f"   {merchant} -> {result if result else '키워드 매칭 없음'}")


if __name__ == "__main__":
    print("🎯 지속적 모델 업데이트 카드 분류기")
    print("📅 새로운 기능:")
    print("   ✅ 기존 학습된 모델 자동 로드")
    print("   ✅ 새 데이터로 기존 모델 업데이트")
    print("   ✅ 중단 후 재시작 가능")
    print("   ✅ 모델 백업 및 버전 관리")
    print("   ✅ 여러 모델 동시 관리")
    print("   ✅ 월별 배치 자동 분할")
    print("   ✅ ML 분류 디버깅 강화")
    
    # 기능 테스트 실행
    test_model_functionality()
    
    # 메인 실행
    main()