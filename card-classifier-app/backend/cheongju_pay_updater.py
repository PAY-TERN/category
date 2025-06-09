#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
청주페이 카테고리 업데이트 스크립트
- 카드 분류 결과 CSV와 청주사랑 상품권 가맹점 CSV를 비교
- 겹치는 가맹점의 카테고리를 "청주페이"로 변경
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Set, Dict, Tuple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

class CheongJuPayUpdater:
    """청주페이 카테고리 업데이트 클래스"""
    
    def __init__(self):
        self.classification_df = None
        self.cheongju_merchants_df = None
        self.matched_merchants = set()
        self.updated_count = 0
        

    def load_classification_results(self, file_path: str, merchant_column: str = '가맹점명') -> pd.DataFrame:
        """카드 분류 결과 CSV 파일 로드"""
        logger.info(f"📁 카드 분류 결과 로드: {file_path}")
        
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                # CSV 파일 인코딩 자동 감지
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='cp949')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='euc-kr')
            
            logger.info(f"📊 로드된 데이터: {len(df)}개 행, {len(df.columns)}개 컬럼")
            logger.info(f"컬럼명: {list(df.columns)}")
            
            # 가맹점 컬럼 확인
            if merchant_column not in df.columns:
                # 가맹점 컬럼 자동 감지
                possible_columns = ['가맹점명', '가맹점', 'merchant', 'store', '상점명', '업체명']
                found_column = None
                
                for col in df.columns:
                    for possible in possible_columns:
                        if possible in col:
                            found_column = col
                            break
                    if found_column:
                        break
                
                if found_column:
                    logger.info(f"🔍 가맹점 컬럼 자동 감지: {found_column}")
                    merchant_column = found_column
                else:
                    raise ValueError(f"가맹점 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
            
            # 카테고리 컬럼 확인
            if '카테고리' not in df.columns:
                # 카테고리 컬럼 생성
                df['카테고리'] = '기타'
                logger.info("⚠️ 카테고리 컬럼이 없어서 '기타'로 초기화했습니다.")
            
            self.classification_df = df
            self.merchant_column = merchant_column
            
            # 가맹점명 분포 확인
            unique_merchants = df[merchant_column].nunique()
            logger.info(f"📊 고유 가맹점 수: {unique_merchants}개")
            
            # 기존 카테고리 분포
            if '카테고리' in df.columns:
                category_dist = df['카테고리'].value_counts()
                logger.info(f"📊 기존 카테고리 분포:")
                for category, count in category_dist.head(10).items():
                    logger.info(f"   - {category}: {count}개")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 파일 로드 실패: {e}")
            raise e
    
    def load_cheongju_merchants(self, file_path: str, merchant_column: str = None) -> pd.DataFrame:
        """청주사랑 상품권 가맹점 CSV 파일 로드"""
        logger.info(f"📁 청주사랑 상품권 가맹점 로드: {file_path}")
        
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                # CSV 파일 인코딩 자동 감지
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='cp949')
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding='euc-kr')
            
            logger.info(f"📊 로드된 청주 가맹점: {len(df)}개 행, {len(df.columns)}개 컬럼")
            logger.info(f"컬럼명: {list(df.columns)}")
            
            # 가맹점 컬럼 자동 감지
            if merchant_column is None:
                possible_columns = ['가맹점명', '가맹점', 'merchant', 'store', '상점명', '업체명', '상호명', '업소명']
                found_column = None
                
                for col in df.columns:
                    for possible in possible_columns:
                        if possible in col:
                            found_column = col
                            break
                    if found_column:
                        break
                
                if found_column:
                    merchant_column = found_column
                    logger.info(f"🔍 청주 가맹점 컬럼 자동 감지: {found_column}")
                else:
                    # 첫 번째 컬럼을 가맹점명으로 사용
                    merchant_column = df.columns[0]
                    logger.warning(f"⚠️ 가맹점 컬럼을 찾지 못해 첫 번째 컬럼 사용: {merchant_column}")
            
            self.cheongju_merchants_df = df
            self.cheongju_merchant_column = merchant_column
            
            # 고유 가맹점 수 확인
            unique_cheongju = df[merchant_column].nunique()
            logger.info(f"📊 청주 고유 가맹점 수: {unique_cheongju}개")
            
            # 샘플 데이터 출력
            logger.info(f"📋 청주 가맹점 샘플:")
            for i, merchant in enumerate(df[merchant_column].head(5)):
                logger.info(f"   {i+1}. {merchant}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 청주 가맹점 파일 로드 실패: {e}")
            raise e
    
    def normalize_merchant_name(self, name: str) -> str:
        """지점별 고유성을 유지하는 정규화"""
        if pd.isna(name):
            return ""
        
        name = str(name).strip()
        name = re.sub(r'\s+', ' ', name)  # 공백 정규화
        name = name.lower()
        
        # ❌ 기존: 모든 접미사 제거
        # ✅ 개선: 지점 정보는 보존하고 일반 접미사만 제거
        general_suffixes = ['(주)', '㈜', '서비스']  # 지점명이 아닌 것들만
        for suffix in general_suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)].strip()
        
        return name
    
    def find_matching_merchants(self, exact_match: bool = True, fuzzy_match: bool = True) -> Set[str]:
        """매칭되는 가맹점 찾기"""
        logger.info("🔍 가맹점 매칭 시작...")
        
        if self.classification_df is None or self.cheongju_merchants_df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 분류 결과의 가맹점명들
        classification_merchants = set(self.classification_df[self.merchant_column].dropna().astype(str))
        
        # 청주 가맹점명들
        cheongju_merchants = set(self.cheongju_merchants_df[self.cheongju_merchant_column].dropna().astype(str))
        
        matched = set()
        
        # 1단계: 정확한 매칭
        if exact_match:
            exact_matches = classification_merchants.intersection(cheongju_merchants)
            matched.update(exact_matches)
            logger.info(f"✅ 정확한 매칭: {len(exact_matches)}개")
            
            if len(exact_matches) > 0:
                logger.info("📋 정확한 매칭 샘플:")
                for i, merchant in enumerate(list(exact_matches)[:5]):
                    logger.info(f"   {i+1}. {merchant}")
        
        # 2단계: 정규화된 매칭
        if fuzzy_match:
            logger.info("🔄 정규화된 매칭 시도...")
            
            # 정규화된 가맹점명 딕셔너리 생성
            normalized_classification = {
                self.normalize_merchant_name(m): m 
                for m in classification_merchants
            }
            
            normalized_cheongju = {
                self.normalize_merchant_name(m): m 
                for m in cheongju_merchants
            }
            
            # 정규화된 이름으로 매칭
            normalized_matches = set(normalized_classification.keys()).intersection(
                set(normalized_cheongju.keys())
            )
            
            # 원본 이름으로 변환
            fuzzy_matches = set()
            for norm_name in normalized_matches:
                if norm_name in normalized_classification:
                    original_name = normalized_classification[norm_name]
                    if original_name not in matched:  # 중복 제거
                        fuzzy_matches.add(original_name)
            
            matched.update(fuzzy_matches)
            logger.info(f"✅ 정규화 매칭: {len(fuzzy_matches)}개 (중복 제외)")
            
            if len(fuzzy_matches) > 0:
                logger.info("📋 정규화 매칭 샘플:")
                for i, merchant in enumerate(list(fuzzy_matches)[:5]):
                    normalized = self.normalize_merchant_name(merchant)
                    logger.info(f"   {i+1}. {merchant} -> {normalized}")
        
        # 3단계: 부분 문자열 매칭 (선택적)
        partial_matches = set()
        if len(matched) < 10:  # 매칭이 적을 때만 실행
            logger.info("🔄 부분 문자열 매칭 시도...")
            
            for c_merchant in classification_merchants:
                if c_merchant in matched:
                    continue
                    
                c_normalized = self.normalize_merchant_name(c_merchant)
                if len(c_normalized) < 3:  # 너무 짧은 이름은 제외
                    continue
                
                for j_merchant in cheongju_merchants:
                    j_normalized = self.normalize_merchant_name(j_merchant)
                    
                    # 부분 문자열 확인 (양방향)
                    if (len(c_normalized) >= 3 and c_normalized in j_normalized) or \
                       (len(j_normalized) >= 3 and j_normalized in c_normalized):
                        partial_matches.add(c_merchant)
                        logger.debug(f"부분 매칭: {c_merchant} <-> {j_merchant}")
                        break
            
            matched.update(partial_matches)
            logger.info(f"✅ 부분 매칭: {len(partial_matches)}개")
        
        self.matched_merchants = matched
        logger.info(f"🎯 총 매칭된 가맹점: {len(matched)}개")
        
        return matched
    
    def update_categories_to_cheongju_pay(self) -> int:
        """매칭된 가맹점들의 카테고리를 "청주페이"로 업데이트"""
        logger.info("🔄 카테고리를 '청주페이'로 업데이트 중...")
        
        if len(self.matched_merchants) == 0:
            logger.warning("⚠️ 매칭된 가맹점이 없습니다.")
            return 0
        
        # 카테고리 업데이트
        updated_count = 0
        update_log = []
        
        for merchant in self.matched_merchants:
            # 해당 가맹점의 모든 행을 업데이트
            mask = self.classification_df[self.merchant_column] == merchant
            affected_rows = mask.sum()
            
            if affected_rows > 0:
                # 기존 카테고리 저장 (로그용)
                old_categories = self.classification_df.loc[mask, '카테고리'].unique()
                
                # 카테고리 업데이트
                self.classification_df.loc[mask, '카테고리'] = '청주페이'
                
                updated_count += affected_rows
                update_log.append({
                    'merchant': merchant,
                    'rows_affected': affected_rows,
                    'old_categories': list(old_categories)
                })
        
        self.updated_count = updated_count
        
        logger.info(f"✅ 업데이트 완료: {updated_count}개 행")
        logger.info(f"📊 업데이트된 가맹점: {len(self.matched_merchants)}개")
        
        # 업데이트 로그 출력
        logger.info("📋 업데이트 상세 내역:")
        for i, log in enumerate(update_log[:10]):  # 상위 10개만 출력
            merchant = log['merchant']
            rows = log['rows_affected']
            old_cats = ', '.join(log['old_categories'])
            logger.info(f"   {i+1}. {merchant}: {rows}개 행 ({old_cats} -> 청주페이)")
        
        if len(update_log) > 10:
            logger.info(f"   ... 외 {len(update_log) - 10}개 가맹점")
        
        return updated_count
    
    def save_updated_results(self, output_path: str = None) -> str:
        """업데이트된 결과를 파일로 저장"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"card_classification_with_cheongju_pay_{timestamp}.xlsx"
        
        logger.info(f"💾 업데이트된 결과 저장: {output_path}")
        
        try:
            # 엑셀 파일로 저장
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 메인 결과
                self.classification_df.to_excel(writer, sheet_name='Classification_Results', index=False)
                
                # 매칭된 가맹점 목록
                if len(self.matched_merchants) > 0:
                    matched_df = pd.DataFrame({
                        '매칭된_가맹점': list(self.matched_merchants),
                        '카테고리': '청주페이'
                    })
                    matched_df.to_excel(writer, sheet_name='Matched_Merchants', index=False)
                
                # 통계 정보
                stats_df = pd.DataFrame({
                    '항목': [
                        '전체 거래 건수',
                        '고유 가맹점 수',
                        '매칭된 가맹점 수',
                        '청주페이로 변경된 거래 건수',
                        '청주페이 비율(%)'
                    ],
                    '값': [
                        len(self.classification_df),
                        self.classification_df[self.merchant_column].nunique(),
                        len(self.matched_merchants),
                        self.updated_count,
                        round(self.updated_count / len(self.classification_df) * 100, 2)
                    ]
                })
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            logger.info(f"✅ 저장 완료: {output_path}")
            
            # 최종 카테고리 분포 출력
            final_dist = self.classification_df['카테고리'].value_counts()
            logger.info(f"📊 최종 카테고리 분포:")
            for category, count in final_dist.items():
                percentage = count / len(self.classification_df) * 100
                logger.info(f"   - {category}: {count}개 ({percentage:.1f}%)")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ 저장 실패: {e}")
            raise e
    
    def generate_summary_report(self) -> Dict:
        """처리 결과 요약 리포트 생성"""
        if self.classification_df is None:
            return {}
        
        total_transactions = len(self.classification_df)
        unique_merchants = self.classification_df[self.merchant_column].nunique()
        
        # 카테고리별 분포
        category_dist = self.classification_df['카테고리'].value_counts().to_dict()
        
        # 청주페이 관련 통계
        cheongju_pay_count = category_dist.get('청주페이', 0)
        cheongju_pay_ratio = (cheongju_pay_count / total_transactions * 100) if total_transactions > 0 else 0
        
        summary = {
            'total_transactions': total_transactions,
            'unique_merchants': unique_merchants,
            'matched_merchants_count': len(self.matched_merchants),
            'updated_transactions': self.updated_count,
            'cheongju_pay_transactions': cheongju_pay_count,
            'cheongju_pay_ratio': round(cheongju_pay_ratio, 2),
            'category_distribution': category_dist,
            'matched_merchants': list(self.matched_merchants)
        }
        
        return summary

def main():
    """메인 실행 함수"""
    print("🏪 청주페이 카테고리 업데이트 스크립트")
    print("=" * 50)
    
    updater = CheongJuPayUpdater()
    
    # 1. 카드 분류 결과 파일 로드 (고정 경로)
    classification_file = "card.xlsx"
    merchant_column = '가맹점명'
    
    print(f"📁 카드 분류 결과 파일: {classification_file}")
    
    try:
        updater.load_classification_results(classification_file, merchant_column)
    except Exception as e:
        print(f"❌ 카드 분류 결과 로드 실패: {e}")
        print("💡 'card.xlsx' 파일이 현재 디렉토리에 있는지 확인해주세요.")
        return
    
    # 2. 청주사랑 상품권 가맹점 파일 로드 (고정 경로)
    cheongju_file = "cheongju-pay_u.csv"
    
    print(f"📁 청주사랑 상품권 가맹점 파일: {cheongju_file}")
    
    try:
        updater.load_cheongju_merchants(cheongju_file)
    except Exception as e:
        print(f"❌ 청주 가맹점 파일 로드 실패: {e}")
        print("💡 'cheongju-pay_u.csv' 파일이 현재 디렉토리에 있는지 확인해주세요.")
        return
    
    # 3. 자동으로 권장 매칭 방식 사용 (정확한 + 정규화 매칭)
    print("\n🔍 매칭 방식: 정확한 매칭 + 정규화 매칭 (자동 선택)")
    
    exact_match = True
    fuzzy_match = True
    
    # 4. 매칭 실행
    try:
        matched_merchants = updater.find_matching_merchants(
            exact_match=exact_match, 
            fuzzy_match=fuzzy_match
        )
        
        if len(matched_merchants) == 0:
            print("⚠️ 매칭된 가맹점이 없습니다.")
            print("💡 다른 매칭 방식을 시도하거나 파일을 확인해주세요.")
            return
        
        # 5. 매칭 결과 확인
        print(f"\n🎯 매칭된 가맹점: {len(matched_merchants)}개")
        print("📋 매칭된 가맹점 샘플:")
        for i, merchant in enumerate(list(matched_merchants)[:10]):
            print(f"   {i+1}. {merchant}")
        
        if len(matched_merchants) > 10:
            print(f"   ... 외 {len(matched_merchants) - 10}개")
        
        # 6. 자동으로 업데이트 진행
        print(f"\n✅ {len(matched_merchants)}개 가맹점의 카테고리를 '청주페이'로 자동 변경합니다...")
        
        # 7. 카테고리 업데이트
        updated_count = updater.update_categories_to_cheongju_pay()
        
        # 8. 결과 자동 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"card_with_cheongju_pay.xlsx"
        
        print(f"💾 결과를 '{output_file}'로 저장합니다...")
        saved_file = updater.save_updated_results(output_file)
        
        # 9. 최종 결과 요약
        summary = updater.generate_summary_report()
        
        print(f"\n🎉 처리 완료!")
        print(f"📊 결과 요약:")
        print(f"   - 전체 거래: {summary['total_transactions']:,}건")
        print(f"   - 고유 가맹점: {summary['unique_merchants']:,}개")
        print(f"   - 매칭된 가맹점: {summary['matched_merchants_count']}개")
        print(f"   - 청주페이로 변경: {summary['updated_transactions']:,}건")
        print(f"   - 청주페이 비율: {summary['cheongju_pay_ratio']}%")
        print(f"💾 저장된 파일: {saved_file}")
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        logger.error(f"처리 오류: {e}")

if __name__ == "__main__":
    main()

