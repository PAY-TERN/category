#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카드 분류기 FastAPI 백엔드 - 하나의 통합 모델 지속적 업데이트
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
import uuid
import subprocess
from pathlib import Path
import shutil
from datetime import datetime, timedelta
import logging
import re
from io import BytesIO

# 통합 모델 분류기 import
from enhanced_classifier import PersistentIncrementalCategorizer

# FastAPI 앱 초기화
app = FastAPI(
    title="카드 분류기 API (통합 모델)",
    description="하나의 모델을 지속적으로 업데이트하는 점진적 학습 시스템",
    version="3.1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수
classifier_instances = {}  # 세션별 분류기 인스턴스 관리
sessions = {}  # 세션 데이터 저장
temp_dir = Path("temp_files")
temp_dir.mkdir(exist_ok=True)

# 하나의 글로벌 분류기 인스턴스 (모든 세션이 공유)
global_classifier = None

# NaN 안전 처리 함수들
def clean_for_json(obj):
    """모든 NaN 값을 안전하게 처리하여 JSON 직렬화 가능하게 만듦"""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        df_clean = obj.fillna('')
        return clean_for_json(df_clean.to_dict('records'))
    elif isinstance(obj, pd.Series):
        series_clean = obj.fillna('')
        return clean_for_json(series_clean.to_dict())
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        if pd.isna(obj):
            return None
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, np.datetime64)):
        try:
            return str(obj)
        except:
            return None
    else:
        return obj

class SafeJSONResponse(JSONResponse):
    """NaN 값을 자동으로 처리하는 안전한 JSON 응답"""
    
    def render(self, content: Any) -> bytes:
        try:
            cleaned_content = clean_for_json(content)
            json_str = json.dumps(
                cleaned_content,
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
            )
            return json_str.encode("utf-8")
        except (TypeError, ValueError) as e:
            logger.error(f"JSON 직렬화 오류: {e}")
            error_content = {
                "error": "JSON 직렬화 실패",
                "message": str(e),
                "original_type": str(type(content).__name__)
            }
            json_str = json.dumps(error_content, ensure_ascii=False)
            return json_str.encode("utf-8")

def get_global_classifier():
    """글로벌 분류기 인스턴스 가져오기"""
    global global_classifier
    if global_classifier is None:
        global_classifier = PersistentIncrementalCategorizer()
        logger.info("🎯 통합 모델 분류기 초기화 완료")
    return global_classifier

# Ollama 연결 확인 함수들
def check_ollama_connection():
    """Ollama 서비스 연결 확인"""
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

def get_available_ollama_models():
    """사용 가능한 Ollama 모델 목록 조회"""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            models = []
            for line in lines[1:]:  # 헤더 제외
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except Exception as e:
        logger.error(f"Ollama 모델 목록 조회 실패: {e}")
        return []

# 날짜 변환 함수들
def convert_excel_date(value):
    """엑셀 시리얼 날짜를 실제 날짜로 변환"""
    if pd.isna(value):
        return None
    
    if isinstance(value, (int, float)):
        try:
            if value >= 60:
                excel_epoch = datetime(1899, 12, 30)
            else:
                excel_epoch = datetime(1899, 12, 31)
            
            converted_date = excel_epoch + timedelta(days=int(value))
            return converted_date.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"날짜 변환 실패: {value} -> {e}")
            return str(value)
    
    elif isinstance(value, datetime):
        return value.strftime('%Y-%m-%d')
    
    elif isinstance(value, str):
        if re.match(r'\d{4}-\d{2}-\d{2}', value):
            return value
        
        try:
            parsed_date = pd.to_datetime(value)
            return parsed_date.strftime('%Y-%m-%d')
        except:
            return value
    
    return str(value)

def is_likely_excel_date(series, threshold=0.3):
    """시리즈가 엑셀 날짜 컬럼인지 판단"""
    if series.dtype not in ['int64', 'float64']:
        return False
        
    numeric_values = series.dropna()
    if len(numeric_values) == 0:
        return False
    
    date_range_values = numeric_values[(numeric_values >= 1) & (numeric_values <= 73050)]
    recent_date_values = numeric_values[(numeric_values >= 40000) & (numeric_values <= 50000)]
    
    recent_ratio = len(recent_date_values) / len(numeric_values)
    total_ratio = len(date_range_values) / len(numeric_values)
    
    return recent_ratio >= threshold or total_ratio >= 0.5

def preprocess_excel_data(df: pd.DataFrame) -> tuple:
    """엑셀 데이터 전처리 - NaN 안전 처리"""
    df_processed = df.copy()
    df_processed = df_processed.fillna('')  # 먼저 모든 NaN 처리
    
    conversion_info = []
    
    logger.info("엑셀 데이터 전처리 시작")
    
    date_keywords = ['일자', '날짜', 'date', 'day', '사용일', '승인일', '거래일', '이용일']
    
    for col in df_processed.columns:
        original_values = df_processed[col].copy()
        should_convert = False
        conversion_method = None
        
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            logger.info(f"날짜 키워드 감지: {col}")
            should_convert = True
            conversion_method = "keyword_based"
        
        elif is_likely_excel_date(df_processed[col]):
            logger.info(f"엑셀 날짜로 추정: {col}")
            should_convert = True
            conversion_method = "numeric_analysis"
        
        if should_convert:
            try:
                converted_values = df_processed[col].apply(convert_excel_date)
                converted_values = converted_values.fillna('')  # NaN 처리
                
                valid_conversions = 0
                total_non_null = 0
                
                for orig, conv in zip(original_values.dropna(), converted_values.dropna()):
                    total_non_null += 1
                    if isinstance(conv, str) and re.match(r'\d{4}-\d{2}-\d{2}', conv):
                        valid_conversions += 1
                
                if total_non_null > 0 and (valid_conversions / total_non_null) >= 0.5:
                    df_processed[col] = converted_values
                    
                    sample_original = str(original_values.dropna().iloc[0]) if len(original_values.dropna()) > 0 else ''
                    sample_converted = str(converted_values.dropna().iloc[0]) if len(converted_values.dropna()) > 0 else ''
                    
                    conversion_info.append({
                        'original': col,
                        'converted': col,
                        'method': conversion_method,
                        'sample_original': sample_original,
                        'sample_converted': sample_converted,
                        'success_rate': valid_conversions / total_non_null
                    })
                    
                    logger.info(f"날짜 변환 완료: {col} (성공률: {valid_conversions}/{total_non_null})")
                else:
                    logger.warning(f"날짜 변환 실패: {col} (성공률 낮음: {valid_conversions}/{total_non_null})")
                    
            except Exception as e:
                logger.error(f"날짜 변환 중 오류: {col} -> {e}")
    
    # 최종 NaN 처리
    df_processed = df_processed.fillna('')
    
    logger.info(f"날짜 전처리 완료. 변환된 컬럼: {len(conversion_info)}개")
    return df_processed, conversion_info

# Pydantic 모델들
class ClassificationRequest(BaseModel):
    session_id: str
    merchant_column: str = "가맹점명"
    date_column: str = "이용일자"

class BatchProcessRequest(BaseModel):
    session_id: str
    batch_number: int
    corrections: Dict[str, str]

class PredictionRequest(BaseModel):
    merchant_names: List[str]
    session_id: str

class BatchInfo(BaseModel):
    batch_number: int
    month: Optional[str]
    data: List[Dict[str, Any]]
    classifications: Dict[str, str]
    confidence_scores: Dict[str, float]

class PerformanceInfo(BaseModel):
    batch: int
    train_accuracy: float
    cv_mean: float
    cv_std: float
    training_samples: int
    categories: int
    confidence_threshold: float

# API 엔드포인트들
@app.get("/")
async def root():
    """루트 엔드포인트 - NaN 안전 처리"""
    try:
        ollama_status = check_ollama_connection()
        available_models = get_available_ollama_models() if ollama_status else []
        
        # 통합 모델 상태 확인
        classifier = get_global_classifier()
        model_status = classifier.get_model_status()
        
        response_data = {
            "message": "카드 분류기 API (통합 모델)",
            "version": "3.1.0",
            "status": "running",
            "ollama_connected": ollama_status,
            "available_ollama_models": available_models,
            "unified_model_status": model_status,
            "features": [
                "하나의 통합 모델 지속적 업데이트",
                "월별 배치 자동 분할",
                "기존 학습 데이터 자동 로드",
                "중단 후 재시작 가능",
                "ML 분류 디버깅 강화"
            ]
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"루트 엔드포인트 오류: {e}")
        error_response = {
            "error": "서버 오류",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.get("/model-status")
async def get_unified_model_status():
    """통합 모델 상태 조회 - NaN 안전 처리"""
    try:
        classifier = get_global_classifier()
        status = classifier.get_model_status()
        
        category_distribution = {}
        if len(classifier.all_training_data) > 0:
            try:
                category_counts = classifier.all_training_data['카테고리'].value_counts()
                category_distribution = category_counts.to_dict()
            except Exception as e:
                logger.warning(f"카테고리 분포 계산 실패: {e}")
        
        response_data = {
            "model_info": status,
            "performance_history": classifier.performance_history[-10:] if classifier.performance_history else [],
            "category_distribution": category_distribution
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"모델 상태 조회 오류: {e}")
        error_response = {
            "error": "모델 상태 조회 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.get("/ollama-status")
async def get_ollama_status():
    """Ollama 연결 상태 확인"""
    ollama_connected = check_ollama_connection()
    available_models = get_available_ollama_models() if ollama_connected else []
    
    response_data = {
        "connected": ollama_connected,
        "models": available_models,
        "recommended_models": ["llama3", "llama3.1", "gemma2", "mistral"],
        "install_command": "ollama pull llama3" if not available_models else None
    }
    
    return SafeJSONResponse(content=response_data)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """엑셀 파일 업로드 및 세션 생성 - NaN 안전 처리"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="엑셀 파일만 업로드 가능합니다.")
        
        session_id = str(uuid.uuid4())
        
        # 파일 읽기
        content = await file.read()
        df_original = pd.read_excel(BytesIO(content))
        
        # NaN 값 처리
        df_original = df_original.fillna('')
        
        # 날짜 전처리
        df_processed, conversion_info = preprocess_excel_data(df_original)
        
        # 임시 파일 저장
        temp_file_path = temp_dir / f"{session_id}_{file.filename}"
        df_processed.to_excel(temp_file_path, index=False)
        
        # 미리보기 데이터 NaN 처리
        preview_data = clean_for_json(df_processed.head(5).to_dict('records'))
        
        # 세션 데이터에 저장
        sessions[session_id] = {
            'filename': file.filename,
            'original_data': df_processed,
            'upload_time': datetime.now().isoformat(),
            'total_rows': len(df_processed),
            'columns': df_processed.columns.tolist(),
            'preview': preview_data,
            'classifications': {},
            'status': 'uploaded',
            'merchant_column': '가맹점명',
            'total_batches': 0
        }
        
        logger.info(f"파일 업로드 완료: {file.filename}, 세션: {session_id}")
        
        response_data = {
            "session_id": session_id,
            "filename": file.filename,
            "total_rows": len(df_processed),
            "columns": df_processed.columns.tolist(),
            "preview": preview_data,
            "date_conversions": conversion_info,
            "status": "uploaded"
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"파일 업로드 오류: {e}")
        error_response = {
            "error": "파일 처리 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=400)

@app.post("/start-classification")
async def start_classification(request: ClassificationRequest):
    """분류 프로세스 시작 - NaN 안전 처리"""
    try:
        session_id = request.session_id
        
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        session_data = sessions[session_id]
        df = session_data['original_data'].copy()
        df = df.fillna('')  # 추가 NaN 처리
        
        session_data['merchant_column'] = request.merchant_column
        
        # Ollama 연결 확인
        ollama_connected = check_ollama_connection()
        if not ollama_connected:
            logger.warning("Ollama가 연결되지 않음. 키워드 기반 분류만 사용됩니다.")
        
        # 글로벌 통합 모델 사용
        classifier = get_global_classifier()
        classifier.merchant_column = request.merchant_column
        
        # 날짜 컬럼 자동 감지
        date_column_candidates = ['이용일자', '승인일', '거래일', '일자', '날짜']
        detected_date_col = None
        
        for col in df.columns:
            if any(keyword in col for keyword in date_column_candidates):
                detected_date_col = col
                break

        if not detected_date_col:
            raise HTTPException(status_code=400, detail="날짜 컬럼을 자동으로 인식할 수 없습니다.")

        classifier.date_column = detected_date_col
        
        # 월별 배치 생성
        monthly_batches = classifier.create_monthly_batches(df)
        
        if not monthly_batches:
            raise HTTPException(status_code=400, detail="월별 배치를 생성할 수 없습니다.")
        
        session_data['total_batches'] = len(monthly_batches)
        
        model_status = classifier.get_model_status()
        
        classifier_instances[session_id] = {
            "classifier": classifier,
            "batches": monthly_batches,
            "current_batch": 0,
            "total_batches": len(monthly_batches),
            "status": "ready",
            "created_at": datetime.now(),
            "ollama_available": ollama_connected
        }
        
        # 배치 정보 안전 처리
        batch_info = []
        for i, batch in enumerate(monthly_batches):
            try:
                month = batch['년월'].iloc[0] if '년월' in batch.columns else f"배치{i+1}"
                count = len(batch)
                batch_info.append({"month": str(month), "count": int(count)})
            except Exception as e:
                logger.warning(f"배치 {i+1} 정보 처리 실패: {e}")
                batch_info.append({"month": f"배치{i+1}", "count": 0})
        
        response_data = {
            "session_id": session_id,
            "total_batches": len(monthly_batches),
            "total_samples": len(df),
            "status": "ready",
            "ollama_available": ollama_connected,
            "model_status": model_status,
            "batch_info": batch_info,
            "message": f"월별 분류 프로세스가 시작되었습니다. (통합 모델 사용)"
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"분류 시작 오류: {e}")
        error_response = {
            "error": "분류 시작 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.get("/batch/{session_id}/{batch_number}")
async def get_batch(session_id: str, batch_number: int):
    """특정 배치 데이터 가져오기 - NaN 안전 처리"""
    try:
        if session_id not in classifier_instances:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        session_data = classifier_instances[session_id]
        classifier = session_data["classifier"]
        batches = session_data["batches"]
        
        if batch_number < 1 or batch_number > len(batches):
            raise HTTPException(status_code=400, detail="잘못된 배치 번호입니다.")
        
        batch_df = batches[batch_number - 1].copy()
        batch_df = batch_df.fillna('')  # NaN 처리
        
        month = batch_df['년월'].iloc[0] if '년월' in batch_df.columns else f"배치{batch_number}"
        
        # 분류 수행
        if classifier.ensemble_classifier is not None:
            classified_batch = classifier.classify_batch_with_enhanced_model(batch_df, batch_number, str(month))
        else:
            classified_batch = classifier.classify_batch_with_llm(batch_df, batch_number, str(month))
        
        classified_batch = classified_batch.fillna('')  # 추가 NaN 처리
        
        confidence_scores = {}
        classifications = {}
        
        for _, row in classified_batch.iterrows():
            try:
                merchant = str(row[classifier.merchant_column])
                category = str(row['카테고리'])
                classifications[merchant] = category
                confidence_scores[merchant] = 0.8
            except Exception as e:
                logger.warning(f"행 처리 실패: {e}")
                continue
        
        session_data["current_batch"] = batch_number
        session_data["status"] = "processing"
        
        # DataFrame을 안전하게 딕셔너리로 변환
        batch_data = clean_for_json(classified_batch.to_dict('records'))
        
        response_data = {
            "batch_number": batch_number,
            "month": str(month),
            "data": batch_data,
            "classifications": classifications,
            "confidence_scores": confidence_scores
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"배치 조회 오류: {e}")
        error_response = {
            "error": "배치 조회 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.post("/submit-batch")
async def submit_batch(request: BatchProcessRequest):
    """수정된 배치 제출 및 통합 모델 업데이트 - NaN 안전 처리"""
    try:
        session_id = request.session_id
        
        if session_id not in classifier_instances:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        session_data = classifier_instances[session_id]
        classifier = session_data["classifier"]
        batches = session_data["batches"]
        
        batch_df = batches[request.batch_number - 1].copy()
        
        # 수정사항 적용
        for merchant, corrected_category in request.corrections.items():
            batch_df.loc[batch_df[classifier.merchant_column] == merchant, '카테고리'] = corrected_category
        
        # sessions에 분류 결과 저장
        if session_id in sessions:
            if 'classifications' not in sessions[session_id]:
                sessions[session_id]['classifications'] = {}
            sessions[session_id]['classifications'][str(request.batch_number)] = request.corrections
        
        # 통합 모델 업데이트
        classifier.update_training_data_with_corrections(batch_df, request.batch_number)
        performance = classifier.retrain_enhanced_model(request.batch_number)
        
        session_data["status"] = "trained"
        
        response_data = {
            "message": "배치 처리 완료 (통합 모델 업데이트됨)",
            "batch_number": request.batch_number,
            "performance": performance,
            "model_status": classifier.get_model_status(),
            "next_batch_available": request.batch_number < session_data["total_batches"]
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"배치 제출 오류: {e}")
        error_response = {
            "error": "배치 제출 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """세션 정보 조회 - NaN 안전 처리"""
    try:
        if session_id not in classifier_instances:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        session_data = classifier_instances[session_id]
        classifier = session_data["classifier"]
        
        response_data = {
            "session_id": session_id,
            "status": session_data["status"],
            "current_batch": session_data["current_batch"],
            "total_batches": session_data["total_batches"],
            "processed_samples": len(classifier.all_training_data),
            "performance_history": classifier.performance_history,
            "created_at": session_data["created_at"].isoformat(),
            "ollama_available": session_data.get("ollama_available", False),
            "model_status": classifier.get_model_status()
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"세션 조회 오류: {e}")
        error_response = {
            "error": "세션 조회 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.get("/performance/{session_id}")
async def get_performance_history(session_id: str):
    """성능 이력 조회 - NaN 안전 처리"""
    try:
        if session_id not in classifier_instances:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        classifier = classifier_instances[session_id]["classifier"]
        return SafeJSONResponse(content=classifier.performance_history)
        
    except Exception as e:
        logger.error(f"성능 조회 오류: {e}")
        error_response = {
            "error": "성능 조회 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.post("/predict")
async def predict_merchants(request: PredictionRequest):
    """가맹점명 예측 - NaN 안전 처리"""
    try:
        classifier = get_global_classifier()
        
        if classifier.ensemble_classifier is None:
            raise HTTPException(status_code=400, detail="모델이 아직 학습되지 않았습니다.")
        
        predictions = {}
        
        for merchant in request.merchant_names:
            keyword_result = classifier.keyword_based_classification(merchant)
            if keyword_result:
                predictions[merchant] = keyword_result
            else:
                try:
                    processed_name = classifier.preprocess_merchant_name(merchant)
                    X = classifier.extract_enhanced_features([processed_name])
                    X_scaled = classifier.scaler.transform(X)
                    
                    probabilities = classifier.ensemble_classifier.predict_proba(X_scaled)[0]
                    predicted_idx = probabilities.argmax()
                    predicted_category = classifier.label_encoder.inverse_transform([predicted_idx])[0]
                    
                    predictions[merchant] = predicted_category
                except Exception:
                    predictions[merchant] = "기타"
        
        return SafeJSONResponse(content=predictions)
        
    except Exception as e:
        logger.error(f"예측 오류: {e}")
        error_response = {
            "error": "예측 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.get("/download/{session_id}")
async def download_results(session_id: str):
    """현재 세션의 분류 결과 다운로드 - NaN 안전 처리"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    try:
        session_data = sessions[session_id]
        
        # 현재 세션의 원본 데이터 가져오기
        original_data = session_data.get('original_data')
        if original_data is None or original_data.empty:
            raise HTTPException(status_code=404, detail="원본 데이터를 찾을 수 없습니다")
        
        # 현재 세션의 모든 분류 결과를 병합
        classified_data = original_data.copy()
        classified_data = classified_data.fillna('')  # NaN 처리
        
        # 각 배치의 분류 결과를 병합
        all_corrections = {}
        total_batches = session_data.get('total_batches', 0)
        for batch_num in range(1, total_batches + 1):
            batch_corrections = session_data.get('classifications', {}).get(str(batch_num), {})
            all_corrections.update(batch_corrections)
        
        # 가맹점명에 따라 카테고리 매핑
        merchant_column = session_data.get('merchant_column', '가맹점명')
        classified_data['카테고리'] = classified_data[merchant_column].map(all_corrections)
        
        # 분류되지 않은 항목은 '기타'로 설정
        classified_data['카테고리'] = classified_data['카테고리'].fillna('기타')
        
        # NaN 값 최종 처리
        classified_data = classified_data.fillna('')
        
        # 엑셀 파일로 저장
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            classified_data.to_excel(writer, sheet_name='Classification_Result', index=False)
        
        output.seek(0)
        
        # 영문 파일명으로 생성
        original_filename = session_data.get('filename', 'classified_results')
        file_base = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        download_filename = f"{file_base}_classified_{timestamp}.xlsx"
        
        logger.info(f"다운로드 생성 완료: {download_filename}, 분류된 항목: {len(all_corrections)}개")
        
        return Response(
            content=output.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={download_filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"다운로드 오류: {e}")
        raise HTTPException(status_code=500, detail=f"다운로드 실패: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제 - NaN 안전 처리"""
    try:
        if session_id in classifier_instances:
            del classifier_instances[session_id]
        
        if session_id in sessions:
            del sessions[session_id]
        
        temp_files = list(temp_dir.glob(f"{session_id}_*"))
        for temp_file in temp_files:
            temp_file.unlink()
        
        response_data = {
            "message": "세션이 삭제되었습니다. (통합 모델은 보존됨)"
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"세션 삭제 오류: {e}")
        error_response = {
            "error": "세션 삭제 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.get("/categories")
async def get_categories():
    """사용 가능한 카테고리 목록 반환"""
    response_data = {
        "categories": PersistentIncrementalCategorizer.CATEGORIES
    }
    return SafeJSONResponse(content=response_data)

@app.post("/diagnose-model")
async def diagnose_model():
    """통합 모델 진단"""
    try:
        classifier = get_global_classifier()
        
        # 진단 메서드가 있는지 확인
        if hasattr(classifier, 'diagnose_ml_model'):
            classifier.diagnose_ml_model()
        else:
            logger.info("진단 메서드가 없습니다. 기본 상태만 반환합니다.")
        
        response_data = {
            "message": "모델 진단 완료",
            "model_status": classifier.get_model_status(),
            "check_logs": "서버 로그를 확인하세요"
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"모델 진단 오류: {e}")
        error_response = {
            "error": "모델 진단 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.post("/adjust-threshold")
async def adjust_confidence_threshold(new_threshold: float):
    """신뢰도 임계값 수동 조정"""
    try:
        if not (0.1 <= new_threshold <= 0.9):
            raise HTTPException(status_code=400, detail="임계값은 0.1과 0.9 사이여야 합니다.")
        
        classifier = get_global_classifier()
        old_threshold = classifier.confidence_threshold
        classifier.confidence_threshold = new_threshold
        
        response_data = {
            "message": "신뢰도 임계값 조정 완료",
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "model_status": classifier.get_model_status()
        }
        
        return SafeJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"임계값 조정 오류: {e}")
        error_response = {
            "error": "임계값 조정 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

@app.get("/stats")
async def get_global_stats():
    """전체 통계 조회 - NaN 안전 처리"""
    try:
        classifier = get_global_classifier()
        
        # 카테고리 분포 안전 처리
        category_distribution = {}
        if len(classifier.all_training_data) > 0:
            try:
                category_counts = classifier.all_training_data['카테고리'].value_counts()
                category_distribution = category_counts.to_dict()
            except Exception as e:
                logger.warning(f"카테고리 분포 계산 실패: {e}")
        
        stats = {
            "model_status": classifier.get_model_status(),
            "total_training_samples": len(classifier.all_training_data),
            "total_batches": len(classifier.performance_history),
            "categories": PersistentIncrementalCategorizer.CATEGORIES,
            "category_distribution": category_distribution,
            "performance_trend": classifier.performance_history[-10:] if classifier.performance_history else []
        }
        
        return SafeJSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        error_response = {
            "error": "통계 조회 실패",
            "message": str(e)
        }
        return SafeJSONResponse(content=error_response, status_code=500)

def find_available_port(start_port=8080, max_attempts=10):
    """사용 가능한 포트 찾기"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    return start_port

if __name__ == "__main__":
    import uvicorn
    
    try:
        port = find_available_port(8080)
        print(f"🚀 지속적 모델 업데이트 서버가 포트 {port}에서 시작됩니다...")
        print(f"📖 API 문서: http://localhost:{port}/docs")
        print(f"🌐 프론트엔드에서 API_BASE_URL을 http://localhost:{port}로 설정하세요")
        
        # Ollama 상태 확인
        if check_ollama_connection():
            models = get_available_ollama_models()
            print(f"✅ Ollama 연결됨 - 사용 가능한 모델: {models}")
        else:
            print("⚠️ Ollama 연결 안됨 - 키워드 기반 분류만 사용됩니다")
            print("💡 Ollama 설치: https://ollama.ai/")
            print("💡 모델 다운로드: ollama pull llama3")
        
        # 통합 모델 초기화 및 상태 확인
        try:
            classifier = get_global_classifier()
            status = classifier.get_model_status()
            
            print(f"\n📚 통합 모델 상태:")
            print(f"   - 학습됨: {status['model_trained']}")
            print(f"   - 누적 샘플: {status['total_training_samples']}개")
            print(f"   - 처리된 배치: {status['total_batches']}개")
            if status['model_trained']:
                print(f"   - 현재 정확도: {status.get('latest_accuracy', 0):.3f}")
                print(f"   - 신뢰도 임계값: {status['confidence_threshold']:.3f}")
            
            if status['total_training_samples'] == 0:
                print("📝 기존 학습된 데이터 없음. 새 데이터부터 시작합니다.")
            else:
                print("🎯 기존 학습된 모델을 계속 업데이트합니다.")
                
        except Exception as e:
            print(f"⚠️ 통합 모델 확인 중 오류: {e}")
        
        print("\n🎯 새로운 기능:")
        print("   ✅ 기존 학습된 모델 자동 로드")
        print("   ✅ 새 데이터로 기존 모델 업데이트")
        print("   ✅ 중단 후 재시작 가능")
        print("   ✅ 모델 백업 및 버전 관리")
        print("   ✅ 월별 배치 자동 분할")
        print("   ✅ NaN 값 안전 처리")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0", 
            port=port,
            reload=True,
            access_log=True,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ 서버 시작 실패: {e}")
        print("💡 다른 터미널에서 실행 중인 서버가 있는지 확인해주세요.")