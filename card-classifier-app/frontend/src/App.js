import React, { useState, useEffect } from 'react';
import { Upload, FileText, Play, CheckCircle, BarChart3, Download, Trash2, RefreshCw } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8080';

const CardClassifierApp = () => {
  const [sessionId, setSessionId] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [currentStep, setCurrentStep] = useState('upload');
  const [batches, setBatches] = useState([]);
  const [currentBatch, setCurrentBatch] = useState(null);
  const [batchData, setBatchData] = useState([]);
  const [corrections, setCorrections] = useState({});
  const [sessionInfo, setSessionInfo] = useState(null);
  const [performanceHistory, setPerformanceHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  
  const categories = [
    "식비", "카페/간식", "편의점/마트/잡화", "술/유흥", "쇼핑", 
    "영화/OTT", "취미/여가", "의료/건강", "주거/통신/공과금", 
    "보험/세금/기타금융", "미용/뷰티", "교통/대중교통", 
    "자동차/주유소", "여행/숙박", "항공", "교육", "생활", "기타","간편결제"
  ];

  // 스타일 객체들
  const styles = {
    container: {
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%)',
      padding: '16px'
    },
    maxWidth: {
      maxWidth: '72rem',
      margin: '0 auto'
    },
    header: {
      textAlign: 'center',
      marginBottom: '32px'
    },
    title: {
      fontSize: '2.25rem',
      fontWeight: 'bold',
      color: '#1f2937',
      marginBottom: '8px'
    },
    subtitle: {
      color: '#6b7280',
      marginTop: '16px'
    },
    connectionStatus: {
      display: 'inline-flex',
      alignItems: 'center',
      padding: '4px 12px',
      borderRadius: '9999px',
      fontSize: '14px',
      fontWeight: '500'
    },
    connectionChecking: {
      backgroundColor: '#fef3c7',
      color: '#92400e'
    },
    connectionConnected: {
      backgroundColor: '#d1fae5',
      color: '#065f46'
    },
    connectionFailed: {
      backgroundColor: '#fee2e2',
      color: '#991b1b'
    },
    errorBox: {
      backgroundColor: '#fee2e2',
      border: '1px solid #fca5a5',
      color: '#991b1b',
      padding: '16px',
      borderRadius: '8px',
      marginBottom: '16px'
    },
    warningBox: {
      backgroundColor: '#fffbeb',
      border: '1px solid #fed7aa',
      borderRadius: '8px',
      padding: '24px',
      marginBottom: '24px'
    },
    card: {
      backgroundColor: 'white',
      borderRadius: '8px',
      boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      padding: '32px'
    },
    button: {
      display: 'inline-flex',
      alignItems: 'center',
      padding: '12px 24px',
      borderRadius: '8px',
      fontSize: '16px',
      fontWeight: '500',
      transition: 'all 0.2s',
      cursor: 'pointer',
      border: 'none'
    },
    buttonPrimary: {
      backgroundColor: '#3b82f6',
      color: 'white'
    },
    buttonSuccess: {
      backgroundColor: '#10b981',
      color: 'white'
    },
    buttonSecondary: {
      backgroundColor: 'transparent',
      color: '#374151',
      border: '1px solid #d1d5db'
    },
    buttonDisabled: {
      opacity: 0.5,
      cursor: 'not-allowed'
    },
    progressSteps: {
      display: 'flex',
      justifyContent: 'center',
      marginBottom: '32px'
    },
    stepContainer: {
      display: 'flex',
      alignItems: 'center',
      gap: '16px'
    },
    step: {
      width: '32px',
      height: '32px',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '14px',
      fontWeight: '600'
    },
    stepActive: {
      backgroundColor: '#3b82f6',
      color: 'white'
    },
    stepCompleted: {
      backgroundColor: '#10b981',
      color: 'white'
    },
    stepInactive: {
      backgroundColor: '#d1d5db',
      color: '#6b7280'
    },
    stepConnector: {
      width: '48px',
      height: '2px',
      backgroundColor: '#d1d5db'
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse',
      border: '1px solid #e5e7eb'
    },
    tableHeader: {
      backgroundColor: '#f9fafb',
      padding: '12px 16px',
      textAlign: 'left',
      fontSize: '14px',
      fontWeight: '500',
      color: '#374151',
      borderBottom: '1px solid #e5e7eb'
    },
    tableCell: {
      padding: '12px 16px',
      fontSize: '14px',
      color: '#6b7280',
      borderBottom: '1px solid #e5e7eb'
    },
    select: {
      width: '100%',
      border: '1px solid #d1d5db',
      borderRadius: '6px',
      padding: '8px 12px',
      fontSize: '14px'
    },
    progressBar: {
      width: '100%',
      backgroundColor: '#e5e7eb',
      borderRadius: '9999px',
      height: '8px',
      marginBottom: '16px'
    },
    progressFill: {
      height: '8px',
      borderRadius: '9999px',
      background: 'linear-gradient(90deg, #3b82f6 0%, #10b981 100%)',
      transition: 'width 0.3s ease'
    },
    badge: {
      padding: '4px 8px',
      borderRadius: '4px',
      fontSize: '12px',
      fontWeight: '500'
    },
    badgeDefault: {
      backgroundColor: '#f3f4f6',
      color: '#374151'
    },
    badgePrimary: {
      backgroundColor: '#dbeafe',
      color: '#1d4ed8'
    },
    grid: {
      display: 'grid',
      gap: '24px'
    },
    gridCols2: {
      gridTemplateColumns: 'repeat(2, 1fr)'
    },
    gridCols4: {
      gridTemplateColumns: 'repeat(4, 1fr)'
    },
    textCenter: {
      textAlign: 'center'
    },
    mb4: { marginBottom: '16px' },
    mb6: { marginBottom: '24px' },
    mt4: { marginTop: '16px' },
    mt6: { marginTop: '24px' },
    flexBetween: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center'
    },
    flexCenter: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center'
    },
    spaceY4: {
      '& > * + *': {
        marginTop: '16px'
      }
    }
  };

  // 백엔드 연결 상태 확인
  const checkBackendConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      
      if (response.ok) {
        setConnectionStatus('connected');
        setError(null);
      } else {
        setConnectionStatus('failed');
        setError('백엔드 서버에 연결할 수 없습니다.');
      }
    } catch (err) {
      setConnectionStatus('failed');
      setError(`백엔드 연결 실패: ${API_BASE_URL}에서 서버를 찾을 수 없습니다.`);
    }
  };

  useEffect(() => {
    checkBackendConnection();
    const interval = setInterval(() => {
      if (connectionStatus === 'failed') {
        checkBackendConnection();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [connectionStatus]);

  // 파일 업로드
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (connectionStatus !== 'connected') {
      setError('백엔드 서버가 연결되지 않았습니다.');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('파일 크기는 50MB를 초과할 수 없습니다.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || '파일 업로드에 실패했습니다.');
      }

      setSessionId(data.session_id);
      setUploadedFile(data);
      setCurrentStep('configure');
      setError(null);
    } catch (err) {
      setError(`업로드 실패: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 분류 시작
  const startClassification = async (batchSize = 20, merchantColumn = '가맹점명') => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/start-classification`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          batch_size: batchSize,
          merchant_column: merchantColumn,
        }),
      });

      if (!response.ok) {
        throw new Error('분류 시작에 실패했습니다.');
      }

      const data = await response.json();
      setBatches(Array.from({ length: data.total_batches }, (_, i) => i + 1));
      setCurrentStep('processing');
      loadBatch(1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // 배치 데이터 로드
  const loadBatch = async (batchNumber) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/batch/${sessionId}/${batchNumber}`);
      
      if (!response.ok) {
        throw new Error(`배치 ${batchNumber} 로드에 실패했습니다.`);
      }

      const data = await response.json();
      setCurrentBatch(batchNumber);
      setBatchData(data.data);
      setCorrections(data.classifications);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // 배치 제출
  const submitBatch = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/submit-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          batch_number: currentBatch,
          corrections: corrections,
        }),
      });

      if (!response.ok) {
        throw new Error('배치 제출에 실패했습니다.');
      }

      const data = await response.json();
      loadPerformanceHistory();
      
      if (data.next_batch_available) {
        loadBatch(currentBatch + 1);
      } else {
        setCurrentStep('results');
        loadSessionInfo();
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // 기타 함수들 (동일한 로직)
  const loadSessionInfo = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setSessionInfo(data);
      }
    } catch (err) {
      console.error('세션 정보 로드 실패:', err);
    }
  };

  const loadPerformanceHistory = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE_URL}/performance/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setPerformanceHistory(data);
      }
    } catch (err) {
      console.error('성능 히스토리 로드 실패:', err);
    }
  };
// 결과 다운로드 함수 - 현재 세션 파일만 다운로드
const downloadResults = async () => {
  try {
    setLoading(true);
    
    // 현재 세션의 분류된 결과만 다운로드 (기존 엔드포인트 사용)
    const response = await fetch(`${API_BASE_URL}/download/${sessionId}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
      }
    });
    
    if (!response.ok) {
      throw new Error('다운로드에 실패했습니다.');
    }
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    
    // 업로드한 원본 파일명을 기반으로 다운로드 파일명 생성
    const originalFileName = uploadedFile?.filename || 'classified_results';
    const fileNameWithoutExt = originalFileName.replace(/\.[^/.]+$/, '');
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    
    a.download = `${fileNameWithoutExt}_분류완료_${timestamp}.xlsx`;
    
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    setError(null);
  } catch (err) {
    setError(`다운로드 실패: ${err.message}`);
  } finally {
    setLoading(false);
  }
};
  const updateCorrection = (merchant, category) => {
    setCorrections(prev => ({ ...prev, [merchant]: category }));
  };

  const resetSession = () => {
    setSessionId(null);
    setUploadedFile(null);
    setCurrentStep('upload');
    setBatches([]);
    setCurrentBatch(null);
    setBatchData([]);
    setCorrections({});
    setSessionInfo(null);
    setPerformanceHistory([]);
    setError(null);
  };

  useEffect(() => {
    if (sessionId && currentStep === 'results') {
      loadSessionInfo();
      loadPerformanceHistory();
    }
  }, [sessionId, currentStep]);

  const getStepStyle = (stepName, index) => {
    const steps = ['upload', 'configure', 'processing', 'results'];
    const currentIndex = steps.indexOf(currentStep);
    
    if (currentStep === stepName) {
      return { ...styles.step, ...styles.stepActive };
    } else if (currentIndex > index) {
      return { ...styles.step, ...styles.stepCompleted };
    } else {
      return { ...styles.step, ...styles.stepInactive };
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.maxWidth}>
        {/* 헤더 */}
        <div style={styles.header}>
          <h1 style={styles.title}>🎯 카드 거래 분류기</h1>
          <p style={styles.subtitle}>AI 기반 점진적 학습 시스템</p>
          
          {/* 연결 상태 표시 */}
          <div style={styles.mt4}>
            {connectionStatus === 'checking' && (
              <div style={{ ...styles.connectionStatus, ...styles.connectionChecking }}>
                <RefreshCw style={{ marginRight: '8px', animation: 'spin 1s linear infinite' }} size={16} />
                백엔드 연결 확인 중...
              </div>
            )}
            {connectionStatus === 'connected' && (
              <div style={{ ...styles.connectionStatus, ...styles.connectionConnected }}>
                <CheckCircle style={{ marginRight: '8px' }} size={16} />
                백엔드 연결됨 ({API_BASE_URL})
              </div>
            )}
            {connectionStatus === 'failed' && (
              <div style={{ ...styles.connectionStatus, ...styles.connectionFailed }}>
                <span style={{ width: '8px', height: '8px', backgroundColor: '#dc2626', borderRadius: '50%', marginRight: '8px' }}></span>
                백엔드 연결 실패 - 자동 재시도 중...
              </div>
            )}
          </div>
        </div>

        {/* 에러 메시지 */}
        {error && (
          <div style={styles.errorBox}>
            <div style={styles.flexBetween}>
              <div>
                <strong>오류:</strong> {error}
                {connectionStatus === 'failed' && (
                  <div style={styles.mt4}>
                    <p>해결 방법:</p>
                    <ul style={{ marginLeft: '20px', marginTop: '8px' }}>
                      <li>백엔드 서버가 실행 중인지 확인 (python main.py)</li>
                      <li>포트 8080이 사용 중이지 않은지 확인</li>
                      <li>방화벽이나 보안 소프트웨어 설정 확인</li>
                    </ul>
                  </div>
                )}
              </div>
              <button onClick={() => setError(null)} style={{ background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer' }}>
                ✕
              </button>
            </div>
          </div>
        )}

        {/* 백엔드 연결 안됨 경고 */}
        {connectionStatus === 'failed' && (
          <div style={styles.warningBox}>
            <div style={{ display: 'flex' }}>
              <div style={{ flexShrink: 0 }}>
                <span style={{ fontSize: '24px' }}>⚠️</span>
              </div>
              <div style={{ marginLeft: '12px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: '500', color: '#92400e' }}>
                  백엔드 서버 연결 필요
                </h3>
                <div style={{ marginTop: '8px', color: '#b45309' }}>
                  <p>카드 분류기를 사용하려면 백엔드 서버가 실행되어야 합니다.</p>
                  <div style={{ marginTop: '12px', backgroundColor: '#fef3c7', borderRadius: '6px', padding: '12px' }}>
                    <p style={{ fontFamily: 'monospace', fontSize: '14px' }}>
                      터미널에서 실행: <br/>
                      <code style={{ backgroundColor: '#1f2937', color: '#10b981', padding: '4px 8px', borderRadius: '4px' }}>
                        cd backend && python main.py
                      </code>
                    </p>
                  </div>
                  <button
                    onClick={checkBackendConnection}
                    style={{ ...styles.button, backgroundColor: '#f59e0b', color: 'white', marginTop: '12px' }}
                  >
                    연결 다시 시도
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 진행 단계 표시 */}
        <div style={styles.progressSteps}>
          <div style={styles.stepContainer}>
            {['upload', 'configure', 'processing', 'results'].map((step, index) => (
              <React.Fragment key={step}>
                <div style={getStepStyle(step, index)}>
                  {['upload', 'configure', 'processing', 'results'].indexOf(currentStep) > index ? 
                    <CheckCircle size={16} /> : index + 1}
                </div>
                {index < 3 && <div style={styles.stepConnector}></div>}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* 파일 업로드 단계 */}
        {currentStep === 'upload' && (
          <div style={{ ...styles.card, ...styles.textCenter }}>
            <Upload style={{ margin: '0 auto 16px', color: '#3b82f6' }} size={48} />
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px' }}>엑셀 파일 업로드</h2>
            <p style={{ color: '#6b7280', marginBottom: '24px' }}>
              카드 거래 내역이 포함된 엑셀 파일을 업로드해주세요.
            </p>
            
            <label style={{ ...styles.button, ...styles.buttonPrimary }}>
              <FileText style={{ marginRight: '8px' }} size={20} />
              파일 선택
              <input
                type="file"
                accept=".xlsx,.xls"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
                disabled={loading}
              />
            </label>
            
            {loading && (
              <div style={{ ...styles.flexCenter, marginTop: '16px' }}>
                <RefreshCw style={{ animation: 'spin 1s linear infinite', color: '#3b82f6' }} size={24} />
              </div>
            )}
          </div>
        )}

        {/* 설정 단계 */}
        {currentStep === 'configure' && uploadedFile && (
          <div style={styles.card}>
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '24px' }}>분류 설정</h2>
            
            <div style={{ ...styles.grid, ...styles.gridCols2, marginBottom: '24px' }}>
              <div>
                <h3 style={{ fontWeight: '600', marginBottom: '8px' }}>파일 정보</h3>
                <p style={{ fontSize: '14px', color: '#6b7280' }}>파일명: {uploadedFile.filename}</p>
                <p style={{ fontSize: '14px', color: '#6b7280' }}>총 행 수: {uploadedFile.total_rows.toLocaleString()}개</p>
              </div>
              
              <div>
                <h3 style={{ fontWeight: '600', marginBottom: '8px' }}>설정</h3>
                <div style={{ ...styles.grid, gap: '12px' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '4px' }}>배치 크기</label>
                    <select style={styles.select} defaultValue="20">
                      <option value="10">10개</option>
                      <option value="20">20개</option>
                      <option value="30">30개</option>
                      <option value="50">50개</option>
                    </select>
                  </div>
                  
                  <div>
                    <label style={{ display: 'block', fontSize: '14px', fontWeight: '500', marginBottom: '4px' }}>가맹점 컬럼</label>
                    <select style={styles.select} defaultValue="가맹점명">
                      {uploadedFile.columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            </div>

            {/* 미리보기 */}
            <div style={styles.mb6}>
              <h3 style={{ fontWeight: '600', marginBottom: '8px' }}>데이터 미리보기</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      {uploadedFile.columns.slice(0, 5).map(col => (
                        <th key={col} style={styles.tableHeader}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {uploadedFile.preview.map((row, index) => (
                      <tr key={index}>
                        {uploadedFile.columns.slice(0, 5).map(col => (
                          <td key={col} style={styles.tableCell}>
                            {String(row[col] || '').slice(0, 50)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div style={styles.flexBetween}>
              <button onClick={resetSession} style={{ ...styles.button, ...styles.buttonSecondary }}>
                다시 업로드
              </button>
              
              <button
                onClick={() => startClassification()}
                disabled={loading}
                style={{ 
                  ...styles.button, 
                  ...styles.buttonSuccess,
                  ...(loading ? styles.buttonDisabled : {})
                }}
              >
                <Play style={{ marginRight: '8px' }} size={20} />
                {loading ? '시작 중...' : '분류 시작'}
              </button>
            </div>
          </div>
        )}

        {/* 처리 단계 */}
        {currentStep === 'processing' && currentBatch && (
          <div style={styles.grid}>
            {/* 진행 상황 */}
            <div style={styles.card}>
              <div style={{ ...styles.flexBetween, marginBottom: '16px' }}>
                <h2 style={{ fontSize: '24px', fontWeight: 'bold' }}>배치 {currentBatch} 처리 중</h2>
                <div style={{ fontSize: '14px', color: '#6b7280' }}>
                  {currentBatch} / {batches.length} 배치
                </div>
              </div>
              
              <div style={styles.progressBar}>
                <div 
                  style={{ 
                    ...styles.progressFill,
                    width: `${(currentBatch / batches.length) * 100}%`
                  }}
                ></div>
              </div>
            </div>

       
            {/* 배치 데이터 */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-semibold mb-4">분류 결과 검토 및 수정</h3>
              
              {loading ? (
                <div className="flex justify-center py-8">
                  <RefreshCw className="animate-spin text-blue-500" size={32} />
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-3 px-4">가맹점명</th>
                        <th className="text-left py-3 px-4">AI 분류</th>
                        <th className="text-left py-3 px-4">수정</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchData.map((row, index) => {
                        const merchant = row['가맹점명'] || row[Object.keys(row)[0]];
                        const currentCategory = corrections[merchant] || '기타';
                        
                        return (
                          <tr key={index} className="border-b hover:bg-gray-50">
                            <td className="py-3 px-4 font-medium">
                              {merchant}
                            </td>
                            <td className="py-3 px-4">
                              <span className={`px-2 py-1 rounded text-sm ${
                                currentCategory === '기타' ? 'bg-gray-100 text-gray-700' :
                                'bg-blue-100 text-blue-700'
                              }`}>
                                {currentCategory}
                              </span>
                            </td>
                            <td className="py-3 px-4">
                              <select
                                value={currentCategory}
                                onChange={(e) => updateCorrection(merchant, e.target.value)}
                                className="w-full border rounded px-2 py-1 text-sm"
                              >
                                {categories.map(cat => (
                                  <option key={cat} value={cat}>{cat}</option>
                                ))}
                              </select>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
              
              <div className="flex justify-between mt-6">
                <div className="flex space-x-2">
                  {currentBatch > 1 && (
                    <button
                      onClick={() => loadBatch(currentBatch - 1)}
                      className="px-4 py-2 border border-gray-300 rounded hover:bg-gray-50"
                      disabled={loading}
                    >
                      이전 배치
                    </button>
                  )}
                </div>
                
                <button
                  onClick={submitBatch}
                  disabled={loading}
                  className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50"
                >
                  {loading ? '처리 중...' : '확인 및 다음'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 결과 단계 */}
        {currentStep === 'results' && (
          <div className="space-y-6">
            {/* 완료 메시지 */}
            <div className="bg-white rounded-lg shadow-lg p-8 text-center">
              <CheckCircle className="mx-auto mb-4 text-green-500" size={64} />
              <h2 className="text-3xl font-bold text-gray-800 mb-2">분류 완료!</h2>
              <p className="text-gray-600 mb-6">
                모든 데이터의 분류가 완료되었습니다.
              </p>
              
              <div className="flex justify-center space-x-4">
                <button
                  onClick={downloadResults}
                  className="px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center"
                >
                  <Download className="mr-2" size={20} />
                  결과 다운로드
                </button>
                
                <button
                  onClick={resetSession}
                  className="px-6 py-3 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors flex items-center"
                >
                  <Trash2 className="mr-2" size={20} />
                  새로 시작
                </button>
              </div>
            </div>

            {/* 세션 정보 */}
            {sessionInfo && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-semibold mb-4">처리 결과 요약</h3>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {sessionInfo.total_batches}
                    </div>
                    <div className="text-sm text-gray-600">처리된 배치</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {sessionInfo.processed_samples}
                    </div>
                    <div className="text-sm text-gray-600">분류된 거래</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {performanceHistory.length > 0 ? 
                        `${(performanceHistory[performanceHistory.length - 1]?.train_accuracy * 100).toFixed(1)}%` : 
                        'N/A'
                      }
                    </div>
                    <div className="text-sm text-gray-600">최종 정확도</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-orange-600">
                      {performanceHistory.length > 0 ? 
                        `${(performanceHistory[performanceHistory.length - 1]?.cv_mean * 100).toFixed(1)}%` : 
                        'N/A'
                      }
                    </div>
                    <div className="text-sm text-gray-600">교차검증 점수</div>
                  </div>
                </div>
              </div>
            )}

            {/* 성능 차트 */}
            {performanceHistory.length > 0 && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <BarChart3 className="mr-2" size={24} />
                  학습 성능 추이
                </h3>
                
                <div className="space-y-4">
                  {performanceHistory.map((perf, index) => (
                    <div key={index} className="border-l-4 border-blue-500 pl-4">
                      <div className="flex justify-between items-center">
                        <div>
                          <h4 className="font-medium">배치 {perf.batch}</h4>
                          <p className="text-sm text-gray-600">
                            학습 샘플: {perf.training_samples}개
                          </p>
                        </div>
                        
                        <div className="text-right">
                          <div className="text-lg font-bold text-blue-600">
                            {(perf.train_accuracy * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-600">
                            CV: {(perf.cv_mean * 100).toFixed(1)}% (±{(perf.cv_std * 100).toFixed(1)}%)
                          </div>
                        </div>
                      </div>
                      
                      {/* 성능 바 */}
                      <div className="mt-2">
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-gradient-to-r from-blue-500 to-green-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${perf.train_accuracy * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CardClassifierApp;