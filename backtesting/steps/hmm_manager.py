# backtesting/steps/hmm_manager.py

from logs.logger_config import setup_logger

# 모듈 로깅 인스턴스 설정
logger = setup_logger(__name__)

def update_hmm(backtester, dynamic_params):
    """
    백테스터 객체의 HMM(은닉 마르코프 모델) 상태를 업데이트하고, 업데이트된 regime(시장 체제)의 분포를 로그로 출력합니다.
    
    Parameters:
        backtester (object): HMM 업데이트 메서드를 가진 백테스터 객체.
        dynamic_params (dict): 동적 파라미터(예: 시장 환경, 유동성 정보 등)를 포함하는 딕셔너리.
    
    Returns:
        pandas.Series: 업데이트된 HMM regime 시리즈.
    """
    # 백테스터 내부의 HMM 업데이트 함수 호출 (예: 시장 체제 분류 업데이트)
    regime_series = backtester.update_hmm_regime(dynamic_params)
    try:
        # 각 regime 값의 빈도수를 계산하여 딕셔너리 형태로 변환 후 디버그 로그 출력
        counts = regime_series.value_counts().to_dict()
        logger.debug(f"HMM 업데이트 완료: 총 {len(regime_series)} 샘플, regime 분포: {counts}")
    except Exception:
        logger.error("HMM 업데이트 완료: regime 분포 정보 산출 실패")
    return regime_series
