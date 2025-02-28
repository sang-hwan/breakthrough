# backtesting/steps/hmm_manager.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def update_hmm(backtester, dynamic_params):
    regime_series = backtester.update_hmm_regime(dynamic_params)
    try:
        counts = regime_series.value_counts().to_dict()
        logger.debug(f"HMM 업데이트 완료: 총 {len(regime_series)} 샘플, regime 분포: {counts}")
    except Exception:
        logger.error("HMM 업데이트 완료: regime 분포 정보 산출 실패")
    return regime_series
