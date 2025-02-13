# backtesting/steps/hmm_manager.py
from logs.logger_config import setup_logger
from markets.regime_model import MarketRegimeHMM

logger = setup_logger(__name__)

def update_hmm(backtester, dynamic_params):
    try:
        regime_series = backtester.update_hmm_regime(dynamic_params)
        logger.debug("HMM 레짐 업데이트 완료")
        return regime_series
    except Exception as e:
        logger.error(f"HMM 레짐 업데이트 실패: {e}", exc_info=True)
        raise
