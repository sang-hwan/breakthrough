# backtesting/steps/hmm_manager.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def update_hmm(backtester, dynamic_params):
    regime_series = backtester.update_hmm_regime(dynamic_params)
    logger.debug("HMM regime updated.")
    return regime_series
