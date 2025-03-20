# market_analysis/onchain_analysis.py
from logs.log_config import setup_logger
import pandas as pd
import numpy as np

logger = setup_logger(__name__)

def calculate_mvrv_ratio(market_cap: float, realized_cap: float) -> float:
    """
    온체인 분석을 위한 MVRV (Market Value to Realized Value) 비율을 계산합니다.
    
    Parameters:
        market_cap (float): 현재 시가총액.
        realized_cap (float): 실현 시가총액.
        
    Returns:
        float: MVRV 비율.
    """
    try:
        if realized_cap == 0:
            logger.error("Realized capitalization is zero, cannot compute MVRV ratio.")
            return float('inf')
        ratio = market_cap / realized_cap
        logger.debug(f"Calculated MVRV ratio: {ratio}")
        return ratio
    except Exception as e:
        logger.error(f"Error calculating MVRV ratio: {e}", exc_info=True)
        raise

def analyze_exchange_flow(exchange_inflow: float, exchange_outflow: float) -> str:
    """
    거래소의 유입 및 유출 데이터를 기반으로 매수/매도 신호를 판단합니다.
    
    Parameters:
        exchange_inflow (float): 거래소 유입량.
        exchange_outflow (float): 거래소 유출량.
        
    Returns:
        str: 'distribution' (매도 압력) 또는 'accumulation' (매수, 축적) 신호.
    """
    try:
        if exchange_inflow > exchange_outflow:
            signal = "distribution"
        else:
            signal = "accumulation"
        logger.debug(f"Exchange flow analysis: {signal} (inflow: {exchange_inflow}, outflow: {exchange_outflow})")
        return signal
    except Exception as e:
        logger.error(f"Error analyzing exchange flow: {e}", exc_info=True)
        raise
