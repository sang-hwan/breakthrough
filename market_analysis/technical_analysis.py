# market_analysis/technical_analysis.py
from logs.log_config import setup_logger
import pandas as pd
import numpy as np

logger = setup_logger(__name__)

def compute_sma(data: pd.Series, window: int = 20) -> pd.Series:
    """
    주어진 시계열 데이터에 대해 단순 이동평균(SMA)을 계산합니다.
    
    Parameters:
        data (pd.Series): 가격 데이터.
        window (int): 이동평균 기간 (기본: 20).
        
    Returns:
        pd.Series: SMA 값 시리즈.
    """
    try:
        sma = data.rolling(window=window).mean()
        logger.debug(f"Computed SMA with window={window}.")
        return sma
    except Exception as e:
        logger.error(f"Error computing SMA: {e}", exc_info=True)
        raise

def compute_bollinger_bands(data: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    주어진 시계열 데이터에 대해 Bollinger Bands를 계산합니다.
    
    Parameters:
        data (pd.Series): 가격 데이터.
        window (int): 이동평균 기간.
        num_std (int): 표준편차 배수 (기본: 2).
        
    Returns:
        pd.DataFrame: 'SMA', 'Upper Band', 'Lower Band' 컬럼 포함 데이터프레임.
    """
    try:
        sma = compute_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        logger.debug("Computed Bollinger Bands.")
        return pd.DataFrame({
            "SMA": sma,
            "Upper Band": upper_band,
            "Lower Band": lower_band
        })
    except Exception as e:
        logger.error(f"Error computing Bollinger Bands: {e}", exc_info=True)
        raise

def compute_rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """
    주어진 시계열 데이터에 대해 RSI(Relative Strength Index)를 계산합니다.
    
    Parameters:
        data (pd.Series): 가격 데이터.
        window (int): RSI 계산 기간 (기본: 14).
        
    Returns:
        pd.Series: RSI 값 시리즈.
    """
    try:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        logger.debug("Computed RSI.")
        return rsi
    except Exception as e:
        logger.error(f"Error computing RSI: {e}", exc_info=True)
        raise
