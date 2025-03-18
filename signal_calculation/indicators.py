# trading/indicators.py

import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from logging.logger_config import setup_logger

# 전역 변수 및 객체 정의
# logger: 이 모듈에서 발생하는 디버그 및 에러 메시지를 기록하는 로깅 객체입니다.
logger = setup_logger(__name__)


def compute_sma(data: pd.DataFrame, price_column: str = 'close', period: int = 20, fillna: bool = False, output_col: str = 'sma') -> pd.DataFrame:
    """
    주어진 데이터프레임의 지정된 가격 열을 기준으로 단순 이동평균(SMA)을 계산합니다.
    
    Parameters:
      - data (pd.DataFrame): 가격 정보가 포함된 입력 데이터프레임.
      - price_column (str): SMA 계산에 사용할 가격 열 이름 (예: 'close').
      - period (int): SMA 계산에 사용될 기간 (예: 20일).
      - fillna (bool): 결측치가 있을 경우 보정 여부.
      - output_col (str): 계산된 SMA 결과를 저장할 컬럼 이름.
    
    Returns:
      - pd.DataFrame: 입력 데이터프레임에 SMA 컬럼이 추가된 결과 데이터프레임.
    
    동작 방식:
      - ta 라이브러리의 SMAIndicator 클래스를 사용하여 주어진 기간의 단순 이동평균을 계산합니다.
      - 계산된 결과를 데이터프레임의 새로운 컬럼(output_col)에 추가합니다.
    """
    try:
        # SMAIndicator 객체를 생성하여 지정된 가격 열과 기간에 따른 SMA를 계산하도록 설정합니다.
        sma = SMAIndicator(close=data[price_column], window=period, fillna=fillna)
        # 계산된 SMA 값을 output_col 이름의 컬럼에 저장합니다.
        data[output_col] = sma.sma_indicator()
        logger.debug(f"SMA computed with period {period}")
    except Exception as e:
        # 예외 발생 시 에러 로그에 기록합니다.
        logger.error(f"compute_sma error: {e}", exc_info=True)
    return data


def compute_macd(data: pd.DataFrame, price_column: str = 'close', slow_period: int = 26, fast_period: int = 12, signal_period: int = 9, fillna: bool = False, prefix: str = 'macd_') -> pd.DataFrame:
    """
    주어진 데이터프레임의 가격 정보를 바탕으로 MACD (이동평균 수렴·발산 지표)를 계산합니다.
    
    Parameters:
      - data (pd.DataFrame): 가격 정보가 포함된 입력 데이터프레임.
      - price_column (str): MACD 계산에 사용할 가격 열 이름.
      - slow_period (int): 느린 이동평균에 사용되는 기간.
      - fast_period (int): 빠른 이동평균에 사용되는 기간.
      - signal_period (int): 시그널 라인 계산에 사용되는 기간.
      - fillna (bool): 결측치 보정 여부.
      - prefix (str): 결과 컬럼 이름에 붙일 접두사.
    
    Returns:
      - pd.DataFrame: MACD, 시그널 라인, 그리고 두 값의 차이(diff)를 포함하는 컬럼들이 추가된 데이터프레임.
    
    동작 방식:
      - ta 라이브러리의 MACD 클래스를 활용해 MACD 관련 값을 계산하고,
        각 결과를 접두사(prefix)를 붙여 데이터프레임에 저장합니다.
    """
    try:
        # MACD 객체 생성: 지정된 가격 열과 기간 값들을 이용하여 MACD 계산을 설정합니다.
        macd = MACD(close=data[price_column],
                    window_slow=slow_period,
                    window_fast=fast_period,
                    window_sign=signal_period,
                    fillna=fillna)
        # MACD 관련 값을 각각 새로운 컬럼에 저장합니다.
        data[f'{prefix}macd'] = macd.macd()
        data[f'{prefix}signal'] = macd.macd_signal()
        data[f'{prefix}diff'] = macd.macd_diff()
        logger.debug(f"MACD computed (slow={slow_period}, fast={fast_period}, signal={signal_period})")
    except Exception as e:
        logger.error(f"compute_macd error: {e}", exc_info=True)
    return data


def compute_rsi(data: pd.DataFrame, price_column: str = 'close', period: int = 14, fillna: bool = False, output_col: str = 'rsi') -> pd.DataFrame:
    """
    주어진 데이터프레임의 가격 정보를 바탕으로 상대 강도 지수(RSI)를 계산합니다.
    
    Parameters:
      - data (pd.DataFrame): 가격 정보가 포함된 입력 데이터프레임.
      - price_column (str): RSI 계산에 사용할 가격 열 이름.
      - period (int): RSI 계산에 사용될 기간.
      - fillna (bool): 결측치 보정 여부.
      - output_col (str): 계산된 RSI 결과를 저장할 컬럼 이름.
    
    Returns:
      - pd.DataFrame: RSI 컬럼이 추가된 데이터프레임.
    
    동작 방식:
      - ta 라이브러리의 RSIIndicator 클래스를 사용하여 주어진 기간 동안의 RSI를 계산하고,
        결과를 데이터프레임에 추가합니다.
    """
    try:
        # RSIIndicator 객체 생성: 지정된 가격 열과 기간에 따른 RSI를 계산하도록 설정합니다.
        rsi = RSIIndicator(close=data[price_column], window=period, fillna=fillna)
        # 계산된 RSI 값을 output_col 이름의 컬럼에 저장합니다.
        data[output_col] = rsi.rsi()
        logger.debug(f"RSI computed with period {period}")
    except Exception as e:
        logger.error(f"compute_rsi error: {e}", exc_info=True)
    return data


def compute_bollinger_bands(data: pd.DataFrame, price_column: str = 'close', period: int = 20, std_multiplier: float = 2.0, fillna: bool = False, prefix: str = 'bb_') -> pd.DataFrame:
    """
    주어진 가격 데이터를 바탕으로 Bollinger Bands(볼린저 밴드) 및 관련 지표들을 계산합니다.
    
    Parameters:
      - data (pd.DataFrame): 가격 정보가 포함된 입력 데이터프레임.
      - price_column (str): Bollinger Bands 계산에 사용할 가격 열 이름.
      - period (int): 중간 이동평균 계산에 사용되는 기간.
      - std_multiplier (float): 표준편차 배수 (상한/하한 밴드 계산에 사용).
      - fillna (bool): 결측치 보정 여부.
      - prefix (str): 결과 컬럼 이름에 붙일 접두사.
    
    Returns:
      - pd.DataFrame: 중간 이동평균, 상한/하한 밴드, 퍼센트 밴드, 폭 밴드 및 밴드 지표들을 포함하는 컬럼들이 추가된 데이터프레임.
    
    동작 방식:
      - ta 라이브러리의 BollingerBands 클래스를 사용하여 각 밴드와 지표들을 계산한 후,
        계산 결과를 각각의 컬럼에 저장합니다.
    """
    try:
        # BollingerBands 객체 생성: 지정된 가격 열, 기간, 표준편차 배수를 이용하여 볼린저 밴드를 계산하도록 설정합니다.
        bb = BollingerBands(close=data[price_column], window=period, window_dev=std_multiplier, fillna=fillna)
        # 계산된 각 지표들을 접두사(prefix)를 붙여 데이터프레임에 저장합니다.
        data[f'{prefix}mavg'] = bb.bollinger_mavg()
        data[f'{prefix}hband'] = bb.bollinger_hband()
        data[f'{prefix}lband'] = bb.bollinger_lband()
        data[f'{prefix}pband'] = bb.bollinger_pband()
        data[f'{prefix}wband'] = bb.bollinger_wband()
        data[f'{prefix}hband_ind'] = bb.bollinger_hband_indicator()
        data[f'{prefix}lband_ind'] = bb.bollinger_lband_indicator()
        logger.debug(f"Bollinger Bands computed (period={period}, std_multiplier={std_multiplier})")
    except Exception as e:
        logger.error(f"compute_bollinger_bands error: {e}", exc_info=True)
    return data
