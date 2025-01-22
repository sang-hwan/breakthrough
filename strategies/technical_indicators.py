# strategies/technical_indicators.py

import pandas as pd

# `ta` 라이브러리를 사용하여 기술적 지표를 계산합니다.
from ta.trend import SMAIndicator, MACD # 추세 지표(SMA, MACD 등)
from ta.momentum import RSIIndicator # 모멘텀 지표(RSI 등)
from ta.volatility import BollingerBands # 변동성 지표(볼린저 밴드 등)

# 단순 이동평균(SMA) 계산 함수
def apply_sma(
    df: pd.DataFrame,  # 시세 데이터 (DataFrame 형식)
    price_col: str = 'close',  # 사용할 가격 데이터 컬럼 (기본값: '종가')
    window: int = 20,  # 이동평균 기간 (기본값: 20일)
    fillna: bool = False,  # 결측치(NaN) 처리 여부
    colname: str = 'sma'  # 계산 결과를 저장할 컬럼 이름
) -> pd.DataFrame:
    """
    단순 이동평균(SMA)을 계산하고 DataFrame에 결과를 추가합니다.
    """
    # ta 라이브러리의 SMAIndicator 클래스 생성
    indicator = SMAIndicator(
        close=df[price_col],
        window=window,
        fillna=fillna
    )
    # 계산된 SMA 값을 새로운 컬럼으로 추가
    df[colname] = indicator.sma_indicator()
    return df

# MACD (이동평균 수렴/확산) 계산 함수
def apply_macd(
    df: pd.DataFrame,
    price_col: str = 'close',  # 사용할 가격 데이터 컬럼
    window_slow: int = 26,  # 느린 이동평균 기간
    window_fast: int = 12,  # 빠른 이동평균 기간
    window_sign: int = 9,  # 시그널선 이동평균 기간
    fillna: bool = False,  # 결측치 처리 여부
    prefix: str = 'macd_'  # 결과 컬럼 접두사
) -> pd.DataFrame:
    """
    MACD 지표를 계산하고 MACD, 시그널선, 히스토그램을 추가합니다.
    """
    # MACD 지표 생성
    macd_indicator = MACD(
        close=df[price_col],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna
    )
    # 계산 결과를 DataFrame에 추가
    df[f'{prefix}macd'] = macd_indicator.macd()
    df[f'{prefix}signal'] = macd_indicator.macd_signal()
    df[f'{prefix}diff'] = macd_indicator.macd_diff()
    return df

# RSI (Relative Strength Index) 계산 함수
def apply_rsi(
    df: pd.DataFrame,
    price_col: str = 'close',  # 사용할 가격 데이터 컬럼
    window: int = 14,  # RSI 계산 기간
    fillna: bool = False,  # 결측치 처리 여부
    colname: str = 'rsi'  # 결과를 저장할 컬럼 이름
) -> pd.DataFrame:
    """
    RSI 지표를 계산하고 DataFrame에 추가합니다.
    """
    # RSIIndicator 클래스 생성
    rsi_indicator = RSIIndicator(
        close=df[price_col],
        window=window,
        fillna=fillna
    )
    # 계산된 RSI 값을 DataFrame에 추가
    df[colname] = rsi_indicator.rsi()
    return df

# 볼린저 밴드 계산 함수
def apply_bollinger(
    df: pd.DataFrame,
    price_col: str = 'close',  # 사용할 가격 데이터 컬럼
    window: int = 20,  # 이동평균 및 표준편차 계산 기간
    window_dev: float = 2.0,  # 표준편차 곱
    fillna: bool = False,  # 결측치 처리 여부
    prefix: str = 'bb_'  # 결과 컬럼 접두사
) -> pd.DataFrame:
    """
    볼린저 밴드를 계산하고 관련 데이터를 DataFrame에 추가합니다.
    """
    # BollingerBands 클래스 생성
    bb = BollingerBands(
        close=df[price_col],
        window=window,
        window_dev=window_dev,
        fillna=fillna
    )
    # 각 계산 결과를 DataFrame에 추가
    df[f'{prefix}mavg'] = bb.bollinger_mavg()  # 중간선
    df[f'{prefix}hband'] = bb.bollinger_hband()  # 상단선
    df[f'{prefix}lband'] = bb.bollinger_lband()  # 하단선
    df[f'{prefix}pband'] = bb.bollinger_pband()  # 퍼센트 밴드
    df[f'{prefix}wband'] = bb.bollinger_wband()  # 폭 밴드
    df[f'{prefix}hband_ind'] = bb.bollinger_hband_indicator()  # 상단선 접촉 여부
    df[f'{prefix}lband_ind'] = bb.bollinger_lband_indicator()  # 하단선 접촉 여부
    return df
