# strategies/technical_indicators.py
# SMA, MACD, RSI, 볼린저밴드 등 다양한 기술적 지표를 쉽게 적용하기 위해 만든 함수 모음.

import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def apply_sma(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 20,
    fillna: bool = False,
    colname: str = 'sma'
) -> pd.DataFrame:
    """
    SMA(단순 이동평균)을 계산해 컬럼을 추가합니다.
    """

    indicator = SMAIndicator(
        close=df[price_col],
        window=window,
        fillna=fillna
    )
    df[colname] = indicator.sma_indicator()
    return df


def apply_macd(
    df: pd.DataFrame,
    price_col: str = 'close',
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
    fillna: bool = False,
    prefix: str = 'macd_'
) -> pd.DataFrame:
    """
    MACD 지표 (움직이는 두 이동평균의 차이).
    - macd: 기본 MACD 라인
    - signal: 시그널(9일 이동평균)
    - diff: 둘의 차이
    """

    macd_indicator = MACD(
        close=df[price_col],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
        fillna=fillna
    )
    df[f'{prefix}macd'] = macd_indicator.macd()
    df[f'{prefix}signal'] = macd_indicator.macd_signal()
    df[f'{prefix}diff'] = macd_indicator.macd_diff()
    return df


def apply_rsi(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 14,
    fillna: bool = False,
    colname: str = 'rsi'
) -> pd.DataFrame:
    """
    RSI(과매수/과매도 지표)를 계산해 DataFrame에 컬럼 추가.
    """

    rsi_indicator = RSIIndicator(
        close=df[price_col],
        window=window,
        fillna=fillna
    )
    df[colname] = rsi_indicator.rsi()
    return df


def apply_bollinger(
    df: pd.DataFrame,
    price_col: str = 'close',
    window: int = 20,
    window_dev: float = 2.0,
    fillna: bool = False,
    prefix: str = 'bb_'
) -> pd.DataFrame:
    """
    볼린저 밴드를 계산해 (상단선, 중간선, 하단선 등) 여러 컬럼을 추가.
    """

    bb = BollingerBands(
        close=df[price_col],
        window=window,
        window_dev=window_dev,
        fillna=fillna
    )
    df[f'{prefix}mavg'] = bb.bollinger_mavg()
    df[f'{prefix}hband'] = bb.bollinger_hband()
    df[f'{prefix}lband'] = bb.bollinger_lband()
    df[f'{prefix}pband'] = bb.bollinger_pband()
    df[f'{prefix}wband'] = bb.bollinger_wband()
    df[f'{prefix}hband_ind'] = bb.bollinger_hband_indicator()
    df[f'{prefix}lband_ind'] = bb.bollinger_lband_indicator()
    return df
