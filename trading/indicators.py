# trading/indicators.py
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def compute_sma(
    data: pd.DataFrame,
    price_column: str = 'close',
    period: int = 20,
    fillna: bool = False,
    output_col: str = 'sma'
) -> pd.DataFrame:
    sma = SMAIndicator(close=data[price_column], window=period, fillna=fillna)
    data[output_col] = sma.sma_indicator()
    return data

def compute_macd(
    data: pd.DataFrame,
    price_column: str = 'close',
    slow_period: int = 26,
    fast_period: int = 12,
    signal_period: int = 9,
    fillna: bool = False,
    prefix: str = 'macd_'
) -> pd.DataFrame:
    macd = MACD(close=data[price_column],
                window_slow=slow_period,
                window_fast=fast_period,
                window_sign=signal_period,
                fillna=fillna)
    data[f'{prefix}macd'] = macd.macd()
    data[f'{prefix}signal'] = macd.macd_signal()
    data[f'{prefix}diff'] = macd.macd_diff()
    return data

def compute_rsi(
    data: pd.DataFrame,
    price_column: str = 'close',
    period: int = 14,
    fillna: bool = False,
    output_col: str = 'rsi'
) -> pd.DataFrame:
    rsi = RSIIndicator(close=data[price_column], window=period, fillna=fillna)
    data[output_col] = rsi.rsi()
    return data

def compute_bollinger_bands(
    data: pd.DataFrame,
    price_column: str = 'close',
    period: int = 20,
    std_multiplier: float = 2.0,
    fillna: bool = False,
    prefix: str = 'bb_'
) -> pd.DataFrame:
    bb = BollingerBands(close=data[price_column], window=period, window_dev=std_multiplier, fillna=fillna)
    data[f'{prefix}mavg'] = bb.bollinger_mavg()
    data[f'{prefix}hband'] = bb.bollinger_hband()
    data[f'{prefix}lband'] = bb.bollinger_lband()
    data[f'{prefix}pband'] = bb.bollinger_pband()
    data[f'{prefix}wband'] = bb.bollinger_wband()
    data[f'{prefix}hband_ind'] = bb.bollinger_hband_indicator()
    data[f'{prefix}lband_ind'] = bb.bollinger_lband_indicator()
    return data
