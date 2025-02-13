# data/ohlcv/ohlcv_aggregator.py
import pandas as pd
from ta.trend import SMAIndicator
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def aggregate_to_weekly(df: pd.DataFrame, compute_indicators: bool = True) -> pd.DataFrame:
    if df.empty:
        return df

    weekly = df.resample('W-MON', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    if weekly.empty:
        return weekly

    if compute_indicators:
        sma_indicator = SMAIndicator(close=weekly['close'], window=5, fillna=True)
        weekly['weekly_sma'] = sma_indicator.sma_indicator()
        weekly['weekly_momentum'] = weekly['close'].pct_change() * 100
    return weekly
