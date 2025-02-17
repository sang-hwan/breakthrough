# data/ohlcv/ohlcv_aggregator.py
import pandas as pd
from ta.trend import SMAIndicator
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def aggregate_to_weekly(df: pd.DataFrame, compute_indicators: bool = True) -> pd.DataFrame:
    """
    Aggregate OHLCV data to a weekly frequency, and optionally compute technical indicators.
    Additionally, renames 'high' and 'low' columns to 'weekly_high' and 'weekly_low' to support
    the weekly breakout and momentum strategies.
    """
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

    # Rename columns for clarity in strategy configuration
    weekly.rename(columns={'high': 'weekly_high', 'low': 'weekly_low'}, inplace=True)

    if compute_indicators:
        sma_indicator = SMAIndicator(close=weekly['close'], window=5, fillna=True)
        weekly['weekly_sma'] = sma_indicator.sma_indicator()
        weekly['weekly_momentum'] = weekly['close'].pct_change() * 100
        # Compute weekly volatility as a ratio
        weekly['weekly_volatility'] = (weekly['weekly_high'] - weekly['weekly_low']) / weekly['weekly_low']
    return weekly
