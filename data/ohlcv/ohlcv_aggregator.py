# data/ohlcv/ohlcv_aggregator.py
import pandas as pd
from ta.trend import SMAIndicator
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def aggregate_to_weekly(
    df: pd.DataFrame,
    compute_indicators: bool = True,
    resample_rule: str = "W-MON",
    label: str = "left",
    closed: str = "left",
    timezone: str = None,
    sma_window: int = 5
) -> pd.DataFrame:
    """
    Aggregate OHLCV data to a weekly frequency, and optionally compute technical indicators.
    Optionally, adjust timezone of the index.

    Parameters:
        df (pd.DataFrame): OHLCV data with a datetime index.
        compute_indicators (bool): Whether to compute additional indicators (weekly SMA, momentum, volatility).
        resample_rule (str): Resample rule string for pandas (default "W-MON" means weekly starting on Monday).
        label (str): Labeling convention for the resample (default "left").
        closed (str): Which side is closed (default "left").
        timezone (str): Optional timezone string (e.g., "UTC", "Asia/Seoul") to convert the index.
        sma_window (int): Window size for computing weekly SMA.

    Returns:
        pd.DataFrame: Weekly aggregated DataFrame with columns:
                      open, weekly_high, weekly_low, close, volume, and, if requested,
                      weekly_sma, weekly_momentum, weekly_volatility.
    """
    # Validate required columns
    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns - set(df.columns)
    if missing:
        logger.error(f"Input DataFrame is missing required columns: {missing}", exc_info=True)
        return pd.DataFrame()

    # Ensure index is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            logger.error(f"Failed to convert index to datetime: {e}", exc_info=True)
            return pd.DataFrame()

    if df.empty:
        logger.error("Input DataFrame for aggregation is empty.", exc_info=True)
        return df

    try:
        if timezone:
            # If timezone is specified and the index is naive, localize to UTC then convert
            if df.index.tz is None:
                df = df.tz_localize('UTC')
            df = df.tz_convert(timezone)
    except Exception as e:
        logger.error(f"Timezone conversion error: {e}", exc_info=True)

    try:
        weekly = df.resample(rule=resample_rule, label=label, closed=closed).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    except Exception as e:
        logger.error(f"Resampling error: {e}", exc_info=True)
        return pd.DataFrame()

    if weekly.empty:
        logger.error("Aggregated weekly DataFrame is empty after resampling.", exc_info=True)
        return weekly

    # Rename columns for easier reference in strategies
    weekly.rename(columns={'high': 'weekly_high', 'low': 'weekly_low'}, inplace=True)

    if compute_indicators:
        try:
            # Weekly SMA: using a parameterized window (default 5)
            sma_indicator = SMAIndicator(close=weekly['close'], window=sma_window, fillna=True)
            weekly['weekly_sma'] = sma_indicator.sma_indicator()
            # Weekly momentum: percentage change of the weekly close
            weekly['weekly_momentum'] = weekly['close'].pct_change() * 100
            # Weekly volatility: (weekly_high - weekly_low) / weekly_low, with fallback for division by zero
            weekly['weekly_volatility'] = weekly.apply(
                lambda row: (row['weekly_high'] - row['weekly_low']) / row['weekly_low']
                if row['weekly_low'] != 0 else 0.0, axis=1)
        except Exception as e:
            logger.error(f"Error computing weekly indicators: {e}", exc_info=True)
    return weekly
