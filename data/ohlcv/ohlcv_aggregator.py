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
    timezone: str = None
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

    Returns:
        pd.DataFrame: Weekly aggregated DataFrame with columns:
                      open, weekly_high, weekly_low, close, volume, and, if requested,
                      weekly_sma, weekly_momentum, weekly_volatility.
    """
    if df.empty:
        logger.error("Input DataFrame for aggregation is empty.")
        return df

    try:
        if timezone:
            # 만약 타임존 지정 시, 기존 인덱스가 timezone-naive라면 UTC로 간주 후 변환
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
        logger.error("Aggregated weekly DataFrame is empty after resampling.")
        return weekly

    # 컬럼 명칭 재지정: 전략 모듈에서 쉽게 참조할 수 있도록 high→weekly_high, low→weekly_low
    weekly.rename(columns={'high': 'weekly_high', 'low': 'weekly_low'}, inplace=True)

    if compute_indicators:
        try:
            # 주간 SMA: 기본적으로 5주간의 단순 이동 평균
            sma_indicator = SMAIndicator(close=weekly['close'], window=5, fillna=True)
            weekly['weekly_sma'] = sma_indicator.sma_indicator()
            # 주간 모멘텀: 주간 종가의 백분율 변화
            weekly['weekly_momentum'] = weekly['close'].pct_change() * 100
            # 주간 변동성: (주간 최고가 - 주간 최저가) / 주간 최저가
            weekly['weekly_volatility'] = (weekly['weekly_high'] - weekly['weekly_low']) / weekly['weekly_low']
        except Exception as e:
            logger.error(f"Error computing weekly indicators: {e}", exc_info=True)

    return weekly
