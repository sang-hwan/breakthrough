# data/ohlcv/ohlcv_aggregator.py
import pandas as pd
from ta.trend import SMAIndicator

def aggregate_to_weekly(df: pd.DataFrame, compute_indicators: bool = True) -> pd.DataFrame:
    """
    입력 데이터프레임(df)을 주간(월요일 시작) 단위로 집계합니다.
    - 주간 시가: 각 주의 첫 행의 open 값
    - 주간 최고가: 해당 주의 high 중 최대값
    - 주간 최저가: 해당 주의 low 중 최소값
    - 주간 종가: 각 주의 마지막 행의 close 값
    - 거래량: 해당 주의 volume 합계
    옵션 compute_indicators=True인 경우 주간 SMA와 주간 모멘텀을 추가합니다.
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

    if compute_indicators:
        sma_indicator = SMAIndicator(close=weekly['close'], window=5, fillna=True)
        weekly['weekly_sma'] = sma_indicator.sma_indicator()
        weekly['weekly_momentum'] = weekly['close'].pct_change() * 100

    return weekly
