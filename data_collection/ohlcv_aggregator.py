# data_collection/ohlcv_aggregator.py
import pandas as pd
from ta.trend import SMAIndicator

def aggregate_to_weekly(df: pd.DataFrame, compute_indicators: bool = True) -> pd.DataFrame:
    """
    입력 데이터프레임(df)의 날짜 인덱스를 기준으로 주간(월요일 시작) 단위 그룹화하여 집계한다.
    - 주간 시가: 각 주의 첫 행의 open 값
    - 주간 최고가: 해당 주의 high 값 중 최대값
    - 주간 최저가: 해당 주의 low 값 중 최소값
    - 주간 종가: 각 주의 마지막 행의 close 값
    - 거래량: 해당 주의 volume 값 합계
    옵션: compute_indicators=True인 경우 주간 SMA, 모멘텀(주간 수익률) 등의 인디케이터를 계산하여 컬럼에 추가한다.
    
    인자:
      df: 입력 OHLCV 데이터 (인덱스는 datetime, 컬럼은 최소 'open', 'high', 'low', 'close', 'volume')
      compute_indicators: 주간 인디케이터 계산 여부 (기본 True)
    
    반환:
      주간 캔들 데이터프레임 (필요 시 인디케이터 컬럼 추가)
    """
    if df.empty:
        return df

    # resample: 주 단위 집계 (ISO 기준, 월요일 시작)
    weekly = df.resample('W-MON', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # 데이터가 비어있는 경우 예외 처리
    if weekly.empty:
        return weekly

    if compute_indicators:
        # 예시: 5주 SMA (주간 종가 기준)
        sma_indicator = SMAIndicator(close=weekly['close'], window=5, fillna=True)
        weekly['weekly_sma'] = sma_indicator.sma_indicator()
        # 예시: 주간 모멘텀 지표 (전주 대비 종가 변화율, %)
        weekly['weekly_momentum'] = weekly['close'].pct_change() * 100

    return weekly
