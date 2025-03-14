# tests/backtesting/test_ohlcv_aggregator.py

import pandas as pd
import numpy as np
# 주간(weekly) 데이터로 집계하는 함수를 임포트합니다.
from data.ohlcv.ohlcv_aggregator import aggregate_to_weekly

def test_aggregate_to_weekly_includes_weekly_low_high():
    """
    주간 데이터 집계 시, weekly_low 및 weekly_high 컬럼이 포함되는지와
    최소 두 개 이상의 주간 데이터가 생성되는지 테스트합니다.
    
    - 21일치 데이터(약 3주 분량)를 생성한 후, aggregate_to_weekly 함수를 호출합니다.
    - 결과 DataFrame에 weekly_low와 weekly_high 컬럼이 포함되어 있는지 검증합니다.
    - 집계 결과로 생성된 주간 데이터가 2개 이상임을 확인합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    # 21일치 날짜 범위 생성
    dates = pd.date_range(start='2020-01-01', periods=21, freq='D')
    # OHLCV 데이터 생성 (open, high, low, close는 선형 분포, volume은 임의의 정수)
    data = pd.DataFrame({
        'open': np.linspace(100, 120, len(dates)),
        'high': np.linspace(105, 125, len(dates)),
        'low': np.linspace(95, 115, len(dates)),
        'close': np.linspace(102, 122, len(dates)),
        'volume': np.random.randint(100, 200, len(dates))
    }, index=dates)
    # 주간 데이터로 집계 (compute_indicators=True, SMA 윈도우 5)
    weekly = aggregate_to_weekly(data, compute_indicators=True, sma_window=5)
    # weekly_low, weekly_high 컬럼이 존재하는지 확인
    assert 'weekly_low' in weekly.columns
    assert 'weekly_high' in weekly.columns
    # 주간 데이터가 최소 2개 이상 생성되었는지 확인
    assert len(weekly) >= 2
