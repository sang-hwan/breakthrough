# tests/backtesting/test_ohlcv_aggregator.py
import pandas as pd
import numpy as np
from data.ohlcv.ohlcv_aggregator import aggregate_to_weekly

def test_aggregate_to_weekly_includes_weekly_low_high():
    # 21일치 데이터(약 3주 분량)를 생성하여 주간 집계 시 두 개 이상의 주간 데이터가 생성되도록 함
    dates = pd.date_range(start='2020-01-01', periods=21, freq='D')
    data = pd.DataFrame({
        'open': np.linspace(100, 120, len(dates)),
        'high': np.linspace(105, 125, len(dates)),
        'low': np.linspace(95, 115, len(dates)),
        'close': np.linspace(102, 122, len(dates)),
        'volume': np.random.randint(100, 200, len(dates))
    }, index=dates)
    weekly = aggregate_to_weekly(data, compute_indicators=True, sma_window=5)
    # weekly_low, weekly_high가 존재하는지 확인
    assert 'weekly_low' in weekly.columns
    assert 'weekly_high' in weekly.columns
    # 집계 결과가 두 주 이상 생성되었는지 확인
    assert len(weekly) >= 2
