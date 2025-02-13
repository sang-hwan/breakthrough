# tests/test_ohlcv_aggregator.py
import pandas as pd
import numpy as np
from data.ohlcv.ohlcv_aggregator import aggregate_to_weekly

def test_aggregate_to_weekly():
    # 2주 이상의 데이터 (1일 간격, 월요일 시작 가정)
    dates = pd.date_range(start="2023-01-02", periods=14, freq="D")
    data = pd.DataFrame({
        'open': np.linspace(100, 113, 14),
        'high': np.linspace(105, 118, 14),
        'low': np.linspace(95, 108, 14),
        'close': np.linspace(102, 115, 14),
        'volume': np.random.randint(1000, 5000, 14)
    }, index=dates)
    
    weekly = aggregate_to_weekly(data, compute_indicators=True)
    
    # 첫 주의 open과 close 확인
    first_week = data.loc["2023-01-02":"2023-01-08"]
    assert abs(weekly.iloc[0]['open'] - first_week.iloc[0]['open']) < 1e-6
    assert abs(weekly.iloc[0]['close'] - first_week.iloc[-1]['close']) < 1e-6
    # volume 합계 확인
    assert abs(weekly.iloc[0]['volume'] - first_week['volume'].sum()) < 1e-6
    # 주간 인디케이터 컬럼 확인
    assert 'weekly_sma' in weekly.columns
    assert 'weekly_momentum' in weekly.columns
