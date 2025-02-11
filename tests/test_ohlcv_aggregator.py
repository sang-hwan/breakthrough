# tests/test_ohlcv_aggregator.py
import pandas as pd
import numpy as np
from data_collection.ohlcv_aggregator import aggregate_to_weekly

def test_aggregate_to_weekly():
    # 예시 데이터 생성: 2주 이상의 분 단위(또는 1일) 데이터
    dates = pd.date_range(start="2023-01-02", periods=14, freq="D")  # 월요일 시작 가정
    data = pd.DataFrame({
        'open': np.linspace(100, 113, 14),
        'high': np.linspace(105, 118, 14),
        'low': np.linspace(95, 108, 14),
        'close': np.linspace(102, 115, 14),
        'volume': np.random.randint(1000, 5000, 14)
    }, index=dates)
    
    weekly = aggregate_to_weekly(data, compute_indicators=True)
    
    # 각 주의 open은 그룹의 첫 행, close는 마지막 행 등으로 검증
    # 예: 첫 주 (2023-01-02 ~ 2023-01-08)
    first_week = data.loc["2023-01-02":"2023-01-08"]
    assert np.isclose(weekly.iloc[0]['open'], first_week.iloc[0]['open'])
    assert np.isclose(weekly.iloc[0]['close'], first_week.iloc[-1]['close'])
    # volume 합계 비교
    assert np.isclose(weekly.iloc[0]['volume'], first_week['volume'].sum())
    # 주간 인디케이터 컬럼이 추가되었는지 확인
    assert 'weekly_sma' in weekly.columns
    assert 'weekly_momentum' in weekly.columns
