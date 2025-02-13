# tests/test_ensemble.py
import pytest
import pandas as pd
from trading.ensemble import Ensemble

@pytest.fixture
def dummy_data():
    # 최소한의 컬럼을 가진 간단한 데이터프레임 생성
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    df = pd.DataFrame({
        "open": [100]*10,
        "close": [101]*10,
        "high": [102]*10,
        "low": [99]*10,
        "sma": [100.5]*10,
        "rsi": [30]*10,
        "bb_lband": [99]*10
    }, index=dates)
    return df

@pytest.fixture
def dummy_weekly_data():
    dates = pd.date_range("2023-01-01", periods=2, freq="W-MON")
    df = pd.DataFrame({
        "close": [101, 103],
        "high": [102, 104],
        "low": [99, 100],
        "weekly_momentum": [0.6, 0.6]
    }, index=dates)
    return df

def test_get_final_signal(dummy_data, dummy_weekly_data):
    ens = Ensemble()
    current_time = dummy_data.index[-1]
    final_signal = ens.get_final_signal(market_regime="bullish", liquidity_info="high", data=dummy_data, current_time=current_time, data_weekly=dummy_weekly_data)
    assert final_signal in ["enter_long", "exit_all", "hold"]
