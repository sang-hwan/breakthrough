# tests/integration_tests/signal_calc/test_integration_signal_calc.py
import pandas as pd
import pytest
from datetime import datetime

from signal_calc.calc_signal import Ensemble


def create_dummy_data():
    """
    단기 OHLC, SMA, RSI, Bollinger Bands 등 다양한 지표를 포함하는 더미 데이터 생성.
    """
    dates = pd.date_range(start="2025-03-01", periods=20, freq="D")
    return pd.DataFrame({
        "open": [100 + i for i in range(20)],
        "high": [105 + i for i in range(20)],
        "low": [95 + i for i in range(20)],
        "close": [102 + i for i in range(20)],
        "sma": [101 + i for i in range(20)],
        "rsi": [25 if i % 3 == 0 else 50 for i in range(20)],
        "bb_lband": [98 for _ in range(20)],
        "bb_hband": [110 for _ in range(20)]
    }, index=dates)


def create_dummy_weekly_data(data):
    """
    주간 데이터로 리샘플링 후, 주간 모멘텀 지표를 포함하는 더미 데이터 생성.
    """
    weekly_data = data.resample('W').last()
    weekly_data["weekly_momentum"] = [0.6 if i % 2 == 0 else 0.4 for i in range(len(weekly_data))]
    return weekly_data


def test_integration_ensemble_signal():
    """
    전체 앙상블 전략이 bullish 시장 상황에서 최종 신호를 도출하는지 검증합니다.
    """
    data = create_dummy_data()
    weekly_data = create_dummy_weekly_data(data)
    ensemble = Ensemble()
    current_time = data.index[-1]
    final_signal = ensemble.get_final_signal(
        market_regime="bullish",
        liquidity_info="high",
        data=data,
        current_time=current_time,
        data_weekly=weekly_data,
        market_volatility=0.03,
        volume=1500
    )
    assert final_signal in ["enter_long", "exit_all", "hold"]
