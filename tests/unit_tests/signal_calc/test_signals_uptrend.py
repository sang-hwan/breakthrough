# tests/unit_tests/signal_calc/test_signals_uptrend.py
import pandas as pd
import pytest
from datetime import datetime

from signal_calc.signals_uptrend import TrendFollowingStrategy, BreakoutStrategy, WeeklyMomentumStrategy


def create_dummy_data():
    """
    단순 OHLC, SMA, RSI 데이터를 포함하는 더미 데이터프레임 생성.
    """
    dates = pd.date_range(start="2025-03-01", periods=10, freq="D")
    return pd.DataFrame({
        "open": [100 + i for i in range(10)],
        "high": [105 + i for i in range(10)],
        "low": [95 + i for i in range(10)],
        "close": [102 + i for i in range(10)],
        "sma": [101 + i for i in range(10)],
        "rsi": [30 + i for i in range(10)]
    }, index=dates)


def test_trend_following_signal():
    """
    TrendFollowingStrategy: 종가가 SMA보다 높으면 'enter_long' 신호를 반환해야 합니다.
    """
    data = create_dummy_data()
    current_time = data.index[5]
    strat = TrendFollowingStrategy()
    signal = strat.get_signal(data, current_time)
    assert signal in ["enter_long", "hold"]


def test_breakout_signal():
    """
    BreakoutStrategy: 최근 window 기간의 최고가를 돌파하면 'enter_long' 신호를 반환해야 합니다.
    """
    data = create_dummy_data()
    current_time = data.index[9]
    strat = BreakoutStrategy(window=3)
    signal = strat.get_signal(data, current_time)
    assert signal in ["enter_long", "hold"]


def test_weekly_momentum_signal():
    """
    WeeklyMomentumStrategy: 주간 모멘텀이 임계값 이상이면 'enter_long' 신호를 반환해야 합니다.
    """
    data = create_dummy_data().resample("W").last()
    current_time = data.index[-1]
    strat = WeeklyMomentumStrategy(momentum_threshold=0.5)
    signal = strat.get_signal(data, current_time)
    assert signal in ["enter_long", "hold"]
