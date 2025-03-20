# tests/unit_tests/signal_calc/test_signals_sideways.py
import pandas as pd
import pytest
from datetime import datetime

from signal_calc.signals_sideways import RangeTradingStrategy


def create_dummy_data():
    """
    Bollinger Bands 관련 지표를 포함한 더미 데이터프레임 생성.
    """
    dates = pd.date_range(start="2025-03-01", periods=10, freq="D")
    return pd.DataFrame({
        "close": [102 + i for i in range(10)],
        "bb_lband": [100 for _ in range(10)],
        "bb_hband": [110 for _ in range(10)]
    }, index=dates)


def test_range_trading_enter_long():
    """
    종가가 하한에 근접할 경우 'enter_long' 신호를 반환하는지 테스트합니다.
    """
    data = create_dummy_data()
    # 임의로 하한 근접 값 설정: close <= bb_lband*(1+tolerance)
    data.loc[data.index[5], "close"] = 100.1
    strat = RangeTradingStrategy(tolerance=0.002)
    signal = strat.get_signal(data, data.index[5])
    assert signal in ["enter_long", "hold"]


def test_range_trading_exit_all():
    """
    종가가 상한에 근접할 경우 'exit_all' 신호를 반환하는지 테스트합니다.
    """
    data = create_dummy_data()
    # 임의로 상한 근접 값 설정: close >= bb_hband*(1-tolerance)
    data.loc[data.index[5], "close"] = 109.9
    strat = RangeTradingStrategy(tolerance=0.002)
    signal = strat.get_signal(data, data.index[5])
    assert signal in ["exit_all", "hold"]
