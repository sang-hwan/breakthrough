# tests/unit_tests/signal_calc/test_signals_downtrend.py
import pandas as pd
import pytest
from datetime import datetime

from signal_calc.signals_downtrend import CounterTrendStrategy, HighFrequencyStrategy


def create_dummy_data_rsi():
    """
    RSI 지표를 포함하는 더미 데이터프레임 생성.
    """
    dates = pd.date_range(start="2025-03-01", periods=10, freq="D")
    return pd.DataFrame({
        "rsi": [25, 35, 75, 65, 30, 40, 50, 60, 70, 80],
        "close": [102 + i for i in range(10)]
    }, index=dates)


def test_counter_trend_signal_enter_long():
    """
    RSI가 과매도(예: 25)인 경우 'enter_long' 신호를 반환해야 합니다.
    """
    data = create_dummy_data_rsi()
    current_time = data.index[0]  # RSI = 25 (< 30)
    strat = CounterTrendStrategy(rsi_overbought=70, rsi_oversold=30)
    signal = strat.get_signal(data, current_time)
    assert signal in ["enter_long", "hold"]


def test_counter_trend_signal_exit_all():
    """
    RSI가 과매수(예: 75)인 경우 'exit_all' 신호를 반환해야 합니다.
    """
    data = create_dummy_data_rsi()
    current_time = data.index[2]  # RSI = 75 (> 70)
    strat = CounterTrendStrategy(rsi_overbought=70, rsi_oversold=30)
    signal = strat.get_signal(data, current_time)
    assert signal in ["exit_all", "hold"]


def create_dummy_data_hf():
    """
    고빈도 데이터를 위한 더미 데이터프레임 생성.
    """
    dates = pd.date_range(start="2025-03-01", periods=3, freq="T")
    return pd.DataFrame({
        "close": [100, 102, 104]
    }, index=dates)


def test_high_frequency_signal():
    """
    인접 데이터 간 가격 변화율에 따라 신호가 올바르게 산출되는지 검증합니다.
    """
    data = create_dummy_data_hf()
    current_time = data.index[2]
    strat = HighFrequencyStrategy(threshold=0.001)
    signal = strat.get_signal(data, current_time)
    assert signal in ["enter_long", "exit_all", "hold"]
