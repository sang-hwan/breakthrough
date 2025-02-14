# tests/test_strategies.py
import pandas as pd
import pytest
from strategies.trading_strategies import (
    SelectStrategy, TrendFollowingStrategy, BreakoutStrategy,
    CounterTrendStrategy, HighFrequencyStrategy,
    WeeklyBreakoutStrategy, WeeklyMomentumStrategy,
    TradingStrategies
)

@pytest.fixture
def sample_data():
    # 간단한 테스트용 데이터프레임 (일별 데이터)
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    data = pd.DataFrame({
        "open": [100, 102, 103, 104, 105],
        "high": [105, 107, 108, 109, 110],
        "low": [95, 97, 98, 99, 100],
        "close": [102, 104, 103, 107, 108],
        "sma": [100, 101, 102, 103, 104],
        "rsi": [30, 35, 40, 25, 45],
        "bb_lband": [98, 99, 100, 101, 102],
    }, index=dates)
    return data

@pytest.fixture
def sample_weekly_data():
    # 간단한 주간 데이터프레임
    dates = pd.date_range(start="2023-01-01", periods=3, freq="W-MON")
    data = pd.DataFrame({
        "open": [100, 110, 120],
        "high": [105, 115, 125],
        "low": [95, 105, 115],
        "close": [102, 112, 122],
        "weekly_momentum": [0.5, 0.6, 0.4],
    }, index=dates)
    return data

def test_select_strategy(sample_data):
    strat = SelectStrategy()
    current_time = sample_data.index[-1]
    signal = strat.get_signal(sample_data, current_time)
    assert signal in ["enter_long", "hold"]

def test_trend_following_strategy(sample_data):
    strat = TrendFollowingStrategy()
    current_time = sample_data.index[-1]
    signal = strat.get_signal(sample_data, current_time)
    assert signal in ["enter_long", "hold"]

def test_breakout_strategy(sample_data):
    strat = BreakoutStrategy(window=3)
    current_time = sample_data.index[-1]
    signal = strat.get_signal(sample_data, current_time)
    assert signal in ["enter_long", "hold"]

def test_counter_trend_strategy(sample_data):
    strat = CounterTrendStrategy()
    current_time = sample_data.index[-1]
    # RSI 낮은 경우 -> 진입 신호
    sample_data.loc[current_time, 'rsi'] = 25
    signal = strat.get_signal(sample_data, current_time)
    assert signal == "enter_long"
    # RSI 높은 경우 -> 청산 신호
    sample_data.loc[current_time, 'rsi'] = 75
    signal = strat.get_signal(sample_data, current_time)
    assert signal == "exit_all"

def test_high_frequency_strategy():
    # 2분 단위의 간단한 데이터 생성
    dates = pd.date_range(start="2023-01-01", periods=2, freq="min")
    data = pd.DataFrame({"close": [100, 100.5]}, index=dates)
    strat = HighFrequencyStrategy()
    signal = strat.get_signal(data, data.index[-1])
    assert signal in ["enter_long", "exit_all", "hold"]

def test_weekly_breakout_strategy(sample_weekly_data):
    strat = WeeklyBreakoutStrategy()
    current_time = sample_weekly_data.index[-1]
    signal = strat.get_signal(sample_weekly_data, current_time, breakout_threshold=0.01)
    assert signal in ["enter_long", "exit_all", "hold"]

def test_weekly_momentum_strategy(sample_weekly_data):
    strat = WeeklyMomentumStrategy()
    current_time = sample_weekly_data.index[-1]
    signal = strat.get_signal(sample_weekly_data, current_time, momentum_threshold=0.5)
    assert signal in ["enter_long", "exit_all", "hold"]

def test_trading_strategies_ensemble(sample_data, sample_weekly_data):
    ensemble = TradingStrategies()
    current_time = sample_data.index[-1]
    signal = ensemble.get_final_signal("bullish", "high", sample_data, current_time, data_weekly=sample_weekly_data)
    assert signal in ["enter_long", "exit_all", "hold"]
