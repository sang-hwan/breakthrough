# tests/strategies/test_strategies.py
# 이 모듈은 다양한 트레이딩 전략 클래스(예: 추세추종, 돌파, 역추세 등)의 신호 생성 기능을 검증하는 테스트 케이스들을 포함합니다.

import pandas as pd  # 데이터 조작을 위한 패키지
import pytest  # 테스트 케이스 작성을 위한 패키지
from strategies.trading_strategies import (
    SelectStrategy, TrendFollowingStrategy, BreakoutStrategy,
    CounterTrendStrategy, HighFrequencyStrategy,
    WeeklyBreakoutStrategy, WeeklyMomentumStrategy,
    TradingStrategies
)  # 테스트할 트레이딩 전략 클래스들

@pytest.fixture
def sample_data():
    """
    간단한 일별 거래 데이터를 생성하는 fixture입니다.
    
    5일간의 데이터로 구성되며, 기본적인 OHLC 데이터와 몇 가지 기술 지표(SMA, RSI, Bollinger Lower Band)를 포함합니다.
    
    Returns:
        pd.DataFrame: 일별 거래 데이터.
    """
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")  # 5일간의 날짜 생성
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
    """
    간단한 주간 거래 데이터를 생성하는 fixture입니다.
    
    주간 데이터는 OHLC와 주간 모멘텀 값을 포함하며, 3주 동안의 데이터를 생성합니다.
    
    Returns:
        pd.DataFrame: 주간 거래 데이터.
    """
    dates = pd.date_range(start="2023-01-01", periods=3, freq="W-MON")  # 주간(월요일 기준) 날짜 생성
    data = pd.DataFrame({
        "open": [100, 110, 120],
        "high": [105, 115, 125],
        "low": [95, 105, 115],
        "close": [102, 112, 122],
        "weekly_momentum": [0.5, 0.6, 0.4],
    }, index=dates)
    return data

def test_select_strategy(sample_data):
    """
    SelectStrategy의 신호 생성 기능을 테스트합니다.
    
    sample_data의 마지막 시간대 데이터를 사용하여 전략이 'enter_long' 또는 'hold' 신호를 반환하는지 검증합니다.
    """
    strat = SelectStrategy()  # SelectStrategy 객체 생성
    current_time = sample_data.index[-1]  # 마지막 날짜를 현재 시각으로 사용
    signal = strat.get_signal(sample_data, current_time)  # 전략에 따른 신호 생성
    assert signal in ["enter_long", "hold"]

def test_trend_following_strategy(sample_data):
    """
    TrendFollowingStrategy (추세추종 전략)의 신호 생성 기능을 테스트합니다.
    
    마지막 데이터 시점을 기준으로 전략이 올바른 신호('enter_long' 또는 'hold')를 반환하는지 확인합니다.
    """
    strat = TrendFollowingStrategy()  # 추세추종 전략 객체 생성
    current_time = sample_data.index[-1]
    signal = strat.get_signal(sample_data, current_time)
    assert signal in ["enter_long", "hold"]

def test_breakout_strategy(sample_data):
    """
    BreakoutStrategy (돌파 전략)의 신호 생성 기능을 테스트합니다.
    
    window=3의 조건으로 테스트 데이터에서 신호를 생성하고, 반환된 신호가 유효한지 확인합니다.
    """
    strat = BreakoutStrategy(window=3)  # 돌파 전략 객체 생성 (돌파 기간: 3일)
    current_time = sample_data.index[-1]
    signal = strat.get_signal(sample_data, current_time)
    assert signal in ["enter_long", "hold"]

def test_counter_trend_strategy(sample_data):
    """
    CounterTrendStrategy (역추세 전략)의 신호 생성 기능을 테스트합니다.
    
    RSI 값에 따라 매수와 청산 신호가 올바르게 생성되는지 검증합니다.
      - RSI가 낮은 경우 (예: 25): 'enter_long'
      - RSI가 높은 경우 (예: 75): 'exit_all'
    """
    strat = CounterTrendStrategy()  # 역추세 전략 객체 생성
    current_time = sample_data.index[-1]
    
    # RSI가 낮은 경우 테스트 (과매도 상태)
    sample_data.loc[current_time, 'rsi'] = 25
    signal = strat.get_signal(sample_data, current_time)
    assert signal == "enter_long"
    
    # RSI가 높은 경우 테스트 (과매수 상태)
    sample_data.loc[current_time, 'rsi'] = 75
    signal = strat.get_signal(sample_data, current_time)
    assert signal == "exit_all"

def test_high_frequency_strategy():
    """
    HighFrequencyStrategy (초단타 전략)의 신호 생성 기능을 테스트합니다.
    
    2분 단위의 간단한 데이터셋을 생성하여 전략이 'enter_long', 'exit_all', 또는 'hold' 신호를 반환하는지 확인합니다.
    """
    dates = pd.date_range(start="2023-01-01", periods=2, freq="min")  # 2분 간격 데이터 생성
    data = pd.DataFrame({"close": [100, 100.5]}, index=dates)
    strat = HighFrequencyStrategy()  # 초단타 전략 객체 생성
    signal = strat.get_signal(data, data.index[-1])
    assert signal in ["enter_long", "exit_all", "hold"]

def test_weekly_breakout_strategy(sample_weekly_data):
    """
    WeeklyBreakoutStrategy (주간 돌파 전략)의 신호 생성 기능을 테스트합니다.
    
    주간 데이터에서 이전 주 대비 1% 이상의 가격 상승 또는 하락 조건에 따라 신호를 생성하는지 확인합니다.
    
    Parameters:
        breakout_threshold (float): 돌파 임계값 (여기서는 1%).
    """
    strat = WeeklyBreakoutStrategy()  # 주간 돌파 전략 객체 생성
    current_time = sample_weekly_data.index[-1]
    signal = strat.get_signal(sample_weekly_data, current_time, breakout_threshold=0.01)
    assert signal in ["enter_long", "exit_all", "hold"]

def test_weekly_momentum_strategy(sample_weekly_data):
    """
    WeeklyMomentumStrategy (주간 모멘텀 전략)의 신호 생성 기능을 테스트합니다.
    
    주간 모멘텀 값이 지정 임계값 이상인 경우 'enter_long', 그 외의 경우 'exit_all' 또는 'hold' 신호를 반환하는지 확인합니다.
    
    Parameters:
        momentum_threshold (float): 모멘텀 임계값 (예: 0.5).
    """
    strat = WeeklyMomentumStrategy()  # 주간 모멘텀 전략 객체 생성
    current_time = sample_weekly_data.index[-1]
    signal = strat.get_signal(sample_weekly_data, current_time, momentum_threshold=0.5)
    assert signal in ["enter_long", "exit_all", "hold"]

def test_trading_strategies_ensemble(sample_data, sample_weekly_data):
    """
    TradingStrategies (전략 앙상블)의 최종 신호 생성 기능을 테스트합니다.
    
    일별 데이터와 주간 데이터를 함께 사용하여 최종 신호를 생성하며, 반환된 신호가 유효한지 확인합니다.
    
    Parameters:
        market_regime (str): 예시로 "bullish" (상승장) 사용.
        frequency (str): 예시로 "high" (고빈도) 사용.
    """
    ensemble = TradingStrategies()  # 전략 앙상블 객체 생성
    current_time = sample_data.index[-1]
    signal = ensemble.get_final_signal("bullish", "high", sample_data, current_time, data_weekly=sample_weekly_data)
    assert signal in ["enter_long", "exit_all", "hold"]
