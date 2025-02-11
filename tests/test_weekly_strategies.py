# tests/test_weekly_strategies.py
import pytest
import pandas as pd
from trading.strategies import TradingStrategies

@pytest.fixture
def ts_instance():
    """TradingStrategies 인스턴스를 생성하는 fixture"""
    return TradingStrategies()

@pytest.fixture
def weekly_data_breakout():
    """
    주간 돌파 전략 테스트용 DataFrame 생성.
    인덱스: 2주치 데이터 (예: 2023-01-02, 2023-01-09)
    컬럼: high, low, close
    """
    # 첫 번째 주: 고점=100, 저점=90, 종가=95
    # 두 번째 주에서 테스트 대상 값을 변경하여 돌파 여부 확인
    dates = [pd.to_datetime("2023-01-02"), pd.to_datetime("2023-01-09")]
    data = {
        "high": [100, None],
        "low": [90, None],
        "close": [95, None]
    }
    df = pd.DataFrame(data, index=dates)
    return df

@pytest.fixture
def weekly_data_momentum():
    """
    주간 모멘텀 전략 테스트용 DataFrame 생성.
    인덱스: 1주치 데이터 (또는 2주치 데이터에서 마지막 행 사용)
    컬럼: weekly_momentum
    """
    dates = [pd.to_datetime("2023-01-09")]
    data = {
        "weekly_momentum": [0.0]  # 나중에 개별 테스트에서 값을 수정
    }
    df = pd.DataFrame(data, index=dates)
    return df

# === Weekly Breakout Strategy Tests ===

def test_weekly_breakout_enter_long(ts_instance, weekly_data_breakout):
    """
    전 주 고점을 1% 이상 돌파한 경우 "enter_long" 신호가 반환되어야 함.
    첫 주: high=100, low=90, 두 번째 주의 close를 102로 설정하면
    102 >= 100 * 1.01 (101.0) → "enter_long"
    """
    # 복사본 생성 후 두 번째 주 데이터 수정
    df = weekly_data_breakout.copy()
    df.at[df.index[1], "close"] = 102
    # 두 번째 주에 대한 테스트를 위해 current_time을 두 번째 주의 날짜로 설정
    current_time = df.index[1]
    signal = ts_instance.weekly_breakout_strategy(df, current_time, breakout_threshold=0.01)
    assert signal == "enter_long"

def test_weekly_breakout_exit_all(ts_instance, weekly_data_breakout):
    """
    전 주 저점을 1% 이상 하락한 경우 "exit_all" 신호가 반환되어야 함.
    첫 주: low=90, 두 번째 주의 close를 88로 설정하면
    88 <= 90 * 0.99 (89.1) → "exit_all"
    """
    df = weekly_data_breakout.copy()
    df.at[df.index[1], "close"] = 88
    current_time = df.index[1]
    signal = ts_instance.weekly_breakout_strategy(df, current_time, breakout_threshold=0.01)
    assert signal == "exit_all"

def test_weekly_breakout_hold(ts_instance, weekly_data_breakout):
    """
    돌파 조건을 충족하지 않으면 "hold" 신호가 반환되어야 함.
    예를 들어, 첫 주 high=100, 두 번째 주 close=95 → 조건 미충족
    """
    df = weekly_data_breakout.copy()
    df.at[df.index[1], "close"] = 95
    current_time = df.index[1]
    signal = ts_instance.weekly_breakout_strategy(df, current_time, breakout_threshold=0.01)
    assert signal == "hold"

# === Weekly Momentum Strategy Tests ===

def test_weekly_momentum_enter_long(ts_instance, weekly_data_momentum):
    """
    주간 모멘텀이 threshold 이상인 경우 "enter_long" 신호가 반환되어야 함.
    예: weekly_momentum = 0.6, momentum_threshold = 0.5
    """
    df = weekly_data_momentum.copy()
    df.at[df.index[0], "weekly_momentum"] = 0.6
    current_time = df.index[0]
    signal = ts_instance.weekly_momentum_strategy(df, current_time, momentum_threshold=0.5)
    assert signal == "enter_long"

def test_weekly_momentum_exit_all(ts_instance, weekly_data_momentum):
    """
    주간 모멘텀이 -threshold 이하인 경우 "exit_all" 신호가 반환되어야 함.
    예: weekly_momentum = -0.6, momentum_threshold = 0.5
    """
    df = weekly_data_momentum.copy()
    df.at[df.index[0], "weekly_momentum"] = -0.6
    current_time = df.index[0]
    signal = ts_instance.weekly_momentum_strategy(df, current_time, momentum_threshold=0.5)
    assert signal == "exit_all"

def test_weekly_momentum_hold(ts_instance, weekly_data_momentum):
    """
    주간 모멘텀이 임계치 내에 있으면 "hold" 신호가 반환되어야 함.
    예: weekly_momentum = 0.3, momentum_threshold = 0.5
    """
    df = weekly_data_momentum.copy()
    df.at[df.index[0], "weekly_momentum"] = 0.3
    current_time = df.index[0]
    signal = ts_instance.weekly_momentum_strategy(df, current_time, momentum_threshold=0.5)
    assert signal == "hold"
