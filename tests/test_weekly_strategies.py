# tests/test_weekly_strategies.py
import pandas as pd
import numpy as np
from strategies.trading_strategies import WeeklyBreakoutStrategy, WeeklyMomentumStrategy

def create_weekly_data():
    # 주간 데이터 생성을 위해, 주간 집계에 필요한 필드들을 포함하여 4주치 데이터 생성
    dates = pd.date_range(start='2020-01-06', periods=4, freq='W-MON')
    data = pd.DataFrame({
        'open': np.linspace(100, 110, len(dates)),
        'weekly_high': np.linspace(105, 115, len(dates)),
        'weekly_low': np.linspace(95, 105, len(dates)),
        'close': np.linspace(102, 112, len(dates)),
        'volume': np.random.randint(1000, 2000, len(dates)),
        'weekly_sma': np.linspace(100, 110, len(dates)),
        'weekly_momentum': np.linspace(0.2, 1.0, len(dates)),
        'weekly_volatility': np.linspace(0.01, 0.03, len(dates))
    }, index=dates)
    return data

def test_weekly_breakout_signal():
    data_weekly = create_weekly_data()
    strategy = WeeklyBreakoutStrategy()
    current_time = data_weekly.index[-1]
    signal = strategy.get_signal(data_weekly, current_time, breakout_threshold=0.01)
    assert signal in ['enter_long', 'exit_all', 'hold']

def test_weekly_momentum_signal():
    data_weekly = create_weekly_data()
    strategy = WeeklyMomentumStrategy()
    current_time = data_weekly.index[-1]
    signal = strategy.get_signal(data_weekly, current_time, momentum_threshold=0.5)
    assert signal in ['enter_long', 'exit_all', 'hold']
