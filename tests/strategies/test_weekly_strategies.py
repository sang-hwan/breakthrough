# tests/strategies/test_weekly_strategies.py
# 이 모듈은 주간 집계 데이터를 기반으로 하는 주간 전략(돌파, 모멘텀)들의 신호 생성 기능을 테스트합니다.

import pandas as pd  # 데이터 조작용 패키지
import numpy as np  # 수치 계산용 패키지
from strategies.trading_strategies import WeeklyBreakoutStrategy, WeeklyMomentumStrategy  # 주간 전략 클래스들

def create_weekly_data():
    """
    4주간의 주간 데이터를 생성하는 함수입니다.
    
    각 주에 대해 open, high, low, close, volume, 주간 SMA, 모멘텀, 변동성을 포함한 데이터를 생성합니다.
    
    Returns:
        pd.DataFrame: 주간 데이터, 인덱스는 각 주의 시작일 (월요일).
    """
    dates = pd.date_range(start='2020-01-06', periods=4, freq='W-MON')  # 4주간의 월요일 날짜 생성
    data = pd.DataFrame({
        'open': np.linspace(100, 110, len(dates)),  # 주간 시가
        'weekly_high': np.linspace(105, 115, len(dates)),  # 주간 최고가
        'weekly_low': np.linspace(95, 105, len(dates)),  # 주간 최저가
        'close': np.linspace(102, 112, len(dates)),  # 주간 종가
        'volume': np.random.randint(1000, 2000, len(dates)),  # 임의의 거래량
        'weekly_sma': np.linspace(100, 110, len(dates)),  # 주간 단순 이동 평균
        'weekly_momentum': np.linspace(0.2, 1.0, len(dates)),  # 주간 모멘텀
        'weekly_volatility': np.linspace(0.01, 0.03, len(dates))  # 주간 변동성
    }, index=dates)
    return data

def test_weekly_breakout_signal():
    """
    WeeklyBreakoutStrategy의 신호 생성 기능을 테스트합니다.
    
    생성된 주간 데이터에 대해 돌파 임계값(예: 1%) 조건을 적용하여 반환된 신호가 유효한지 확인합니다.
    """
    data_weekly = create_weekly_data()  # 주간 데이터 생성
    strategy = WeeklyBreakoutStrategy()  # 주간 돌파 전략 객체 생성
    current_time = data_weekly.index[-1]  # 마지막 주 데이터를 현재 시각으로 사용
    signal = strategy.get_signal(data_weekly, current_time, breakout_threshold=0.01)
    assert signal in ['enter_long', 'exit_all', 'hold']

def test_weekly_momentum_signal():
    """
    WeeklyMomentumStrategy의 신호 생성 기능을 테스트합니다.
    
    생성된 주간 데이터에 대해 모멘텀 임계값(예: 0.5)을 적용하여 반환된 신호가 올바른지 검증합니다.
    """
    data_weekly = create_weekly_data()  # 주간 데이터 생성
    strategy = WeeklyMomentumStrategy()  # 주간 모멘텀 전략 객체 생성
    current_time = data_weekly.index[-1]
    signal = strategy.get_signal(data_weekly, current_time, momentum_threshold=0.5)
    assert signal in ['enter_long', 'exit_all', 'hold']
