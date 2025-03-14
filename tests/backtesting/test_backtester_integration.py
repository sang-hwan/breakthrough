# tests/backtesting/test_backtester_integration.py

import pandas as pd
import numpy as np
import pytest
from backtesting.backtester import Backtester

@pytest.fixture
def dummy_data():
    """
    통합 테스트용 더미 데이터를 생성하는 fixture.
    
    - 30일치 long 데이터 생성 (날짜 범위: 2023-01-01 ~)
    - 생성된 long 데이터를 시간해상도로 리샘플링하여 short 데이터를 생성
     
    Returns:
        tuple: (df_long, df_short) 두 개의 DataFrame을 반환
    """
    # 30일치 long 데이터 생성
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    df_long = pd.DataFrame({
        "open": np.linspace(100, 130, 30),
        "high": np.linspace(105, 135, 30),
        "low": np.linspace(95, 125, 30),
        "close": np.linspace(100, 130, 30),
    }, index=dates)
    # long 데이터를 시간별(h)로 리샘플링하여 short 데이터 생성 (앞의 값 채움)
    df_short = df_long.resample('h').ffill()
    return df_long, df_short

def test_backtester_integration(dummy_data):
    """
    Backtester 클래스의 통합 테스트를 수행합니다.
    
    - dummy_data를 사용하여 백테스터의 long, short, train 데이터를 할당합니다.
    - 외부 의존성을 제거하기 위해 HMM, 지표, 위험 관리, 시그널 결정 등의 메서드를 dummy lambda 함수로 오버라이드합니다.
    - 또한 더미 AssetManager를 생성하여 rebalance 함수는 아무 작업도 하지 않도록 합니다.
    - 최종적으로 백테스트를 실행한 후, trades와 trade_logs가 리스트로 반환되는지 확인합니다.
    
    Parameters:
        dummy_data (tuple): dummy_data fixture로부터 받은 (df_long, df_short)
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    df_long, df_short = dummy_data
    # 백테스터 객체 생성, 초기 계좌 사이즈 10,000달러, 심볼은 "BTC/USDT"
    backtester = Backtester(symbol="BTC/USDT", account_size=10000)
    
    # 더미 데이터를 직접 백테스터 객체의 속성에 할당
    backtester.df_long = df_long.copy()
    backtester.df_short = df_short.copy()
    backtester.df_train = df_short.copy()
    backtester.last_signal_time = None
    backtester.last_rebalance_time = None
    backtester.positions = []
    
    # 외부 의존성을 제거하기 위해 필요한 메서드들을 dummy 함수(lambda)로 오버라이드
    backtester.update_hmm_regime = lambda dynamic_params: pd.Series(["bullish"] * len(backtester.df_long), index=backtester.df_long.index)
    backtester.apply_indicators = lambda: None
    backtester.update_short_dataframe = lambda regime_series, dynamic_params: None
    backtester.risk_manager.compute_risk_parameters_by_regime = lambda base_params, regime, liquidity="high": base_params
    backtester.ensemble_manager.get_final_signal = lambda regime, liquidity, data, current_time, data_weekly=None: "hold"
    # DummyAssetManager: rebalance 메서드는 아무 작업도 수행하지 않음
    backtester.asset_manager = type("DummyAssetManager", (), {"rebalance": lambda self, regime: None})()
    # 주간 데이터를 생성: long 데이터에서 주간 마지막 값을 추출하여 df_weekly 생성
    backtester.df_weekly = df_long.resample('W').last()
    
    # 백테스트 실행: dynamic_params에 시그널 쿨다운과 리밸런싱 간격을 설정
    trades, trade_logs = backtester.run_backtest(dynamic_params={"signal_cooldown_minutes": 5, "rebalance_interval_minutes": 60})
    # trades와 trade_logs가 리스트 타입인지 확인
    assert isinstance(trades, list)
    assert isinstance(trade_logs, list)
