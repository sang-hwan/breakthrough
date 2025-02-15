# tests/test_backtester_integration.py
import pandas as pd
import numpy as np
import pytest
from backtesting.backtester import Backtester

@pytest.fixture
def dummy_data():
    # 30일치 더미 long 데이터 생성
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    df_long = pd.DataFrame({
        "open": np.linspace(100, 130, 30),
        "high": np.linspace(105, 135, 30),
        "low": np.linspace(95, 125, 30),
        "close": np.linspace(100, 130, 30),
    }, index=dates)
    # short 데이터: long 데이터를 시간해상도로 리샘플링하여 생성
    df_short = df_long.resample('h').ffill()
    return df_long, df_short

def test_backtester_integration(dummy_data):
    df_long, df_short = dummy_data
    backtester = Backtester(symbol="BTC/USDT", account_size=10000)
    # 더미 데이터를 직접 할당
    backtester.df_long = df_long.copy()
    backtester.df_short = df_short.copy()
    backtester.df_short["market_regime"] = "bullish"
    backtester.df_train = df_short.copy()
    backtester.last_signal_time = None
    backtester.last_rebalance_time = None
    backtester.positions = []
    
    # 외부 의존성을 제거하기 위해 필요한 메서드 오버라이드
    backtester.update_hmm_regime = lambda dynamic_params: pd.Series(["bullish"] * len(backtester.df_long), index=backtester.df_long.index)
    backtester.apply_indicators = lambda: None
    backtester.update_short_dataframe = lambda regime_series, dynamic_params: None
    backtester.risk_manager.compute_risk_parameters_by_regime = lambda base_params, regime, liquidity="high": base_params
    backtester.ensemble_manager.get_final_signal = lambda regime, liquidity, data, current_time, data_weekly=None: "hold"
    # 더미 AssetManager (rebalance 메서드 no-op)
    backtester.asset_manager = type("DummyAssetManager", (), {"rebalance": lambda self, regime: None})()
    # 주간 데이터도 생성 (단순하게 long 데이터의 주간 마지막값)
    backtester.df_weekly = df_long.resample('W').last()
    
    # 백테스트 실행 (오류 없이 리스트 반환되어야 함)
    trades, trade_logs = backtester.run_backtest(dynamic_params={"signal_cooldown_minutes": 5, "rebalance_interval_minutes": 60})
    assert isinstance(trades, list)
    assert isinstance(trade_logs, list)
