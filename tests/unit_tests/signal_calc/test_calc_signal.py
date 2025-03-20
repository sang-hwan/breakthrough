# tests/unit_tests/signal_calc/test_calc_signal.py
import pandas as pd
import pytest
from datetime import datetime

from signal_calc.calc_signal import compute_dynamic_weights, Ensemble

def test_compute_dynamic_weights_default():
    """
    기본 매개변수(시장 변동성 없음, volume 없음)일 때 동적 가중치 계산을 검증합니다.
    """
    short_weight, weekly_weight = compute_dynamic_weights(
        market_volatility=None, liquidity_info="high", volume=None
    )
    assert 0 <= short_weight <= 1
    assert pytest.approx(weekly_weight, 0.001) == 1 - short_weight
    
def test_ensemble_get_final_signal_bullish():
    """
    Ensemble 클래스가 bullish 시장에서 올바른 최종 신호를 도출하는지 검증합니다.
    """
    # 단기 가상의 OHLC 및 지표 데이터 생성
    dates = pd.date_range(start="2025-03-01", periods=10, freq="D")
    dummy_data = pd.DataFrame({
        "open": [100 + i for i in range(10)],
        "high": [105 + i for i in range(10)],
        "low": [95 + i for i in range(10)],
        "close": [102 + i for i in range(10)],
        "sma": [101 + i for i in range(10)],
        "rsi": [30 + i for i in range(10)],
        "bb_lband": [98 + i for i in range(10)],
        "bb_hband": [107 + i for i in range(10)]
    }, index=dates)

    # 주간 데이터 생성 (단순 resample 예시)
    dummy_weekly = dummy_data.resample('W').last()
    
    ensemble = Ensemble()
    current_time = dates[-1]
    final_signal = ensemble.get_final_signal(
        market_regime="bullish",
        liquidity_info="high",
        data=dummy_data,
        current_time=current_time,
        data_weekly=dummy_weekly,
        market_volatility=0.03,
        volume=1500
    )
    # 최종 신호가 세 가지 값 중 하나여야 함
    assert final_signal in ["enter_long", "exit_all", "hold"]