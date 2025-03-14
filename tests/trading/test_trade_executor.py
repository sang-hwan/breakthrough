# tests/trading/test_trade_executor.py
# 이 모듈은 TradeExecutor 클래스의 기능(ATR 계산, 동적 스탑/테이크 계산, 후행 스탑 조정, 부분 청산 목표 계산)을 검증하는 테스트 케이스들을 포함합니다.

import pandas as pd  # 데이터 조작용
import pytest  # 테스트 프레임워크용
from trading.trade_executor import TradeExecutor  # 테스트할 TradeExecutor 클래스

def test_compute_atr_via_trade_executor():
    """
    TradeExecutor.compute_atr 메서드가 ATR 값을 계산하여 DataFrame에 'atr' 컬럼을 추가하는지 테스트합니다.
    """
    df = pd.DataFrame({
        "high": [110, 115, 120],
        "low": [100, 105, 110],
        "close": [105, 110, 115]
    })
    df_atr = TradeExecutor.compute_atr(df, period=2)
    assert "atr" in df_atr.columns

def test_calculate_partial_exit_targets():
    """
    TradeExecutor.calculate_partial_exit_targets 메서드가 부분 청산 목표를 계산하여 리스트 형태로 반환하는지 테스트합니다.
    """
    targets = TradeExecutor.calculate_partial_exit_targets(100, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)
    assert isinstance(targets, list)
    assert len(targets) == 2

def test_trade_executor_dynamic_stop_invalid_entry():
    """
    진입 가격이 0인 경우, TradeExecutor.calculate_dynamic_stop_and_take 메서드가 ValueError를 발생시키는지 테스트합니다.
    """
    with pytest.raises(ValueError):
        TradeExecutor.calculate_dynamic_stop_and_take(0, 5, {"atr_multiplier": 2.0, "profit_ratio": 0.05})

def test_trade_executor_dynamic_stop_invalid_atr():
    """
    ATR 값이 0인 경우에도 TradeExecutor.calculate_dynamic_stop_and_take 메서드가 기본 조정값을 사용하여 스탑로스와 테이크프로핏을 계산하는지 테스트합니다.
    """
    stop_loss, take_profit = TradeExecutor.calculate_dynamic_stop_and_take(100, 0, {"atr_multiplier": 2.0, "profit_ratio": 0.05})
    assert stop_loss == 98.00, f"Expected stop_loss 98.00, got {stop_loss}"
    assert take_profit == 104.00, f"Expected take_profit 104.00, got {take_profit}"

def test_trade_executor_adjust_trailing_stop_extreme():
    """
    TradeExecutor.adjust_trailing_stop 메서드가 극단적인 상황에서 올바른 후행 스탑 값을 계산하는지 테스트합니다.
    
    검증 조건: 계산된 후행 스탑 값은 현재 가격보다 낮아야 합니다.
    """
    current_stop = 80
    current_price = 120
    highest_price = 130
    trailing_percentage = 0.05
    new_stop = TradeExecutor.adjust_trailing_stop(current_stop, current_price, highest_price, trailing_percentage)
    assert new_stop < current_price

def test_trade_executor_partial_exit_targets_invalid():
    """
    잘못된 진입가(0)가 입력될 경우, TradeExecutor.calculate_partial_exit_targets 메서드가 ValueError를 발생시키는지 테스트합니다.
    """
    with pytest.raises(ValueError):
        TradeExecutor.calculate_partial_exit_targets(0, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)
