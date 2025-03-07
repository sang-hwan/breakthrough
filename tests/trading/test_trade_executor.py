# tests/trading/test_trade_executor.py
import pandas as pd
import pytest
from trading.trade_executor import TradeExecutor

def test_compute_atr_via_trade_executor():
    df = pd.DataFrame({
        "high": [110, 115, 120],
        "low": [100, 105, 110],
        "close": [105, 110, 115]
    })
    df_atr = TradeExecutor.compute_atr(df, period=2)
    assert "atr" in df_atr.columns

def test_calculate_partial_exit_targets():
    targets = TradeExecutor.calculate_partial_exit_targets(100, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)
    assert isinstance(targets, list)
    assert len(targets) == 2

def test_trade_executor_dynamic_stop_invalid_entry():
    with pytest.raises(ValueError):
        TradeExecutor.calculate_dynamic_stop_and_take(0, 5, {"atr_multiplier": 2.0, "profit_ratio": 0.05})

def test_trade_executor_dynamic_stop_invalid_atr():
    stop_loss, take_profit = TradeExecutor.calculate_dynamic_stop_and_take(100, 0, {"atr_multiplier": 2.0, "profit_ratio": 0.05})
    assert stop_loss == 98.00, f"Expected stop_loss 98.00, got {stop_loss}"
    assert take_profit == 104.00, f"Expected take_profit 104.00, got {take_profit}"

def test_trade_executor_adjust_trailing_stop_extreme():
    current_stop = 80
    current_price = 120
    highest_price = 130
    trailing_percentage = 0.05
    new_stop = TradeExecutor.adjust_trailing_stop(current_stop, current_price, highest_price, trailing_percentage)
    assert new_stop < current_price

def test_trade_executor_partial_exit_targets_invalid():
    with pytest.raises(ValueError):
        TradeExecutor.calculate_partial_exit_targets(0, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)
