# tests/test_calculators.py
import pytest
import pandas as pd
import numpy as np
from trading.calculators import calculate_atr, calculate_dynamic_stop_and_take, calculate_partial_exit_targets

def test_calculate_atr():
    data = pd.DataFrame({
        "high": np.linspace(110, 120, 15),
        "low": np.linspace(100, 110, 15),
        "close": np.linspace(105, 115, 15)
    })
    data = calculate_atr(data, period=14)
    # ATR 컬럼이 추가되어야 함
    assert "atr" in data.columns

def test_dynamic_stop_and_take():
    entry_price = 100
    atr = 5
    risk_params = {"atr_multiplier": 2.0, "profit_ratio": 0.05}
    stop_loss, take_profit = calculate_dynamic_stop_and_take(entry_price, atr, risk_params)
    assert stop_loss < entry_price
    assert take_profit > entry_price

def test_partial_exit_targets():
    targets = calculate_partial_exit_targets(100, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)
    assert isinstance(targets, list)
    assert len(targets) == 2
