# tests/test_calculators.py
import pandas as pd
import numpy as np
import pytest
from trading.calculators import (
    calculate_atr,
    calculate_dynamic_stop_and_take,
    calculate_partial_exit_targets,
    adjust_trailing_stop
)

def test_calculate_atr():
    data = pd.DataFrame({
        "high": np.linspace(110, 120, 15),
        "low": np.linspace(100, 110, 15),
        "close": np.linspace(105, 115, 15)
    })
    data = calculate_atr(data, period=14)
    assert "atr" in data.columns

def test_dynamic_stop_and_take():
    entry_price = 100
    atr = 5
    risk_params = {"atr_multiplier": 2.0, "profit_ratio": 0.05, "volatility_multiplier": 1.0}
    stop_loss, take_profit = calculate_dynamic_stop_and_take(entry_price, atr, risk_params)
    assert stop_loss < entry_price
    assert take_profit > entry_price

def test_partial_exit_targets():
    targets = calculate_partial_exit_targets(100, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)
    assert isinstance(targets, list)
    assert len(targets) == 2

def test_dynamic_stop_and_take_invalid_entry_price():
    with pytest.raises(ValueError):
        calculate_dynamic_stop_and_take(0, 5, {"atr_multiplier": 2.0, "profit_ratio": 0.05})

def test_dynamic_stop_and_take_invalid_atr():
    stop_loss, take_profit = calculate_dynamic_stop_and_take(100, 0, {"atr_multiplier": 2.0, "profit_ratio": 0.05})
    assert stop_loss == 98.00, f"Expected stop_loss 98.00, got {stop_loss}"
    assert take_profit == 105.00, f"Expected take_profit 105.00, got {take_profit}"

def test_adjust_trailing_stop_extreme():
    current_stop = 80
    current_price = 120
    highest_price = 130
    trailing_percentage = 0.05
    extreme_volatility = 10.0
    new_stop = adjust_trailing_stop(current_stop, current_price, highest_price, trailing_percentage, volatility=extreme_volatility)
    assert new_stop < current_price

def test_calculate_partial_exit_targets_invalid():
    with pytest.raises(ValueError):
        calculate_partial_exit_targets(0, partial_exit_ratio=0.5, partial_profit_ratio=0.03, final_profit_ratio=0.06)

def test_calculate_dynamic_stop_and_take_extreme_profit_ratio():
    entry_price = 100
    atr = 5
    stop_loss, take_profit = calculate_dynamic_stop_and_take(entry_price, atr, {"atr_multiplier": 2.0, "profit_ratio": 1.5})
    assert take_profit == pytest.approx(200, rel=1e-3)
    assert stop_loss == pytest.approx(90, rel=1e-3)
