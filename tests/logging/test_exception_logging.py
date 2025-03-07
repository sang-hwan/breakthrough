# tests/logging/test_exception_logging.py
import logging
import pytest
from trading.calculators import calculate_dynamic_stop_and_take, adjust_trailing_stop

def test_exception_logging_dynamic_stop_and_take(caplog):
    caplog.set_level(logging.ERROR)
    # entry_price가 0인 경우 ValueError가 발생하도록 되어 있음.
    with pytest.raises(ValueError):
        calculate_dynamic_stop_and_take(0, 5, {"atr_multiplier": 2.0, "profit_ratio": 0.05})
    assert "Invalid entry_price" in caplog.text

def test_exception_logging_adjust_trailing_stop(caplog):
    caplog.set_level(logging.ERROR)
    # current_price 또는 highest_price가 0 이하인 경우 에러 발생
    with pytest.raises(ValueError):
        adjust_trailing_stop(0, -100, -100, 0.05)
    assert "Invalid current_price" in caplog.text or "highest_price" in caplog.text
