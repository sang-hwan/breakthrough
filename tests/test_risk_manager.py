# tests/test_risk_manager.py
import pytest
from trading.risk_manager import RiskManager

def test_compute_position_size():
    rm = RiskManager()
    size = rm.compute_position_size(
        available_balance=10000,
        risk_percentage=0.01,
        entry_price=100,
        stop_loss=90,
        fee_rate=0.001
    )
    assert size >= 0

def test_allocate_position_splits():
    rm = RiskManager()
    allocation = rm.allocate_position_splits(total_size=1.0, splits_count=3, allocation_mode="equal")
    assert len(allocation) == 3
    assert abs(sum(allocation) - 1.0) < 1e-6

def test_compute_position_size_invalid_entry():
    rm = RiskManager()
    size = rm.compute_position_size(
        available_balance=10000,
        risk_percentage=0.01,
        entry_price=0,
        stop_loss=90,
        fee_rate=0.001
    )
    assert size == 0.0

def test_compute_position_size_invalid_stop_loss():
    rm = RiskManager()
    size = rm.compute_position_size(
        available_balance=10000,
        risk_percentage=0.01,
        entry_price=100,
        stop_loss=0,
        fee_rate=0.001
    )
    assert size == 0.0

def test_compute_position_size_extreme_volatility():
    rm = RiskManager()
    size = rm.compute_position_size(
        available_balance=10000,
        risk_percentage=0.02,
        entry_price=100,
        stop_loss=90,
        fee_rate=0.001,
        volatility=100
    )
    assert size < 0.01

def test_compute_risk_parameters_invalid_regime():
    rm = RiskManager()
    with pytest.raises(ValueError):
        rm.compute_risk_parameters_by_regime(
            base_params={"risk_per_trade": 0.01, "atr_multiplier": 2.0, "profit_ratio": 0.05},
            regime="invalid"
        )

def test_compute_risk_parameters_missing_liquidity():
    rm = RiskManager()
    with pytest.raises(ValueError):
        rm.compute_risk_parameters_by_regime(
            base_params={"risk_per_trade": 0.01, "atr_multiplier": 2.0, "profit_ratio": 0.05},
            regime="sideways"
        )
