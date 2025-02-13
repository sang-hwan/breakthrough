# tests/test_risk_manager.py
from trading.risk_manager import RiskManager

def test_compute_position_size():
    rm = RiskManager()
    size = rm.compute_position_size(available_balance=10000, risk_percentage=0.01, entry_price=100,
                                    stop_loss=90, fee_rate=0.001)
    # 계산된 사이즈는 0보다 크거나 같아야 함.
    assert size >= 0

def test_allocate_position_splits():
    rm = RiskManager()
    allocation = rm.allocate_position_splits(total_size=1.0, splits_count=3, allocation_mode="equal")
    assert len(allocation) == 3
    assert abs(sum(allocation) - 1.0) < 1e-6
