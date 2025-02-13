# tests/test_core_position.py
import pytest
from core.position import Position

def test_add_execution_and_totals():
    pos = Position(side="LONG", initial_price=100, maximum_size=1.0, total_splits=1, allocation_plan=[1.0])
    pos.add_execution(entry_price=100, size=1.0, entry_time="2023-01-01")
    assert pos.get_total_size() == 1.0
    avg_price = pos.get_average_entry_price()
    assert avg_price == 100

def test_partial_close_execution():
    pos = Position(side="LONG", initial_price=100, maximum_size=1.0, total_splits=1, allocation_plan=[1.0])
    pos.add_execution(entry_price=100, size=1.0, entry_time="2023-01-01")
    closed_qty = pos.partial_close_execution(0, 0.5)
    assert closed_qty == 0.5
    # 남은 수량이 0.5여야 함.
    remaining = pos.get_total_size()
    assert pytest.approx(remaining, rel=1e-3) == 0.5
