# tests/test_core_account.py
from core.account import Account
from core.position import Position

def test_account_balance_after_trade():
    account = Account(initial_balance=10000, fee_rate=0.001)
    initial_balance = account.spot_balance
    trade = {"pnl": 500}
    account.update_after_trade(trade)
    assert account.spot_balance == initial_balance + 500

def test_position_add_and_remove():
    account = Account(initial_balance=10000)
    pos = Position(side="LONG", initial_price=100, maximum_size=1.0, total_splits=1, allocation_plan=[1.0])
    account.add_position(pos)
    assert pos in account.positions
    account.remove_position(pos)
    assert pos not in account.positions

def test_conversion_functions():
    account = Account(initial_balance=10000)
    # convert a portion to stablecoin and then back to spot
    converted = account.convert_to_stablecoin(1000)
    assert account.stablecoin_balance == converted
    new_spot = account.convert_to_spot(converted)
    # 약간의 수수료 손실이 있을 수 있음
    assert new_spot < converted
