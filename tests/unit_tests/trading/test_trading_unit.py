# tests/unit_tests/trading/test_trading_unit.py
import pytest
from trading.position_management import Account, Position, AssetManager
from trading.risk_management import RiskManager
from trading.trade_decision import TradeDecision

@pytest.fixture
def setup_trading_unit():
    """
    Trading 모듈 개별 구성요소 테스트를 위한 fixture.
    """
    account = Account(initial_balance=10000)
    position = Position(
        side="LONG",
        initial_price=100,
        maximum_size=10,
        total_splits=3,
        allocation_plan=[0.3, 0.3, 0.4]
    )
    asset_manager = AssetManager(account)
    risk_manager = RiskManager()
    trade_decision = TradeDecision()
    return account, position, asset_manager, risk_manager, trade_decision

def test_account_initialization(setup_trading_unit):
    account, _, _, _, _ = setup_trading_unit
    assert account.spot_balance == 10000
    assert account.stablecoin_balance == 0.0

def test_add_and_remove_position(setup_trading_unit):
    account, position, _, _, _ = setup_trading_unit
    account.add_position(position)
    assert len(account.positions) == 1
    account.remove_position(position)
    assert len(account.positions) == 0

def test_get_used_and_available_balance(setup_trading_unit):
    account, position, _, _, _ = setup_trading_unit
    # 포지션에 거래 실행 기록 추가
    position.add_execution(entry_price=100, size=1)
    account.add_position(position)
    used_balance = account.get_used_balance()
    available_balance = account.get_available_balance()
    assert used_balance > 0
    assert available_balance >= 0

def test_compute_position_size(setup_trading_unit):
    _, _, _, risk_manager, _ = setup_trading_unit
    size = risk_manager.compute_position_size(
        available_balance=10000, risk_percentage=0.02, entry_price=100, stop_loss=95
    )
    assert size > 0

def test_trade_decision(setup_trading_unit):
    account, _, _, _, trade_decision = setup_trading_unit
    decision = trade_decision.decide_trade("buy", 105, account)
    assert decision in ["buy", "sell", "hold"]
