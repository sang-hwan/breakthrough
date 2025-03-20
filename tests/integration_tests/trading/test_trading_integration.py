# tests/integration_tests/trading/test_trading_integration.py
import pytest
from trading.position_management import Account, Position, AssetManager
from trading.risk_management import RiskManager
from trading.trade_decision import TradeDecision

@pytest.fixture
def setup_trading_integration():
    """
    Trading 모듈 전체 흐름 통합 테스트를 위한 fixture.
    """
    account = Account(initial_balance=10000)
    position = Position(
        side="LONG",
        initial_price=100,
        maximum_size=10,
        total_splits=2,
        allocation_plan=[0.5, 0.5]
    )
    asset_manager = AssetManager(account)
    risk_manager = RiskManager()
    trade_decision = TradeDecision()
    account.add_position(position)
    return account, position, asset_manager, risk_manager, trade_decision

def test_trading_flow(setup_trading_integration):
    """
    거래 흐름 시나리오 통합 테스트:
    1. 포지션 사이즈 계산
    2. 거래 판단
    3. 거래 실행 (포지션 실행 기록 추가)
    4. 거래 후 계좌 잔고 업데이트
    5. 자산 리밸런싱 실행
    """
    account, position, asset_manager, risk_manager, trade_decision = setup_trading_integration

    # 1. 포지션 사이즈 계산
    size = risk_manager.compute_position_size(
        available_balance=account.get_available_balance(),
        risk_percentage=0.02,
        entry_price=100,
        stop_loss=95
    )
    assert size > 0

    # 2. 거래 판단
    decision = trade_decision.decide_trade("buy", 105, account)
    assert decision in ["buy", "sell", "hold"]

    # 3. 거래 실행: 포지션에 거래 실행 기록 추가
    position.add_execution(entry_price=105, size=size)

    # 4. 거래 체결 후 계좌 잔고 업데이트
    trade_details = {"pnl": 50}
    account.update_after_trade(trade_details)
    # 계좌의 현물 잔고는 초기 잔고에 pnl이 반영되어야 합니다.
    assert account.spot_balance == 10000 + 50

    # 5. 자산 리밸런싱 실행 (시장 상태: bullish)
    asset_manager.rebalance("bullish")
    # 리밸런싱 후 사용 가능한 잔고가 음수가 되지 않아야 함
    assert account.get_available_balance() >= 0
