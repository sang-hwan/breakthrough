# tests/core/test_core_account.py

# Account 클래스와 Position 클래스를 임포트합니다.
from asset_position.account import Account
from asset_position.position import Position

def test_account_balance_after_trade():
    """
    거래 후 계좌 잔고가 올바르게 업데이트되는지 테스트합니다.
    
    - 초기 계좌 잔고와 수수료율을 설정한 후, 거래(PnL +500)를 반영하여 잔고가 증가했는지 확인합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    # 초기 잔고 10,000달러, 수수료율 0.1%로 Account 객체 생성
    account = Account(initial_balance=10000, fee_rate=0.001)
    initial_balance = account.spot_balance
    # 거래 정보: pnl(이익) +500
    trade = {"pnl": 500}
    # 거래 후 계좌 잔고 업데이트
    account.update_after_trade(trade)
    # 업데이트된 현물 잔고가 초기 잔고에 +500 되어야 함
    assert account.spot_balance == initial_balance + 500

def test_position_add_and_remove():
    """
    포지션(Position)을 계좌에 추가하고 제거하는 기능을 테스트합니다.
    
    - Position 객체를 생성하여 계좌에 추가한 후 포함 여부를 확인하고,
      이후 제거한 후 삭제되었음을 확인합니다.
      
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    account = Account(initial_balance=10000)
    # 포지션 객체 생성 (롱 포지션, 진입 가격 100, 최대 사이즈 1.0 등)
    pos = Position(side="LONG", initial_price=100, maximum_size=1.0, total_splits=1, allocation_plan=[1.0])
    account.add_position(pos)
    # 추가된 포지션이 account.positions에 존재하는지 확인
    assert pos in account.positions
    # 포지션 제거 후 존재하지 않는지 확인
    account.remove_position(pos)
    assert pos not in account.positions

def test_conversion_functions():
    """
    Account 객체의 자산 전환 함수 (convert_to_stablecoin, convert_to_spot)의 동작을 테스트합니다.
    
    - 일정 금액을 스테이블코인으로 전환한 후, 다시 현물로 전환하여
      수수료 등으로 인해 약간 손실이 발생하는지 확인합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    account = Account(initial_balance=10000)
    # 1000달러를 스테이블코인으로 전환
    converted = account.convert_to_stablecoin(1000)
    # 스테이블코인 잔고가 전환된 값과 일치해야 함
    assert account.stablecoin_balance == converted
    # 스테이블코인을 다시 현물로 전환 (수수료로 인해 손실 발생 가능)
    new_spot = account.convert_to_spot(converted)
    # 전환 후 현물 금액이 원래 스테이블코인 금액보다 작아야 함
    assert new_spot < converted
