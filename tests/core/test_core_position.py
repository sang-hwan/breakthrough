# tests/core/test_core_position.py

import pytest
# Position 클래스를 임포트합니다.
from core.position import Position

def test_add_execution_and_totals():
    """
    포지션에 거래 실행(execution)을 추가하고, 총 거래 수량 및 평균 진입 가격이 올바르게 계산되는지 테스트합니다.
    
    - 포지션 객체를 생성한 후, 거래 실행을 추가합니다.
    - 전체 거래 수량과 평균 진입 가격이 예상값(1.0, 100)이 맞는지 확인합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    pos = Position(side="LONG", initial_price=100, maximum_size=1.0, total_splits=1, allocation_plan=[1.0])
    # entry_price=100, size=1.0, entry_time="2023-01-01"로 거래 실행 추가
    pos.add_execution(entry_price=100, size=1.0, entry_time="2023-01-01")
    # 총 수량이 1.0이어야 함
    assert pos.get_total_size() == 1.0
    # 평균 진입 가격이 100이어야 함
    avg_price = pos.get_average_entry_price()
    assert avg_price == 100

def test_partial_close_execution():
    """
    포지션에서 부분 청산(partial close)이 올바르게 이루어지는지 테스트합니다.
    
    - 먼저 포지션에 거래 실행을 추가한 후, 일부 수량(0.5)만 청산합니다.
    - 청산된 수량과 남은 수량이 예상대로 반영되는지 확인합니다.
    
    Parameters:
        없음
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    pos = Position(side="LONG", initial_price=100, maximum_size=1.0, total_splits=1, allocation_plan=[1.0])
    # 거래 실행 추가
    pos.add_execution(entry_price=100, size=1.0, entry_time="2023-01-01")
    # 포지션의 0번 거래 실행에서 0.5 수량만 청산하도록 호출
    closed_qty = pos.partial_close_execution(0, 0.5)
    # 청산된 수량이 0.5인지 확인
    assert closed_qty == 0.5
    # 남은 수량은 0.5여야 함 (소수점 오차 고려하여 비교)
    remaining = pos.get_total_size()
    assert pytest.approx(remaining, rel=1e-3) == 0.5
