# tests/test_handle_weekly_end.py
import pandas as pd
import pytest
from backtesting.backtester import Backtester
from trading.positions import TradePosition

@pytest.fixture
def dummy_backtester():
    # 간단한 백테스터 인스턴스 생성 (실제 DB 로드나 인디케이터 계산 없이 사용)
    bt = Backtester(symbol="TEST/USDT", account_size=10000, fee_rate=0.001, slippage_rate=0.001)
    
    # Dummy 포지션 생성 (포지션에 미체결된 실행 내역 추가)
    pos = TradePosition(side="LONG", initial_price=100, maximum_size=1.0, total_splits=1, allocation_plan=[1.0])
    pos.add_execution(
        entry_price=100,
        size=1.0,
        stop_loss=95,
        take_profit=110,
        entry_time=pd.Timestamp("2023-01-06 10:00:00"),
        trade_type="new_entry"
    )
    bt.positions.append(pos)
    return bt

def test_handle_weekly_end(dummy_backtester):
    # 청산 전 상태: 포지션 존재, trade_logs 비어있음
    bt = dummy_backtester
    assert len(bt.positions) == 1
    assert len(bt.trade_logs) == 0

    # 주간 종료 시 호출 (예: 금요일 데이터)
    current_time = pd.Timestamp("2023-01-13 16:00:00")  # 금요일 가정
    # 청산에 사용될 dummy row (close 가격 제공)
    row = pd.Series({"close": 105})
    
    # handle_weekly_end() 호출
    bt.handle_weekly_end(current_time, row)
    
    # 모든 포지션이 청산되어 positions 리스트가 비어야 함
    assert len(bt.positions) == 0
    # 거래 내역이 기록되어야 함 (포지션당 최소 1건)
    assert len(bt.trade_logs) >= 1
    # 계좌의 잔고가 업데이트되었는지(예: 초기 잔고 대비 변동) 확인 (단순 비교)
    assert bt.account.get_available_balance() != 10000
