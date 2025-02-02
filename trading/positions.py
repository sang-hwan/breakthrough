# trading/positions.py

import uuid

class TradePosition:
    """
    거래 포지션 관리 클래스.
    
    Attributes:
      - position_id: 포지션에 대한 고유 식별자 (UUID 문자열)
      - side: "LONG" 혹은 "SHORT"
      - executions: 개별 체결 내역 리스트 (각 execution은 entry_price, size 등 기록)
      - initial_price: 첫 체결 기준 가격
      - maximum_size: 포지션의 총 목표 수량
      - total_splits: 계획된 분할 매수 횟수
      - executed_splits: 현재까지 체결된 분할 횟수
      - allocation_plan: 각 분할 매수 비중 리스트 (예: [0.3, 0.3, 0.4])
    """
    def __init__(
        self,
        side: str = "LONG",
        initial_price: float = None,
        maximum_size: float = 0.0,
        total_splits: int = 1,
        allocation_plan: list = None
    ):
        self.position_id = str(uuid.uuid4())  # 포지션 고유 식별자 생성
        self.side = side
        self.executions = []  # 각 체결 내역을 저장
        self.initial_price = initial_price
        self.maximum_size = maximum_size
        self.total_splits = total_splits
        self.executed_splits = 0
        self.allocation_plan = allocation_plan if allocation_plan is not None else []

    def add_execution(
        self,
        entry_price: float,
        size: float,
        stop_loss: float = None,
        take_profit: float = None,
        entry_time = None,
        exit_targets: list = None,
        trade_type: str = "unknown"  # 신규 진입("new_entry") 또는 분할 매수("scale_in") 구분
    ):
        """
        하나의 체결(실제 매수)를 기록.
        exit_targets: [(target_price, exit_ratio), ...]
                      예) 첫 목표가 도달 시 50% 청산, 이후 전량 청산
        trade_type: 체결의 유형. 신규 진입("new_entry") 또는 분할 매수("scale_in") 등
        """
        exit_targets_flagged = []
        if exit_targets is not None:
            for target_price, exit_ratio in exit_targets:
                exit_targets_flagged.append({
                    'price': target_price,
                    'exit_ratio': exit_ratio,
                    'hit': False
                })
        self.executions.append({
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'exit_targets': exit_targets_flagged,
            'trade_type': trade_type
        })
        
    def get_total_size(self) -> float:
        """현재까지 체결된 총 수량"""
        return sum(exec_record['size'] for exec_record in self.executions)

    def get_average_entry_price(self) -> float:
        """가중평균 매수가 계산"""
        total_cost = sum(exec_record['entry_price'] * exec_record['size'] for exec_record in self.executions)
        total_qty = self.get_total_size()
        return (total_cost / total_qty) if total_qty > 0 else 0.0

    def remove_execution(self, index: int):
        """지정된 인덱스의 체결 내역을 제거 (전량 청산)"""
        if 0 <= index < len(self.executions):
            self.executions.pop(index)

    def is_empty(self) -> bool:
        """포지션에 체결 내역이 없으면 True"""
        return len(self.executions) == 0

    def partial_close_execution(self, index: int, close_ratio: float) -> float:
        """
        지정된 체결 내역의 일정 비율(close_ratio)만 청산.
        반환: 청산된 수량
        """
        if 0 <= index < len(self.executions):
            exec_record = self.executions[index]
            qty_to_close = exec_record['size'] * close_ratio
            exec_record['size'] -= qty_to_close
            return qty_to_close
        return 0.0
