# core/position.py
import uuid
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class Position:
    def __init__(self, side: str = "LONG", initial_price: float = None, maximum_size: float = 0.0,
                 total_splits: int = 1, allocation_plan: list = None):
        """
        포지션 생성:
          - side: 거래 방향 ("LONG" 또는 "SHORT")
          - initial_price: 진입 가격
          - maximum_size: 최대 포지션 사이즈
          - total_splits: 분할 진입 횟수
          - allocation_plan: 각 분할 진입 비율 (미지정 시 빈 리스트)
        """
        self.position_id = str(uuid.uuid4())
        self.side = side
        self.executions = []  # 실행 내역 리스트
        self.initial_price = initial_price
        self.maximum_size = maximum_size
        self.total_splits = total_splits
        self.executed_splits = 0
        self.allocation_plan = allocation_plan if allocation_plan is not None else []
        self.highest_price = initial_price if initial_price is not None else 0.0
        logger.debug(f"새 포지션 생성: ID={self.position_id}, side={self.side}, 초기 가격={self.initial_price}")

    def add_execution(self, entry_price: float, size: float, stop_loss: float = None,
                      take_profit: float = None, entry_time = None, exit_targets: list = None,
                      trade_type: str = "unknown", min_order_size: float = 1e-8) -> None:
        """
        포지션 실행 추가:
          - exit_targets: [(target_price, exit_ratio), ...] 형식의 리스트
          - min_order_size: 최소 체결 수량
        """
        if size < min_order_size:
            logger.debug("체결 수량이 최소 주문 수량보다 작아 실행 건너뜀.")
            return
        flagged_targets = []
        if exit_targets:
            for target_price, exit_ratio in exit_targets:
                flagged_targets.append({
                    'price': target_price,
                    'exit_ratio': exit_ratio,
                    'hit': False
                })
        execution = {
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'exit_targets': flagged_targets,
            'trade_type': trade_type,
            'highest_price_since_entry': entry_price,
            'closed': False
        }
        self.executions.append(execution)
        logger.debug(f"실행 추가됨: entry_price={entry_price}, size={size}, trade_type={trade_type}")

    def get_total_size(self) -> float:
        """
        미체결 실행의 총 사이즈를 반환합니다.
        """
        return sum(exec_record['size'] for exec_record in self.executions if not exec_record.get("closed", False))

    def get_average_entry_price(self) -> float:
        """
        미체결 실행의 평균 진입 가격을 계산하여 반환합니다.
        """
        total_cost = sum(exec_record['entry_price'] * exec_record['size'] for exec_record in self.executions if not exec_record.get("closed", False))
        total_qty = self.get_total_size()
        return (total_cost / total_qty) if total_qty > 0 else 0.0

    def remove_execution(self, index: int) -> None:
        """
        지정한 인덱스의 실행 내역을 제거합니다.
        """
        if 0 <= index < len(self.executions):
            self.executions.pop(index)
            logger.debug(f"실행 제거됨: index={index}")

    def is_empty(self) -> bool:
        """
        모든 실행이 종료(closed)되었는지 확인합니다.
        """
        return all(exec_record.get("closed", False) for exec_record in self.executions)

    def partial_close_execution(self, index: int, close_ratio: float, min_order_size: float = 1e-8) -> float:
        """
        부분 청산:
          - 지정된 비율(close_ratio)만큼 실행의 사이즈를 감소시키며,
            남은 수량이 최소 주문 수량보다 작으면 해당 실행을 종료(closed 처리)합니다.
        Returns:
          - 청산된 수량
        """
        if 0 <= index < len(self.executions):
            exec_record = self.executions[index]
            qty_to_close = exec_record['size'] * close_ratio
            exec_record['size'] -= qty_to_close
            if exec_record.get('exit_targets'):
                exec_record['exit_targets'] = [t for t in exec_record['exit_targets'] if not t.get('hit', False)]
            if exec_record['size'] < min_order_size:
                exec_record['closed'] = True
            logger.debug(f"부분 청산 실행: index={index}, close_ratio={close_ratio}, 청산 수량={qty_to_close}")
            return qty_to_close
        return 0.0
