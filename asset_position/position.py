# core/position.py

import uuid  # 포지션 식별자 생성을 위한 uuid 모듈 임포트
from logging.logger_config import setup_logger  # 로그 설정 함수를 임포트하여 로깅 설정

# 모듈 이름(__name__)을 기반으로 로거 인스턴스 생성
logger = setup_logger(__name__)  # 전역 로거: 포지션 관련 이벤트 로깅에 사용됨


class Position:
    """
    Position 클래스는 개별 거래 포지션을 관리합니다.
    
    Attributes:
        position_id (str): 포지션의 고유 식별자.
        side (str): 포지션의 방향 ("LONG" 또는 "SHORT").
        executions (list): 해당 포지션에서 발생한 거래 실행 기록 목록.
        initial_price (float): 포지션이 시작될 때의 진입 가격.
        maximum_size (float): 포지션의 최대 허용 크기.
        total_splits (int): 포지션 진입 시 계획된 분할 수.
        executed_splits (int): 실제 실행된 분할 수.
        allocation_plan (list): 자금 배분 계획.
        highest_price (float) 또는 lowest_price (float): 포지션의 진행 상황에 따른 최고 또는 최저 가격.
    """
    
    def __init__(self, side: str = "LONG", initial_price: float = None, maximum_size: float = 0.0,
                 total_splits: int = 1, allocation_plan: list = None) -> None:
        """
        새로운 거래 포지션을 초기화합니다.
        
        Parameters:
            side (str): 거래 방향 ("LONG"은 매수, "SHORT"는 매도). 기본값은 "LONG".
            initial_price (float): 포지션 진입 시 가격. 반드시 양수여야 함.
            maximum_size (float): 포지션이 가질 수 있는 최대 크기.
            total_splits (int): 포지션을 분할해 진입할 총 횟수.
            allocation_plan (list): 자금 배분 전략을 담은 리스트 (선택 사항).
            
        Raises:
            ValueError: initial_price가 제공되지 않았거나 0 이하인 경우.
        """
        if initial_price is None or initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        # 고유 포지션 식별자 생성 (UUID 사용)
        self.position_id: str = str(uuid.uuid4())
        # 거래 방향을 대문자로 변환하여 저장 (예: "LONG", "SHORT")
        self.side: str = side.upper()
        # 거래 실행 기록을 저장할 리스트 초기화
        self.executions: list = []
        self.initial_price: float = initial_price
        self.maximum_size: float = maximum_size
        self.total_splits: int = total_splits
        self.executed_splits: int = 0
        # 배분 계획이 제공되지 않으면 빈 리스트로 초기화
        self.allocation_plan: list = allocation_plan if allocation_plan is not None else []
        # LONG 포지션은 진입 후 최고가를, SHORT 포지션은 최저가를 추적
        if self.side == "SHORT":
            self.lowest_price: float = initial_price
        else:
            self.highest_price: float = initial_price
        logger.debug(f"New position created: ID={self.position_id}, side={self.side}, entry price={self.initial_price}")

    def add_execution(self, entry_price: float, size: float, stop_loss: float = None,
                      take_profit: float = None, entry_time=None, exit_targets: list = None,
                      trade_type: str = "unknown", min_order_size: float = 1e-8) -> None:
        """
        포지션에 새로운 거래 실행 기록을 추가합니다.
        
        각 실행은 포지션 내에서 개별 거래를 나타내며, 진입 가격, 거래 수량, 
        손절/익절 가격, 거래 유형 등 다양한 거래 정보를 포함합니다.
        
        Parameters:
            entry_price (float): 거래 실행 시의 진입 가격.
            size (float): 거래 수량.
            stop_loss (float): (선택) 손절 가격.
            take_profit (float): (선택) 익절 가격.
            entry_time: (선택) 거래 실행 시각.
            exit_targets (list): (선택) 부분 청산 목표를 나타내는 (목표 가격, 청산 비율) 튜플의 리스트.
            trade_type (str): 거래 유형을 설명하는 문자열 (기본값 "unknown").
            min_order_size (float): 최소 허용 거래 수량 (거래 실행 검증용, 기본값 매우 작은 수).
            
        Returns:
            None
        
        Notes:
            - 거래 수량이 최소 주문 크기보다 작으면 실행이 추가되지 않습니다.
            - exit_targets는 리스트 형식이어야 하며, 각 목표는 딕셔너리 형태로 변환되어 저장됩니다.
            - LONG 포지션은 진입 이후 최고가, SHORT 포지션은 최저가를 추적합니다.
        """
        # 최소 주문 수량 이하일 경우 실행 추가하지 않고 경고 로그 출력
        if size < min_order_size:
            logger.warning("Execution size below minimum order size; execution not added.")
            return
        # exit_targets가 제공되었을 때 리스트 타입이 아니면 오류 로그 출력 후 종료
        if exit_targets and not isinstance(exit_targets, list):
            logger.error("exit_targets must be a list.", exc_info=True)
            return

        targets = []
        # exit_targets가 존재하면, 각 목표 가격과 청산 비율을 포함하는 딕셔너리 생성
        if exit_targets:
            for target_price, exit_ratio in exit_targets:
                targets.append({'price': target_price, 'exit_ratio': exit_ratio, 'hit': False})
        # 실행 기록을 딕셔너리 형태로 생성하여 필요한 모든 정보를 포함
        execution = {
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': entry_time,
            'exit_targets': targets,
            'trade_type': trade_type,
            'closed': False
        }
        # 포지션 방향에 따라 실행 후 추적할 극단값(최고가 또는 최저가) 초기화
        if self.side == "SHORT":
            execution["lowest_price_since_entry"] = entry_price
        else:
            execution["highest_price_since_entry"] = entry_price

        # 생성된 실행 기록을 포지션의 executions 리스트에 추가
        self.executions.append(execution)
        logger.debug(f"Execution added: entry_price={entry_price}, size={size}, type={trade_type}")

    def update_extremum(self, current_price: float) -> None:
        """
        모든 열려 있는 거래 실행에 대해, 현재 시장 가격을 반영하여 최고가 또는 최저가 값을 업데이트합니다.
        
        Parameters:
            current_price (float): 최신 시장 가격.
            
        Returns:
            None
        
        Notes:
            - LONG 포지션의 경우, 현재 가격이 이전 최고가보다 높으면 최고가를 갱신합니다.
            - SHORT 포지션의 경우, 현재 가격이 이전 최저가보다 낮으면 최저가를 갱신합니다.
        """
        for record in self.executions:
            if record.get("closed", False):
                continue  # 이미 종료된 실행은 건너뜁니다.
            if self.side == "LONG":
                # LONG 포지션: 이전 최고가와 비교하여 갱신
                prev = record.get("highest_price_since_entry", record["entry_price"])
                if current_price > prev:
                    record["highest_price_since_entry"] = current_price
                    logger.debug(f"Updated highest price: {prev} -> {current_price} for execution at entry {record['entry_price']}")
            elif self.side == "SHORT":
                # SHORT 포지션: 이전 최저가와 비교하여 갱신
                prev = record.get("lowest_price_since_entry", record["entry_price"])
                if current_price < prev:
                    record["lowest_price_since_entry"] = current_price
                    logger.debug(f"Updated lowest price: {prev} -> {current_price} for execution at entry {record['entry_price']}")
        logger.debug(f"Extremum values updated with current_price={current_price}")

    def get_total_size(self) -> float:
        """
        포지션 내 모든 열려 있는 실행의 총 거래 수량을 계산합니다.
        
        Returns:
            float: 미종료 실행들의 수량 합계.
        """
        return sum(record['size'] for record in self.executions if not record.get("closed", False))

    def get_average_entry_price(self) -> float:
        """
        포지션 내 모든 열려 있는 실행의 가중 평균 진입 가격을 계산합니다.
        
        가중치는 각 실행의 거래 수량에 비례합니다.
        
        Returns:
            float: 가중 평균 진입 가격. 열려 있는 실행이 없으면 0.0을 반환.
        """
        total_cost = sum(record['entry_price'] * record['size'] for record in self.executions if not record.get("closed", False))
        total_qty = self.get_total_size()
        return total_cost / total_qty if total_qty > 0 else 0.0

    def remove_execution(self, index: int) -> None:
        """
        지정한 인덱스의 실행 기록을 포지션에서 제거합니다.
        
        Parameters:
            index (int): 제거할 실행 기록의 인덱스.
            
        Returns:
            None
        
        Notes:
            - 인덱스가 유효하지 않으면 경고 로그를 남깁니다.
        """
        if 0 <= index < len(self.executions):
            self.executions.pop(index)
            logger.debug(f"Execution removed at index {index}")
        else:
            logger.warning(f"Failed to remove execution: invalid index {index}")

    def is_empty(self) -> bool:
        """
        포지션 내 모든 실행이 종료되었는지 여부를 확인합니다.
        
        Returns:
            bool: 모든 실행이 종료되었다면 True, 그렇지 않으면 False.
        """
        return all(record.get("closed", False) for record in self.executions)

    def partial_close_execution(self, index: int, close_ratio: float, min_order_size: float = 1e-8) -> float:
        """
        포지션의 특정 실행을 부분적으로 청산하여 거래 수량을 줄입니다.
        
        Parameters:
            index (int): 청산할 실행 기록의 인덱스.
            close_ratio (float): 청산할 비율 (0보다 크고 1 이하).
            min_order_size (float): 남은 거래 수량이 이 값보다 작으면 해당 실행을 종료 처리합니다.
            
        Returns:
            float: 청산된 거래 수량. 청산에 실패하면 0.0 반환.
        
        Notes:
            - close_ratio가 유효한 범위(0,1]가 아니면 오류 로그를 남깁니다.
            - 실행 후 남은 거래 수량이 최소 주문 크기보다 작으면 해당 실행을 종료 상태로 표시합니다.
        """
        if not (0 < close_ratio <= 1):
            logger.error("close_ratio must be between 0 and 1.", exc_info=True)
            return 0.0
        if 0 <= index < len(self.executions):
            record = self.executions[index]
            # 청산할 거래 수량 계산 (현재 수량의 close_ratio 비율)
            qty_to_close = record['size'] * close_ratio
            # 거래 수량 감소 처리
            record['size'] -= qty_to_close
            # exit_targets가 있다면, 청산 처리된 타겟은 제거
            if record.get('exit_targets'):
                record['exit_targets'] = [t for t in record['exit_targets'] if not t.get('hit', False)]
            # 남은 거래 수량이 최소 주문 크기보다 작으면 해당 실행을 종료로 처리
            if record['size'] < min_order_size:
                record['closed'] = True
                logger.debug(f"Execution at index {index} closed due to size below minimum order size.")
            logger.debug(f"Partial close executed: index={index}, ratio={close_ratio}, closed qty={qty_to_close}")
            return qty_to_close
        logger.warning(f"Partial close failed: invalid index {index}")
        return 0.0
