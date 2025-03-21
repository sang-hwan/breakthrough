# trading/position_management.py
import uuid
from datetime import timedelta
import pandas as pd
from logs.log_config import setup_logger
from logging.logging_util import LoggingUtil

logger = setup_logger(__name__)

class Account:
    """
    계좌의 잔고와 포지션을 관리합니다.

    Attributes:
        initial_balance (float): 계좌 초기 잔고.
        spot_balance (float): 거래에 즉시 사용 가능한 현물 잔고.
        stablecoin_balance (float): 스테이블 코인 잔고.
        fee_rate (float): 거래 수수료율.
        positions (list): 등록된 포지션 목록.
    """

    def __init__(self, initial_balance: float, fee_rate: float = 0.001) -> None:
        """
        계좌를 초기화합니다.

        Parameters:
            initial_balance (float): 초기 잔고 (음수가 아니어야 함).
            fee_rate (float): 거래 수수료율 (기본 0.1%).
        Raises:
            ValueError: 초기 잔고가 음수일 경우.
        """
        if initial_balance < 0:
            raise ValueError("Initial balance must be non-negative.")
        self.initial_balance: float = initial_balance
        self.spot_balance: float = initial_balance
        self.stablecoin_balance: float = 0.0
        self.fee_rate: float = fee_rate
        self.positions: list = []
        logger.debug(f"Account initialized with balance: {initial_balance:.2f}")

    def add_position(self, position) -> None:
        """
        새로운 포지션을 계좌에 추가합니다.

        Parameters:
            position: 포지션 객체 (반드시 position_id 속성을 포함).
        """
        self.positions.append(position)
        logger.debug(f"Position added: ID={position.position_id}")

    def remove_position(self, position) -> None:
        """
        계좌에서 특정 포지션을 제거합니다.

        Parameters:
            position: 제거할 포지션 객체.
        """
        if position in self.positions:
            self.positions.remove(position)
            logger.debug(f"Position removed: ID={position.position_id}")
        else:
            logger.warning(f"Failed to remove position: ID={position.position_id}")

    def get_used_balance(self) -> float:
        """
        사용 중인 잔고(잠긴 금액)를 계산합니다.
        
        Returns:
            float: 사용 중인 총 잔고.
        """
        used = 0.0
        for pos in self.positions:
            for record in pos.executions:
                if not record.get("closed", False):
                    used += record["entry_price"] * record["size"] * (1 + self.fee_rate)
        return used

    def get_available_balance(self) -> float:
        """
        새 거래에 사용 가능한 잔고를 계산합니다.
        
        Returns:
            float: 사용 가능한 잔고 (0 이상).
        """
        available = self.spot_balance - self.get_used_balance()
        return available if available >= 0 else 0.0

    def update_after_trade(self, trade: dict) -> None:
        """
        거래 체결 후 계좌 잔고를 업데이트합니다.

        Parameters:
            trade (dict): 거래 세부 정보 (반드시 'pnl' 포함).
        """
        pnl = trade.get("pnl", 0.0)
        self.spot_balance += pnl
        logger.debug(f"Trade executed: PnL={pnl:.2f}, Updated spot balance={self.spot_balance:.2f}")

    def convert_to_stablecoin(self, amount: float, conversion_fee: float = 0.001) -> float:
        """
        현물 잔고 일부를 스테이블코인으로 변환합니다.

        Parameters:
            amount (float): 변환할 금액.
            conversion_fee (float): 변환 수수료 (기본 0.1%).
        Returns:
            float: 변환 후 순 금액.
        """
        if amount <= 0:
            logger.error("Conversion amount must be positive.", exc_info=True)
            return 0.0
        available = self.get_available_balance()
        if amount > available:
            amount = available
        fee = amount * conversion_fee
        net_amount = amount - fee
        self.spot_balance -= amount
        self.stablecoin_balance += net_amount
        logger.debug(f"Converted {amount:.2f} from spot to stablecoin (fee {fee:.2f}, net {net_amount:.2f}).")
        return net_amount

    def convert_to_spot(self, amount: float, conversion_fee: float = 0.001) -> float:
        """
        스테이블코인 잔고 일부를 현물 잔고로 변환합니다.

        Parameters:
            amount (float): 변환할 금액.
            conversion_fee (float): 변환 수수료 (기본 0.1%).
        Returns:
            float: 변환 후 순 금액.
        """
        if amount <= 0:
            logger.error("Conversion amount must be positive.", exc_info=True)
            return 0.0
        if amount > self.stablecoin_balance:
            amount = self.stablecoin_balance
        fee = amount * conversion_fee
        net_amount = amount - fee
        self.stablecoin_balance -= amount
        self.spot_balance += net_amount
        logger.debug(f"Converted {amount:.2f} from stablecoin to spot (fee {fee:.2f}, net {net_amount:.2f}).")
        return net_amount

    def __str__(self) -> str:
        """
        계좌 상태를 문자열로 반환합니다.
        
        Returns:
            str: 잔고 및 사용 가능 잔고 포함.
        """
        return (
            f"Account(spot_balance={self.spot_balance:.2f}, "
            f"stablecoin_balance={self.stablecoin_balance:.2f}, "
            f"available_balance={self.get_available_balance():.2f})"
        )

class Position:
    """
    개별 거래 포지션을 관리합니다.

    Attributes:
        position_id (str): 고유 포지션 식별자.
        side (str): 포지션 방향 ("LONG" 또는 "SHORT").
        executions (list): 거래 실행 기록 목록.
        initial_price (float): 진입 가격.
        maximum_size (float): 최대 포지션 크기.
        total_splits (int): 계획된 분할 진입 횟수.
        executed_splits (int): 실제 실행된 분할 횟수.
        allocation_plan (list): 분할 진입 시 자금 배분 계획.
        highest_price 또는 lowest_price (float): 포지션 진행 중 극값 추적.
    """

    def __init__(self, side: str = "LONG", initial_price: float = None,
                 maximum_size: float = 0.0, total_splits: int = 1,
                 allocation_plan: list = None) -> None:
        """
        새로운 포지션을 초기화합니다.

        Parameters:
            side (str): 거래 방향 ("LONG" 또는 "SHORT").
            initial_price (float): 진입 가격 (양수여야 함).
            maximum_size (float): 최대 포지션 크기.
            total_splits (int): 계획된 분할 횟수.
            allocation_plan (list): 자금 배분 계획 (옵션).
        Raises:
            ValueError: 유효하지 않은 initial_price.
        """
        if initial_price is None or initial_price <= 0:
            raise ValueError("Initial price must be positive.")
        self.position_id: str = str(uuid.uuid4())
        self.side: str = side.upper()
        self.executions: list = []
        self.initial_price: float = initial_price
        self.maximum_size: float = maximum_size
        self.total_splits: int = total_splits
        self.executed_splits: int = 0
        self.allocation_plan: list = allocation_plan if allocation_plan is not None else []
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

        Parameters:
            entry_price (float): 진입 가격.
            size (float): 거래 수량.
            stop_loss (float): (선택) 손절 가격.
            take_profit (float): (선택) 익절 가격.
            entry_time: (선택) 거래 실행 시각.
            exit_targets (list): (선택) 부분 청산 목표 리스트.
            trade_type (str): 거래 유형.
            min_order_size (float): 최소 거래 수량.
        """
        if size < min_order_size:
            logger.warning("Execution size below minimum order size; execution not added.")
            return
        if exit_targets and not isinstance(exit_targets, list):
            logger.error("exit_targets must be a list.", exc_info=True)
            return

        targets = []
        if exit_targets:
            for target_price, exit_ratio in exit_targets:
                targets.append({'price': target_price, 'exit_ratio': exit_ratio, 'hit': False})
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
        if self.side == "SHORT":
            execution["lowest_price_since_entry"] = entry_price
        else:
            execution["highest_price_since_entry"] = entry_price

        self.executions.append(execution)
        logger.debug(f"Execution added: entry_price={entry_price}, size={size}, type={trade_type}")

    def update_extremum(self, current_price: float) -> None:
        """
        현재 가격을 반영하여 각 실행의 최고가 또는 최저가를 업데이트합니다.

        Parameters:
            current_price (float): 최신 시장 가격.
        """
        for record in self.executions:
            if record.get("closed", False):
                continue
            if self.side == "LONG":
                prev = record.get("highest_price_since_entry", record["entry_price"])
                if current_price > prev:
                    record["highest_price_since_entry"] = current_price
                    logger.debug(f"Updated highest price: {prev} -> {current_price}")
            elif self.side == "SHORT":
                prev = record.get("lowest_price_since_entry", record["entry_price"])
                if current_price < prev:
                    record["lowest_price_since_entry"] = current_price
                    logger.debug(f"Updated lowest price: {prev} -> {current_price}")
        logger.debug(f"Extremum values updated with current_price={current_price}")

    def get_total_size(self) -> float:
        """
        열려있는 모든 실행의 총 거래 수량을 계산합니다.
        
        Returns:
            float: 총 거래 수량.
        """
        return sum(record['size'] for record in self.executions if not record.get("closed", False))

    def get_average_entry_price(self) -> float:
        """
        열려있는 실행들의 가중 평균 진입 가격을 계산합니다.
        
        Returns:
            float: 가중 평균 진입 가격.
        """
        total_cost = sum(record['entry_price'] * record['size'] for record in self.executions if not record.get("closed", False))
        total_qty = self.get_total_size()
        return total_cost / total_qty if total_qty > 0 else 0.0

    def remove_execution(self, index: int) -> None:
        """
        지정 인덱스의 실행 기록을 제거합니다.

        Parameters:
            index (int): 제거할 실행 인덱스.
        """
        if 0 <= index < len(self.executions):
            self.executions.pop(index)
            logger.debug(f"Execution removed at index {index}")
        else:
            logger.warning(f"Failed to remove execution: invalid index {index}")

    def is_empty(self) -> bool:
        """
        포지션 내 모든 실행이 종료되었는지 확인합니다.
        
        Returns:
            bool: 모든 실행 종료 시 True.
        """
        return all(record.get("closed", False) for record in self.executions)

    def partial_close_execution(self, index: int, close_ratio: float, min_order_size: float = 1e-8) -> float:
        """
        특정 실행을 부분 청산합니다.

        Parameters:
            index (int): 청산할 실행 인덱스.
            close_ratio (float): 청산 비율 (0~1].
            min_order_size (float): 남은 수량이 이 값 이하이면 실행 종료.
        Returns:
            float: 청산된 수량.
        """
        if not (0 < close_ratio <= 1):
            logger.error("close_ratio must be between 0 and 1.", exc_info=True)
            return 0.0
        if 0 <= index < len(self.executions):
            record = self.executions[index]
            qty_to_close = record['size'] * close_ratio
            record['size'] -= qty_to_close
            if record.get('exit_targets'):
                record['exit_targets'] = [t for t in record['exit_targets'] if not t.get('hit', False)]
            if record['size'] < min_order_size:
                record['closed'] = True
                logger.debug(f"Execution at index {index} closed due to size below minimum order size.")
            logger.debug(f"Partial close executed: index={index}, ratio={close_ratio}, closed qty={qty_to_close}")
            return qty_to_close
        logger.warning(f"Partial close failed: invalid index {index}")
        return 0.0

class AssetManager:
    """
    계좌의 자산 배분(스팟 vs. 스테이블코인)을 재조정(리밸런싱)하는 기능 제공.

    싱글톤 패턴을 적용하여 동일 조건의 인스턴스가 중복 생성되지 않도록 함.
    """

    _instances = {}

    def __new__(cls, account, min_rebalance_threshold=0.05, min_rebalance_interval_minutes=60):
        key = (id(account), min_rebalance_threshold, min_rebalance_interval_minutes)
        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, account, min_rebalance_threshold=0.05, min_rebalance_interval_minutes=60):
        if hasattr(self, '_initialized') and self._initialized:
            return
        if account is None:
            raise ValueError("Account must not be None.")
        self.account = account
        self.logger = setup_logger(__name__)
        self.log_util = LoggingUtil(__name__)
        self.min_rebalance_threshold = min_rebalance_threshold
        self.min_rebalance_interval = timedelta(minutes=min_rebalance_interval_minutes)
        self.last_rebalance_time = None
        self.last_account_state = None
        self.logger.debug(f"AssetManager initialized with threshold {min_rebalance_threshold} and interval {min_rebalance_interval_minutes} min")
        self._initialized = True

    def _get_account_state(self):
        """
        현재 계좌의 상태(스팟, 스테이블코인 잔고)를 반올림하여 반환합니다.
        
        Returns:
            tuple: (spot_balance, stablecoin_balance)
        """
        return (round(self.account.spot_balance, 4), round(self.account.stablecoin_balance, 4))

    def rebalance(self, market_regime):
        """
        시장 상황에 따라 계좌 자산 배분을 재조정합니다.

        Parameters:
            market_regime (str or numeric): "bullish", "bearish", "sideways" 등.
        """
        current_time = pd.Timestamp.now()
        if self.last_rebalance_time and (current_time - self.last_rebalance_time < self.min_rebalance_interval):
            self.logger.debug("Rebalance skipped due to interval constraint.")
            return

        total_assets = self.account.spot_balance + self.account.stablecoin_balance
        if total_assets <= 0:
            self.logger.warning("Total assets <= 0. Skipping rebalance.")
            return

        if not isinstance(market_regime, str):
            try:
                market_regime = {0.0: "bullish", 1.0: "bearish", 2.0: "sideways"}.get(float(market_regime), "unknown")
            except Exception:
                market_regime = "unknown"
        regime = market_regime.lower()
        if regime not in ["bullish", "bearish", "sideways"]:
            self.logger.warning(f"Market regime '{market_regime}' is unknown; treating as 'sideways'.")
            regime = "sideways"

        if regime in ["bullish", "enter_long"]:
            desired_spot = total_assets * (1.0 if regime == "enter_long" else 0.90)
        elif regime in ["bearish", "exit_all"]:
            desired_spot = total_assets * (0.0 if regime == "exit_all" else 0.10)
        elif regime == "sideways":
            desired_spot = total_assets * 0.60

        current_spot = self.account.spot_balance
        diff_ratio = abs(current_spot - desired_spot) / total_assets
        if diff_ratio < self.min_rebalance_threshold:
            self.logger.debug("No significant imbalance detected; skipping rebalance.")
            return

        try:
            if current_spot < desired_spot:
                amount_to_convert = desired_spot - current_spot
                converted = self.account.convert_to_spot(amount_to_convert)
                self.logger.debug(f"Rebalance ({regime.capitalize()}): Converted {converted:.2f} from stablecoin to spot.")
            else:
                amount_to_convert = current_spot - desired_spot
                converted = self.account.convert_to_stablecoin(amount_to_convert)
                self.logger.debug(f"Rebalance ({regime.capitalize()}): Converted {converted:.2f} from spot to stablecoin.")
        except Exception as e:
            self.logger.error(f"Rebalance conversion failed: {e}", exc_info=True)
            return

        self.last_rebalance_time = current_time
        new_state = self._get_account_state()
        if new_state != self.last_account_state:
            self.last_account_state = new_state
            self.log_util.log_event("Rebalance complete", state_key="asset_state")
