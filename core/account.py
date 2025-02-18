# core/account.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class Account:
    def __init__(self, initial_balance: float, fee_rate: float = 0.001) -> None:
        """
        계좌 생성:
          - 초기 잔고는 현물(spot) 자산으로 간주하며, fee_rate는 거래 시 적용되는 수수료율입니다.
        """
        if initial_balance < 0:
            raise ValueError("Initial balance must be non-negative.")
        self.initial_balance: float = initial_balance
        self.spot_balance: float = initial_balance
        self.stablecoin_balance: float = 0.0
        self.fee_rate: float = fee_rate
        self.positions: list = []  # 보유 중인 포지션 객체 리스트
        logger.debug(f"Account initialized with balance: {initial_balance:.2f}")

    def add_position(self, position) -> None:
        """새로운 포지션을 추가합니다."""
        self.positions.append(position)
        logger.debug(f"Position added: ID={position.position_id}")

    def remove_position(self, position) -> None:
        """보유 포지션 목록에서 지정된 포지션을 제거합니다."""
        if position in self.positions:
            self.positions.remove(position)
            logger.debug(f"Position removed: ID={position.position_id}")
        else:
            logger.warning(f"Failed to remove position: ID={position.position_id}")

    def get_used_balance(self) -> float:
        """미체결 포지션으로 인해 사용 중인 금액(수수료 포함)을 계산합니다."""
        used: float = 0.0
        for pos in self.positions:
            for record in pos.executions:
                if not record.get("closed", False):
                    used += record["entry_price"] * record["size"] * (1 + self.fee_rate)
        return used

    def get_available_balance(self) -> float:
        """현재 현물 잔고에서 사용 중 금액을 차감한 가용 잔액을 반환합니다."""
        available = self.spot_balance - self.get_used_balance()
        return available if available >= 0 else 0.0

    def update_after_trade(self, trade: dict) -> None:
        """
        체결된 거래의 PnL을 반영하여 계좌 잔고를 업데이트합니다.
        trade dict는 최소한 'pnl' 키를 포함해야 합니다.
        """
        pnl = trade.get("pnl", 0.0)
        self.spot_balance += pnl
        logger.debug(f"Trade executed: PnL={pnl:.2f}, Updated spot balance={self.spot_balance:.2f}")

    def convert_to_stablecoin(self, amount: float, conversion_fee: float = 0.001) -> float:
        """
        현물 자산의 일부를 스테이블코인으로 전환합니다.
        가용 잔액보다 큰 금액 요청 시 가용 잔액으로 자동 조정됩니다.
        """
        if amount <= 0:
            logger.error("Conversion amount must be positive.")
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
        스테이블코인을 현물 자산으로 전환합니다.
        요청 금액이 잔고보다 클 경우 잔고로 자동 조정됩니다.
        """
        if amount <= 0:
            logger.error("Conversion amount must be positive.")
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
        return (
            f"Account(spot_balance={self.spot_balance:.2f}, "
            f"stablecoin_balance={self.stablecoin_balance:.2f}, "
            f"available_balance={self.get_available_balance():.2f})"
        )
