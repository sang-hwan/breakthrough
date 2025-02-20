# core/account.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class Account:
    def __init__(self, initial_balance: float, fee_rate: float = 0.001) -> None:
        if initial_balance < 0:
            raise ValueError("Initial balance must be non-negative.")
        self.initial_balance: float = initial_balance
        self.spot_balance: float = initial_balance
        self.stablecoin_balance: float = 0.0
        self.fee_rate: float = fee_rate
        self.positions: list = []
        logger.debug(f"Account initialized with balance: {initial_balance:.2f}")

    def add_position(self, position) -> None:
        self.positions.append(position)
        logger.debug(f"Position added: ID={position.position_id}")

    def remove_position(self, position) -> None:
        if position in self.positions:
            self.positions.remove(position)
            logger.debug(f"Position removed: ID={position.position_id}")
        else:
            logger.warning(f"Failed to remove position: ID={position.position_id}")

    def get_used_balance(self) -> float:
        used: float = 0.0
        for pos in self.positions:
            for record in pos.executions:
                if not record.get("closed", False):
                    used += record["entry_price"] * record["size"] * (1 + self.fee_rate)
        return used

    def get_available_balance(self) -> float:
        available = self.spot_balance - self.get_used_balance()
        return available if available >= 0 else 0.0

    def update_after_trade(self, trade: dict) -> None:
        pnl = trade.get("pnl", 0.0)
        self.spot_balance += pnl
        logger.debug(f"Trade executed: PnL={pnl:.2f}, Updated spot balance={self.spot_balance:.2f}")

    def convert_to_stablecoin(self, amount: float, conversion_fee: float = 0.001) -> float:
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
        return (
            f"Account(spot_balance={self.spot_balance:.2f}, "
            f"stablecoin_balance={self.stablecoin_balance:.2f}, "
            f"available_balance={self.get_available_balance():.2f})"
        )
