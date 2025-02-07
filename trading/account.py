# trading/account.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class Account:
    def __init__(self, initial_balance, fee_rate=0.001):
        """
        초기 잔고는 모두 현물(spot) 자산으로 간주하며,
        fee_rate는 거래 시 적용되는 수수료율입니다.
        """
        self.initial_balance = initial_balance
        self.spot_balance = initial_balance
        self.stablecoin_balance = 0.0
        self.fee_rate = fee_rate
        self.positions = []  # 보유 중인 포지션 리스트

    def add_position(self, position):
        self.positions.append(position)
        logger.debug(f"포지션 추가: {position.position_id}")

    def remove_position(self, position):
        if position in self.positions:
            self.positions.remove(position)
            logger.debug(f"포지션 제거: {position.position_id}")

    def get_used_balance(self):
        """현재 포지션들로 인해 사용된 금액 계산 (수수료 포함)"""
        used = 0.0
        for pos in self.positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    used += exec_record["entry_price"] * exec_record["size"] * (1 + self.fee_rate)
        return used

    def get_available_balance(self):
        """현물 잔고에서 사용 중 금액을 제외한 가용 현금 반환"""
        return self.spot_balance - self.get_used_balance()

    def update_after_trade(self, trade):
        """
        체결된 거래의 pnl을 반영하여 잔고를 업데이트합니다.
        (pnl이 음수면 손실, 양수면 수익으로 반영)
        """
        pnl = trade.get("pnl", 0.0)
        self.spot_balance += pnl
        logger.debug(f"거래 후 업데이트: pnl={pnl:.2f}, 현물 잔고={self.spot_balance:.2f}")

    def convert_to_stablecoin(self, amount, conversion_fee=0.001):
        """
        현물 자산 일부를 스테이블코인으로 전환합니다.
        전환 수수료를 반영하며, 가용 잔액을 초과할 경우 가능한 만큼 전환합니다.
        """
        available = self.get_available_balance()
        if amount > available:
            amount = available
        fee = amount * conversion_fee
        net_amount = amount - fee
        self.spot_balance -= amount
        self.stablecoin_balance += net_amount
        logger.debug(f"convert_to_stablecoin: 전환액={amount:.2f}, 수수료={fee:.2f}, 전환 후 현물 잔고={self.spot_balance:.2f}")
        return net_amount

    def convert_to_spot(self, amount, conversion_fee=0.001):
        """
        스테이블코인 일부를 현물 자산으로 전환합니다.
        """
        if amount > self.stablecoin_balance:
            amount = self.stablecoin_balance
        fee = amount * conversion_fee
        net_amount = amount - fee
        self.stablecoin_balance -= amount
        self.spot_balance += net_amount
        logger.debug(f"convert_to_spot: 전환액={amount:.2f}, 수수료={fee:.2f}, 전환 후 현물 잔고={self.spot_balance:.2f}")
        return net_amount

    def __str__(self):
        return (f"Account(spot_balance={self.spot_balance:.2f}, "
                f"stablecoin_balance={self.stablecoin_balance:.2f}, "
                f"available_balance={self.get_available_balance():.2f})")
