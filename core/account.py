# core/account.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

class Account:
    def __init__(self, initial_balance: float, fee_rate: float = 0.001):
        """
        계좌 생성:
          - 초기 잔고는 현물(spot) 자산으로 간주합니다.
          - fee_rate는 거래 시 적용되는 수수료율입니다.
        """
        self.initial_balance = initial_balance
        self.spot_balance = initial_balance
        self.stablecoin_balance = 0.0
        self.fee_rate = fee_rate
        self.positions = []  # 보유 중인 포지션 객체 리스트
        logger.debug(f"Account 초기화 완료: 초기 잔고 {initial_balance:.2f}")

    def add_position(self, position) -> None:
        """
        새로운 포지션을 추가합니다.
        """
        self.positions.append(position)
        logger.debug(f"포지션 추가됨: ID={position.position_id}")

    def remove_position(self, position) -> None:
        """
        보유 포지션 목록에서 지정된 포지션을 제거합니다.
        """
        if position in self.positions:
            self.positions.remove(position)
            logger.debug(f"포지션 제거됨: ID={position.position_id}")
        else:
            logger.warning(f"포지션 제거 실패: ID={position.position_id} not found")

    def get_used_balance(self) -> float:
        """
        미체결 포지션으로 인해 사용 중인 금액(수수료 포함)을 계산합니다.
        """
        used = 0.0
        for pos in self.positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    calc = exec_record["entry_price"] * exec_record["size"] * (1 + self.fee_rate)
                    used += calc
                    logger.debug(
                        f"계산 중: 포지션 ID={pos.position_id}, "
                        f"entry_price={exec_record['entry_price']}, size={exec_record['size']}, "
                        f"fee_rate={self.fee_rate}, 계산값={calc:.2f}"
                    )
        logger.debug(f"총 사용 금액 계산: {used:.2f}")
        return used

    def get_available_balance(self) -> float:
        """
        현재 현물 잔고에서 사용 중 금액을 차감한 가용 잔액을 반환합니다.
        """
        used = self.get_used_balance()
        available = self.spot_balance - used
        logger.debug(
            f"가용 잔고 계산: spot_balance={self.spot_balance:.2f} - used={used:.2f} = available={available:.2f}"
        )
        return available

    def update_after_trade(self, trade: dict) -> None:
        """
        체결된 거래의 PnL을 반영하여 계좌 잔고를 업데이트합니다.
        """
        pnl = trade.get("pnl", 0.0)
        previous_balance = self.spot_balance
        self.spot_balance += pnl
        logger.debug(
            f"거래 체결 업데이트: pnl={pnl:.2f}, 현물 잔고={self.spot_balance:.2f}"
        )
        logger.debug(
            f"업데이트 상세: 이전 잔고={previous_balance:.2f} → {self.spot_balance:.2f}"
        )

    def convert_to_stablecoin(self, amount: float, conversion_fee: float = 0.001) -> float:
        """
        현물 자산의 일부를 스테이블코인으로 전환합니다.
        """
        available = self.get_available_balance()
        logger.debug(f"전환 요청: amount={amount:.2f}, 가용 잔고={available:.2f}")
        if amount > available:
            logger.debug(f"요청 금액 {amount:.2f}이 가용 잔고 {available:.2f}보다 큽니다. 조정합니다.")
            amount = available
        fee = amount * conversion_fee
        net_amount = amount - fee
        self.spot_balance -= amount
        self.stablecoin_balance += net_amount
        logger.debug(
            f"현물 → 스테이블코인 전환: amount={amount:.2f}, fee={fee:.2f}, "
            f"전환 후 현물 잔고={self.spot_balance:.2f}"
        )
        return net_amount

    def convert_to_spot(self, amount: float, conversion_fee: float = 0.001) -> float:
        """
        스테이블코인을 현물 자산으로 전환합니다.
        """
        logger.debug(f"현물 전환 요청: amount={amount:.2f}, 스테이블코인 잔고={self.stablecoin_balance:.2f}")
        if amount > self.stablecoin_balance:
            logger.debug(f"요청 금액 {amount:.2f}이 잔고 {self.stablecoin_balance:.2f}보다 큽니다. 조정합니다.")
            amount = self.stablecoin_balance
        fee = amount * conversion_fee
        net_amount = amount - fee
        self.stablecoin_balance -= amount
        self.spot_balance += net_amount
        logger.debug(
            f"스테이블코인 → 현물 전환: amount={amount:.2f}, fee={fee:.2f}, "
            f"전환 후 현물 잔고={self.spot_balance:.2f}"
        )
        return net_amount

    def __str__(self) -> str:
        return (
            f"Account(spot_balance={self.spot_balance:.2f}, "
            f"stablecoin_balance={self.stablecoin_balance:.2f}, "
            f"available_balance={self.get_available_balance():.2f})"
        )
