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
        logger.info(f"Account 초기화 완료: 초기 잔고 {initial_balance:.2f}")

    def add_position(self, position):
        """
        포지션을 추가합니다.
        """
        self.positions.append(position)
        logger.info(f"포지션 추가됨: ID={position.position_id}")

    def remove_position(self, position):
        """
        포지션을 제거합니다.
        """
        if position in self.positions:
            self.positions.remove(position)
            logger.info(f"포지션 제거됨: ID={position.position_id}")
        else:
            logger.warning(f"포지션 제거 실패: ID={position.position_id} not found")

    def get_used_balance(self):
        """
        현재 포지션들로 인해 사용된 금액 계산 (수수료 포함)
        """
        used = 0.0
        for pos in self.positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    calc = exec_record["entry_price"] * exec_record["size"] * (1 + self.fee_rate)
                    used += calc
                    # INFO 레벨 로그로 남기면 AggregatingHandler가 동일 기준으로 누적하여 요약합니다.
                    logger.info(
                        f"계산 중: 포지션 ID={pos.position_id}, "
                        f"entry_price={exec_record['entry_price']}, size={exec_record['size']}, "
                        f"fee_rate={self.fee_rate}, 계산값={calc:.2f}"
                    )
        logger.info(f"총 사용 금액 계산: {used:.2f}")
        return used

    def get_available_balance(self):
        """
        현물 잔고에서 사용 중 금액을 제외한 가용 현금 반환
        """
        used = self.get_used_balance()
        available = self.spot_balance - used
        logger.info(
            f"가용 잔고 계산: spot_balance={self.spot_balance:.2f} - used={used:.2f} = available={available:.2f}"
        )
        return available

    def update_after_trade(self, trade):
        """
        체결된 거래의 pnl을 반영하여 잔고를 업데이트합니다.
        """
        pnl = trade.get("pnl", 0.0)
        previous_balance = self.spot_balance
        self.spot_balance += pnl
        logger.info(
            f"거래 체결 업데이트: pnl={pnl:.2f}, 현물 잔고={self.spot_balance:.2f}"
        )
        logger.info(
            f"업데이트 상세: 이전 현물 잔고={previous_balance:.2f}, pnl {pnl:.2f} 반영하여 {self.spot_balance:.2f}가 되었습니다."
        )

    def convert_to_stablecoin(self, amount, conversion_fee=0.001):
        """
        현물 자산 일부를 스테이블코인으로 전환합니다.
        """
        available = self.get_available_balance()
        logger.info(
            f"전환 요청: amount={amount:.2f}, 가용 잔고={available:.2f}"
        )
        if amount > available:
            logger.info(
                f"요청 금액 {amount:.2f}이 가용 잔고 {available:.2f}보다 큽니다. 가용 잔고로 조정합니다."
            )
            amount = available
        fee = amount * conversion_fee
        net_amount = amount - fee
        logger.info(
            f"전환 계산: amount={amount:.2f}, conversion_fee={conversion_fee}, fee={fee:.2f}, net_amount={net_amount:.2f}"
        )
        self.spot_balance -= amount
        self.stablecoin_balance += net_amount
        logger.info(
            f"현물 -> 스테이블코인 전환: 전환액={amount:.2f}, 수수료={fee:.2f}, 전환 후 현물 잔고={self.spot_balance:.2f}"
        )
        return net_amount

    def convert_to_spot(self, amount, conversion_fee=0.001):
        """
        스테이블코인 일부를 현물 자산으로 전환합니다.
        """
        logger.info(
            f"현물 전환 요청: amount={amount:.2f}, 현재 스테이블코인 잔고={self.stablecoin_balance:.2f}"
        )
        if amount > self.stablecoin_balance:
            logger.info(
                f"요청 금액 {amount:.2f}이 스테이블코인 잔고 {self.stablecoin_balance:.2f}보다 큽니다. 잔고로 조정합니다."
            )
            amount = self.stablecoin_balance
        fee = amount * conversion_fee
        net_amount = amount - fee
        logger.info(
            f"전환 계산: amount={amount:.2f}, conversion_fee={conversion_fee}, fee={fee:.2f}, net_amount={net_amount:.2f}"
        )
        self.stablecoin_balance -= amount
        self.spot_balance += net_amount
        logger.info(
            f"스테이블코인 -> 현물 전환: 전환액={amount:.2f}, 수수료={fee:.2f}, 전환 후 현물 잔고={self.spot_balance:.2f}"
        )
        return net_amount

    def __str__(self):
        return (
            f"Account(spot_balance={self.spot_balance:.2f}, "
            f"stablecoin_balance={self.stablecoin_balance:.2f}, "
            f"available_balance={self.get_available_balance():.2f})"
        )
