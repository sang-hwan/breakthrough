# monitoring/real_time_monitor.py
from logs.logger_config import setup_logger

class RealTimeMonitor:
    def __init__(self, price_change_threshold: float = 0.1, slippage_threshold: float = 0.02):
        """
        Args:
            price_change_threshold (float): 가격 변동 임계치 (예: 0.1 = 10% 이상 변동 시 경고)
            slippage_threshold (float): 슬리피지 임계치 (예: 0.02 = 2% 이상 가격 차이 발생 시 경고)
        """
        self.logger = setup_logger(__name__)
        self.price_change_threshold = price_change_threshold
        self.slippage_threshold = slippage_threshold

    def monitor_trade_activity(self, positions, current_market_price: float):
        """
        보유 포지션의 실행 내역과 현재 시장 가격을 비교하여,
        가격 변동 및 슬리피지 발생을 감지하고 경고 로그를 남깁니다.
        
        개선사항:
          - 각 포지션의 entry_price와 현재 시장 가격의 상대적 변화율(price_change)을 계산하여 임계치를 초과하면 경고.
          - 주문 체결 시점의 가격과 현재 가격의 차이가 슬리피지 임계치를 초과하면 별도로 경고.
        """
        for pos in positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    entry_price = exec_record.get("entry_price")
                    if entry_price is None or entry_price <= 0:
                        continue

                    # 가격 변동 감지
                    price_change = abs(current_market_price - entry_price) / entry_price
                    if price_change > self.price_change_threshold:
                        self.logger.warning(
                            f"RealTimeMonitor: Position {pos.position_id} price changed by {price_change*100:.2f}% "
                            f"(Entry: {entry_price:.2f}, Current: {current_market_price:.2f})."
                        )

                    # 슬리피지 감지 (실제 체결 가격과 현재 시장 가격 차이)
                    executed_price = exec_record.get("entry_price")  # 체결 당시 가격
                    if executed_price is not None and executed_price > 0:
                        slippage = abs(current_market_price - executed_price) / executed_price
                        if slippage > self.slippage_threshold:
                            self.logger.warning(
                                f"RealTimeMonitor: Position {pos.position_id} slippage detected: {slippage*100:.2f}% "
                                f"(Executed: {executed_price:.2f}, Current: {current_market_price:.2f})."
                            )

    def send_alert(self, message: str):
        """
        알림 메시지를 로그에 기록합니다.
        추후 이메일, SMS, 슬랙 등과 연동할 수 있도록 확장 가능합니다.
        """
        self.logger.info(f"ALERT: {message}")
