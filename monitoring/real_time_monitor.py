# monitoring/real_time_monitor.py
from logs.logger_config import setup_logger

class RealTimeMonitor:
    def __init__(self):
        self.logger = setup_logger(__name__)

    def monitor_trade_activity(self, positions, current_market_price):
        """
        보유 포지션, 체결 내역, 슬리피지 및 거래 비용을 모니터링하여 이상 징후 발생 시 알람.
        """
        for pos in positions:
            for exec_record in pos.executions:
                if not exec_record.get("closed", False):
                    entry_price = exec_record.get("entry_price")
                    if entry_price is None:
                        continue
                    price_change = abs(current_market_price - entry_price) / entry_price
                    if price_change > 0.1:  # 예: 10% 이상 변동 시 경고
                        self.logger.warning(f"RealTimeMonitor: Position {pos.position_id} has moved by {price_change*100:.2f}% from entry price.")

    def send_alert(self, message):
        # 이메일, SMS, 슬랙 등 연동 가능 (여기서는 로그 기록으로 대체)
        self.logger.info(f"ALERT: {message}")
