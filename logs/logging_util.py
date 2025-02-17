import threading
import os
import glob
from logs.logger_config import setup_logger

class LoggingUtil:
    """
    LoggingUtil는 이벤트 로깅과 로그 파일 관리를 제공합니다.
    이벤트 발생 시 INFO/DEBUG 레벨 로그를 기록하며, clear_log_files()로 logs 폴더 내 .log 파일을 삭제합니다.
    또한, log_weekly_signal() 메서드를 통해 주간 전략 신호 이벤트를 별도로 로깅할 수 있습니다.
    """
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.lock = threading.RLock()
        self.logger = setup_logger(module_name)

    def log_event(self, event_message: str) -> None:
        with self.lock:
            self.logger.debug(f"[{self.module_name}] Event: {event_message}")

    def log_summary(self) -> None:
        with self.lock:
            self.logger.debug(f"[{self.module_name}] Summary requested.")

    def log_weekly_signal(self, event_message: str) -> None:
        """
        주간 전략 신호 이벤트를 INFO 레벨로 로깅하며, 기록에 'is_weekly_signal' 플래그를 추가합니다.
        이를 통해 AggregatingHandler에서 별도로 집계할 수 있습니다.
        """
        with self.lock:
            self.logger.debug(f"[WEEKLY_SIGNAL] {event_message}", extra={'is_weekly_signal': True})

    @staticmethod
    def clear_log_files():
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, "logs")
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        for log_file in log_files:
            try:
                os.remove(log_file)
                print(f"Deleted log file: {log_file}")
            except Exception as e:
                print(f"Failed to remove log file {log_file}: {e}")
