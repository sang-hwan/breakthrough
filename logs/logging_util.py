# logs/logging_util.py
import threading
import os
import glob
from logs.logger_config import setup_logger
from logs.state_change_manager import StateChangeManager

class LoggingUtil:
    """
    LoggingUtil는 이벤트 로깅과 로그 파일 관리를 제공합니다.
    log_event() 함수는 StateChangeManager를 이용하여 상태 변화가 있을 때만 로그를 기록합니다.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.lock = threading.RLock()
        self.logger = setup_logger(module_name)
        self.state_manager = StateChangeManager()  # 상태 변화 관리 인스턴스

    def log_event(self, event_message: str, state_key: str = None) -> None:
        """
        state_key가 제공되면, 이전 상태와 비교하여 변화가 있을 때만 INFO 로그를 기록합니다.
        상태 변화가 기록된 후, 특정 이벤트 사이클(예: 데이터 로드 완료 후)을 마치면 상태를 리셋할 수 있습니다.
        """
        with self.lock:
            if state_key:
                if self.state_manager.has_changed(state_key, event_message):
                    self.logger.info(f"[{self.module_name}] Event: {event_message}")
            else:
                # state_key가 없으면 무조건 기록 (예: 중요한 전역 이벤트)
                self.logger.info(f"[{self.module_name}] Event: {event_message}")

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
