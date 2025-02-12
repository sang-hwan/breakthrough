# logs/logging_util.py
import threading
import os
import glob
from logs.logger_config import setup_logger

class LoggingUtil:
    """
    LoggingUtil는 이벤트 로깅과 로그 파일 관리 기능을 제공합니다.
    
    - 이벤트 로깅: 각 모듈별로 인스턴스를 생성하고, 이벤트 발생 시 INFO 레벨 로그를 기록하여
      AggregatingHandler가 동일 (logger 이름, 파일, 함수) 기준으로 로그를 집계하도록 합니다.
    - 로그 파일 관리: 정적 메서드 clear_log_files()를 통해 프로젝트 루트의 logs 폴더 내 모든 .log 파일을 삭제합니다.
    """
    def __init__(self, module_name: str):
        """
        :param module_name: 해당 로거가 속한 모듈의 이름 (예: "Account", "AssetManager" 등)
        """
        self.module_name = module_name
        self.lock = threading.RLock()
        self.logger = setup_logger(module_name)

    def log_event(self, event_message: str) -> None:
        """
        단일 이벤트를 기록합니다.
        이벤트 발생 시마다 INFO 레벨로 로그를 남기면 AggregatingHandler가 동일 기준의 로그들을 집계하여
        임계치 도달 시 요약 메시지를 출력합니다.
        
        :param event_message: 기록할 이벤트 메시지
        """
        with self.lock:
            self.logger.debug(f"[{self.module_name}] Event: {event_message}")

    def log_summary(self) -> None:
        """
        AggregatingHandler가 자동으로 요약 로그를 출력하므로,
        필요 시 강제로 요약 로그를 남길 수 있도록 INFO 레벨 로그를 기록합니다.
        """
        with self.lock:
            self.logger.debug(f"[{self.module_name}] Summary requested.")

    @staticmethod
    def clear_log_files():
        """
        실행 시 프로젝트 루트의 logs 폴더 내 모든 .log 파일을 삭제합니다.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, "logs")
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
        for log_file in log_files:
            try:
                os.remove(log_file)
                print(f"Deleted log file: {log_file}")
            except Exception as e:
                print(f"Failed to remove log file {log_file}: {e}")

# 테스트 코드에서 요구하는 alias는 LoggingUtil 그대로 사용하면 됩니다.
