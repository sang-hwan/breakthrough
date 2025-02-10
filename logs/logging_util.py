# logs/logging_util.py
import logging
import threading

class EventLogger:
    """
    AggregatingHandler를 사용하여 이벤트 로그를 집계하는 클래스입니다.
    
    각 모듈에서 인스턴스를 생성하고 이벤트 발생 시 INFO 레벨 로그로 기록하면,
    루트 로거에 등록된 AggregatingHandler가 동일 (logger 이름, 파일, 함수) 기준으로
    로그를 누적하여 임계치 도달 시 자동으로 요약 메시지를 출력합니다.
    
    기존의 수동 집계(이벤트 카운터, 이벤트 목록)는 제거되었습니다.
    """
    def __init__(self, module_name: str):
        """
        :param module_name: 해당 로거가 속한 모듈의 이름 (예: "Account", "AssetManager" 등)
        """
        self.module_name = module_name
        self.lock = threading.RLock()
        self.logger = logging.getLogger(module_name)

    def log_event(self, event_message: str) -> None:
        """
        단일 이벤트를 기록합니다.
        이벤트 발생 시마다 INFO 레벨로 로그를 남기면, 
        AggregatingHandler가 동일 기준의 로그들을 집계하여 임계치 도달 시 요약 메시지를 출력합니다.
        
        :param event_message: 기록할 이벤트 메시지
        """
        with self.lock:
            self.logger.info(f"[{self.module_name}] Event: {event_message}")

    def log_summary(self) -> None:
        """
        요약 로그는 AggregatingHandler가 자동으로 출력하므로,
        필요 시 강제로 요약 로그를 남길 수 있도록 INFO 레벨 로그를 기록합니다.
        """
        with self.lock:
            self.logger.info(f"[{self.module_name}] Summary requested.")

# 테스트 코드에서 요구하는 LoggingUtil 클래스가 존재하도록,
# EventLogger를 LoggingUtil이라는 이름으로 alias 처리합니다.
LoggingUtil = EventLogger
