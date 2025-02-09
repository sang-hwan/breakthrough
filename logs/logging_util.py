# logs/logging_util.py
import logging
import threading

class EventLogger:
    """
    모든 모듈에서 공통으로 사용할 이벤트 집계 및 요약 로그 출력 기능을 제공하는 클래스입니다.
    
    - 각 모듈별 인스턴스를 생성하여 이벤트 기록 시 사용합니다.
    - 내부적으로 이벤트 카운터와 이벤트 메시지를 저장하며, 설정된 횟수(기본 2000회)마다 요약 로그를 INFO 레벨로 출력합니다.
    """
    def __init__(self, module_name: str, summary_threshold: int = 2000):
        """
        :param module_name: 해당 로거가 속한 모듈의 이름 (예: "Account", "AssetManager" 등)
        :param summary_threshold: 요약 로그를 출력할 이벤트 횟수 (기본값 2000)
        """
        self.module_name = module_name
        self.summary_threshold = summary_threshold
        self.event_counter = 0
        self.events = []
        self.lock = threading.RLock()
        # 인스턴스용 logger 생성 (이 logger는 외부에서 주입한 logger로 대체될 수 있음)
        self.logger = logging.getLogger(module_name)

    def log_event(self, event_message: str, level: int = logging.DEBUG) -> None:
        """
        단일 이벤트를 기록합니다.
        이벤트 발생 시마다 호출되며, 내부 카운터를 증가시키고 로그를 출력합니다.
        설정된 횟수마다 요약 로그도 함께 출력합니다.
        
        :param event_message: 기록할 이벤트 메시지
        :param level: 로그 레벨 (기본: DEBUG)
        """
        with self.lock:
            self.event_counter += 1
            self.events.append(event_message)
            # 이제 전역 logging 대신 인스턴스의 logger를 사용합니다.
            self.logger.log(level, f"[{self.module_name}] Event: {event_message} (Count: {self.event_counter})")
            
            # 설정된 횟수마다 요약 로그 출력
            if self.event_counter % self.summary_threshold == 0:
                self.log_summary()

    def log_summary(self) -> None:
        """
        현재까지 기록된 이벤트들을 요약하여 INFO 레벨로 출력합니다.
        요약에는 전체 이벤트 횟수와 마지막에 기록된 이벤트 메시지를 포함합니다.
        """
        with self.lock:
            last_event = self.events[-1] if self.events else "No events recorded"
            summary_message = (
                f"[{self.module_name}] 요약 로그: {self.event_counter}회 이벤트, 마지막 이벤트: {last_event}"
            )
            self.logger.info(summary_message)

# 테스트 코드에서 요구하는 LoggingUtil 클래스가 존재하도록,
# EventLogger를 LoggingUtil이라는 이름으로 alias 처리합니다.
LoggingUtil = EventLogger
