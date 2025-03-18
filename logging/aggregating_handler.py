# logs/aggregating_handler.py
import logging  # 로그 기록을 위한 표준 모듈
import os       # 파일 및 경로 관련 작업 지원
import threading  # 스레드 동기화를 위한 모듈

class AggregatingHandler(logging.Handler):
    """
    AggregatingHandler는 (logger 이름, 파일명, 함수명)별로 이벤트 발생 횟수를 집계합니다.
    플러시 시 집계된 결과를 최종 로그로 출력하고, 집계를 초기화합니다.
    
    주요 기능:
      - 로그 이벤트가 발생할 때마다 해당 키의 발생 횟수를 카운트합니다.
      - 'is_weekly_signal' 속성이 있는 경우 별도로 주간 신호 집계합니다.
    """
    def __init__(self, level=logging.DEBUG):
        """
        초기화 메서드
        
        Parameters:
            level (int): 로깅 레벨 (기본값: logging.DEBUG)
        
        주요 동작:
          - 전체 로그 집계와 주간 신호 집계를 위한 딕셔너리 초기화
          - 스레드 안전성을 보장하기 위해 RLock 객체 생성
        """
        super().__init__(level)
        self.total_aggregation = {}       # 전체 로그 집계를 위한 딕셔너리, key: (logger name, filename, funcName)
        self.weekly_signal_aggregation = {}  # 주간 신호 로그 집계를 위한 딕셔너리
        self.lock = threading.RLock()       # 스레드 간 동시 접근 제어를 위한 재진입 락

    def emit(self, record):
        """
        로그 레코드가 발생할 때마다 호출되며, 해당 이벤트를 집계합니다.
        
        Parameters:
            record (logging.LogRecord): 발생한 로그 이벤트에 대한 정보를 담은 객체
        
        주요 동작:
          - record에서 (logger name, 파일명, 함수명)을 추출하여 집계의 key로 사용
          - 전체 집계와 'is_weekly_signal' 속성에 따른 주간 신호 집계를 각각 업데이트
          - 예외 발생 시 handleError() 호출하여 오류 처리
        """
        try:
            # 파일 경로에서 파일 이름만 추출하여 key 생성
            key = (record.name, os.path.basename(record.pathname), record.funcName)
            with self.lock:  # 스레드 안전하게 집계 딕셔너리 업데이트
                self.total_aggregation[key] = self.total_aggregation.get(key, 0) + 1
                # 'is_weekly_signal' 속성이 True면 주간 신호 집계 업데이트
                if getattr(record, 'is_weekly_signal', False):
                    self.weekly_signal_aggregation[key] = self.weekly_signal_aggregation.get(key, 0) + 1
        except Exception:
            self.handleError(record)

    def flush_aggregation_summary(self):
        """
        집계된 로그 결과를 포맷팅하여 최종 로그로 출력한 후, 집계 데이터를 초기화합니다.
        
        주요 동작:
          - 전체 집계와 주간 신호 집계에 대해 각각 문자열 리스트를 생성 후 하나의 메시지로 결합
          - 생성된 메시지를 로그에 기록하고, 집계 딕셔너리를 초기화
        """
        with self.lock:  # 집계 데이터에 대한 동시 접근 제어
            if self.total_aggregation:
                summary_lines = [
                    f"{filename}:{funcname} (logger: {logger_name}) - 총 {count}회 발생"
                    for (logger_name, filename, funcname), count in self.total_aggregation.items()
                ]
                summary = "\n".join(summary_lines)
                try:
                    # 전체 집계 결과를 info 레벨로 로그 출력
                    logging.getLogger().info("전체 누적 로그 집계:\n" + summary)
                except Exception:
                    pass
                self.total_aggregation.clear()  # 전체 집계 초기화
            if self.weekly_signal_aggregation:
                weekly_summary_lines = [
                    f"{filename}:{funcname} (logger: {logger_name}) - 주간 신호 {count}회 발생"
                    for (logger_name, filename, funcname), count in self.weekly_signal_aggregation.items()
                ]
                weekly_summary = "\n".join(weekly_summary_lines)
                try:
                    logging.getLogger().info("전체 주간 신호 로그 집계:\n" + weekly_summary)
                except Exception:
                    pass
                self.weekly_signal_aggregation.clear()  # 주간 신호 집계 초기화
