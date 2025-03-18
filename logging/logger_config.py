# logs/logger_config.py
import logging  # 로그 기록을 위한 표준 모듈
import os       # 파일 경로 및 디렉토리 생성 관련 모듈
import queue    # 로그 메시지 큐를 다루기 위한 모듈
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener  # 회전 파일 핸들러 및 큐 관련 핸들러 임포트
from logging.aggregating_handler import AggregatingHandler  # 사용자 정의 AggregatingHandler 임포트

# 전역 변수: 파일 로그는 INFO 레벨 이상의 메시지만 기록
FILE_LOG_LEVEL = logging.INFO
# 전역 변수: 콘솔 로그는 DEBUG 레벨 이상의 메시지를 기록
detail_level = logging.DEBUG

# 로그 파일을 저장할 디렉토리 설정 (logs/log_files)
LOG_FILES_DIR = os.path.join("logs", "log_files")
if not os.path.exists(LOG_FILES_DIR):
    os.makedirs(LOG_FILES_DIR)  # 디렉토리가 없으면 생성
# 기본 로그 파일 경로 설정
BASE_LOG_FILE = os.path.join(LOG_FILES_DIR, "project.log")

class OneLineFormatter(logging.Formatter):
    """
    OneLineFormatter는 로그 메시지 내의 줄바꿈을 파이프(|) 문자로 대체하여 한 줄로 출력하도록 포맷팅합니다.
    """
    def format(self, record):
        formatted = super().format(record)  # 기본 포맷팅 수행
        return formatted.replace("\n", " | ")  # 줄바꿈을 ' | '로 대체

class LineRotatingFileHandler(RotatingFileHandler):
    """
    LineRotatingFileHandler는 로그 파일의 최대 라인 수를 초과하면 새로운 파일로 롤오버하는 기능을 제공합니다.
    
    주요 기능:
      - 파일 크기 대신 로그 라인 수를 기준으로 롤오버
      - 롤오버 시 새로운 파일 이름 생성 및 라인 카운트 초기화
    """
    def __init__(self, base_filename, mode='a', max_lines=500, encoding=None, delay=False):
        """
        초기화 메서드
        
        Parameters:
            base_filename (str): 기본 로그 파일 이름
            mode (str): 파일 열기 모드 (기본 'a' - append)
            max_lines (int): 하나의 로그 파일에 기록할 최대 라인 수
            encoding (str): 파일 인코딩 (기본값 None)
            delay (bool): 파일 열기를 지연할지 여부 (기본 False)
        """
        self.base_filename = base_filename
        self.current_index = 0  # 현재 파일 인덱스, 0이면 기본 파일 사용
        self._set_current_filename()  # 현재 파일 이름 설정
        super().__init__(self.current_filename, mode, maxBytes=0, encoding=encoding, delay=delay)
        self.max_lines = max_lines  # 최대 라인 수 설정
        self.current_line_count = 0  # 현재까지 기록한 라인 수

    def _set_current_filename(self):
        """
        현재 인덱스에 따라 현재 로그 파일 이름을 설정합니다.
        """
        base, ext = os.path.splitext(self.base_filename)
        # 인덱스 0이면 기본 파일명, 그 외에는 인덱스 번호를 포함한 파일명 사용
        self.current_filename = self.base_filename if self.current_index == 0 else f"{base}{self.current_index}{ext}"
        self.baseFilename = os.path.abspath(self.current_filename)

    def doRollover(self):
        """
        로그 파일 롤오버를 수행합니다.
        
        주요 동작:
          - 기존 스트림을 닫고, 파일 인덱스를 증가시킨 후 새로운 파일로 전환
          - 라인 카운트를 초기화하여 새 파일에서 새로 기록 시작
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        self.current_index += 1  # 파일 인덱스 증가
        self._set_current_filename()  # 새로운 파일 이름 설정
        self.mode = 'w'  # 새 파일은 쓰기 모드로 열림
        self.stream = self._open()  # 새로운 파일 스트림 열기
        self.current_line_count = 0  # 라인 카운트 초기화

    def emit(self, record):
        """
        로그 레코드를 기록하기 전에 메시지의 라인 수를 체크하고,
        최대 라인 수를 초과하면 롤오버를 수행한 후 기록합니다.
        
        Parameters:
            record (logging.LogRecord): 기록할 로그 이벤트
        
        주요 동작:
          - 로그 메시지를 포맷팅하여 몇 줄인지 확인
          - 현재 기록된 라인 수와 합산해 최대치를 초과하면 doRollover() 호출
          - 업데이트된 라인 수를 반영 후 상위 emit() 메서드로 메시지 기록
        """
        try:
            if record.levelno < FILE_LOG_LEVEL:
                return  # 설정된 레벨보다 낮은 메시지는 무시
            msg = self.format(record)  # 로그 메시지 포맷팅
            lines_in_msg = msg.count("\n") or 1  # 메시지 내 줄바꿈 수 계산 (없으면 1라인)
            if self.current_line_count + lines_in_msg > self.max_lines:
                self.doRollover()  # 최대 라인 수 초과 시 롤오버 수행
            self.current_line_count += lines_in_msg  # 현재 라인 수 업데이트
            super().emit(record)  # 상위 클래스의 emit() 호출하여 로그 기록
        except Exception:
            self.handleError(record)

# 전역 변수: 로그 메시지 큐 (크기 무제한)
log_queue = queue.Queue(-1)
queue_listener = None  # 전역 변수: QueueListener 객체 (추후 초기화)

def initialize_root_logger():
    """
    루트 로거를 초기화하여 파일 핸들러, 콘솔 핸들러, 큐 핸들러 및 AggregatingHandler를 추가합니다.
    
    주요 동작:
      - 루트 로거의 레벨 설정 및 기존 핸들러 제거
      - 로그 파일용 핸들러와 콘솔용 핸들러 생성 및 설정
      - 큐 핸들러를 통해 비동기 로깅 구성
      - 외부 라이브러리(ccxt, urllib3)의 로그 레벨을 경고(WARNING)로 설정
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(detail_level)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 파일 핸들러 생성: 로그 파일은 LOG_FILES_DIR 내에 생성됨
    file_handler = LineRotatingFileHandler(
        base_filename=BASE_LOG_FILE,
        max_lines=500,
        encoding="utf-8",
        delay=True
    )
    file_handler.setLevel(FILE_LOG_LEVEL)
    formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_handler.setLevel(detail_level)
    console_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    # _is_summary 속성이 있는 레코드는 필터링(제외)
    console_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))

    # 큐 핸들러 추가: 로그 메시지를 큐에 넣어 비동기 처리 지원
    q_handler = QueueHandler(log_queue)
    root_logger.addHandler(q_handler)

    global queue_listener
    queue_listener = QueueListener(log_queue, console_handler)
    queue_listener.start()

    # AggregatingHandler 추가: 로그 이벤트 집계를 위해 설정
    try:
        aggregator_handler = AggregatingHandler(level=detail_level)
        aggregator_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
        aggregator_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        aggregator_handler.setFormatter(aggregator_formatter)
        root_logger.addHandler(aggregator_handler)
    except Exception as e:
        logging.getLogger().error("Failed to add AggregatingHandler: " + str(e), exc_info=True)

    # 외부 라이브러리 로그 레벨 조정 (불필요한 디버그 메시지 제거)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def setup_logger(module_name: str) -> logging.Logger:
    """
    모듈별 로거를 설정하여 반환합니다.
    
    Parameters:
        module_name (str): 해당 모듈을 식별하기 위한 이름.
    
    Returns:
        logging.Logger: 설정된 로거 객체.
    
    주요 동작:
      - 모듈 이름에 따라 로거 생성 후, 모듈 전용 AggregatingHandler를 추가하여 집계 기능 제공.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(detail_level)
    logger.propagate = True  # 상위 로거로 메시지 전파
    try:
        agg_handler = AggregatingHandler(level=detail_level)
        agg_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
        formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        agg_handler.setFormatter(formatter)
        logger.addHandler(agg_handler)
    except Exception as e:
        logger.error("Failed to add module-specific AggregatingHandler: " + str(e), exc_info=True)
    return logger

def shutdown_logging():
    """
    로깅 시스템을 종료합니다.
    
    주요 동작:
      - 루트 로거에 추가된 핸들러 중 flush_aggregation_summary() 메서드가 있으면 호출하여 집계 결과 플러시
      - 큐 리스너를 중지한 후 logging.shutdown()을 호출하여 모든 로깅 자원 정리
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        try:
            if hasattr(handler, 'flush_aggregation_summary'):
                handler.flush_aggregation_summary()
        except Exception:
            pass
    global queue_listener
    if queue_listener is not None:
        queue_listener.stop()
        queue_listener = None
    logging.shutdown()
