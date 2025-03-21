# logs/log_config.py
import logging
import os
import queue
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener

# 전역 상수 설정
FILE_LOG_LEVEL = logging.INFO       # 파일에는 INFO 이상의 로그 기록
CONSOLE_LOG_LEVEL = logging.DEBUG    # 콘솔에는 DEBUG 이상의 로그 기록

# 로그 파일 저장 디렉토리 및 기본 파일 경로 설정
LOG_FILES_DIR = os.path.join("logs", "log_files")
if not os.path.exists(LOG_FILES_DIR):
    os.makedirs(LOG_FILES_DIR)
BASE_LOG_FILE = os.path.join(LOG_FILES_DIR, "project.log")


class OneLineFormatter(logging.Formatter):
    """
    로그 메시지 내의 줄바꿈을 파이프(|) 문자로 대체하여 한 줄로 출력하도록 포맷팅합니다.
    """
    def format(self, record):
        formatted = super().format(record)
        return formatted.replace("\n", " | ")


class LineRotatingFileHandler(RotatingFileHandler):
    """
    로그 파일의 최대 라인 수를 초과하면 새로운 파일로 롤오버하는 파일 핸들러입니다.
    
    Attributes:
        base_filename (str): 기본 로그 파일 이름
        max_lines (int): 하나의 로그 파일에 기록할 최대 라인 수
    """
    def __init__(self, base_filename, mode='a', max_lines=500, encoding=None, delay=False):
        self.base_filename = base_filename
        self.current_index = 0
        self._set_current_filename()
        super().__init__(self.current_filename, mode, maxBytes=0, encoding=encoding, delay=delay)
        self.max_lines = max_lines
        self.current_line_count = 0

    def _set_current_filename(self):
        """
        현재 인덱스에 따라 로그 파일 이름을 설정합니다.
        """
        base, ext = os.path.splitext(self.base_filename)
        self.current_filename = self.base_filename if self.current_index == 0 else f"{base}_{self.current_index}{ext}"
        self.baseFilename = os.path.abspath(self.current_filename)

    def doRollover(self):
        """
        로그 파일 롤오버를 수행합니다.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        self.current_index += 1
        self._set_current_filename()
        self.mode = 'w'
        self.stream = self._open()
        self.current_line_count = 0

    def emit(self, record):
        """
        로그 레코드를 기록하기 전에 메시지의 라인 수를 체크하고,
        최대 라인 수를 초과하면 롤오버 후 기록합니다.
        """
        try:
            if record.levelno < FILE_LOG_LEVEL:
                return
            msg = self.format(record)
            lines_in_msg = msg.count("\n") or 1
            if self.current_line_count + lines_in_msg > self.max_lines:
                self.doRollover()
            self.current_line_count += lines_in_msg
            super().emit(record)
        except Exception:
            self.handleError(record)


class AggregatingHandler(logging.Handler):
    """
    (logger 이름, 파일명, 함수명)별로 로그 이벤트 발생 횟수를 집계하는 핸들러입니다.
    집계 결과는 flush_aggregation_summary() 호출 시 최종 로그로 출력됩니다.
    """
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)
        self.aggregation = {}

    def emit(self, record):
        try:
            key = (record.name, record.filename, record.funcName)
            self.aggregation[key] = self.aggregation.get(key, 0) + 1
        except Exception:
            self.handleError(record)

    def flush_aggregation_summary(self):
        if self.aggregation:
            summary = "\n".join([
                f"{k[1]}:{k[2]} (logger: {k[0]}) - {v}회 발생"
                for k, v in self.aggregation.items()
            ])
            logging.getLogger().info("누적 로그 집계 요약:\n" + summary)
            self.aggregation.clear()


# 전역 로그 큐와 큐 리스너
log_queue = queue.Queue(-1)
queue_listener = None


def initialize_root_logger():
    """
    루트 로거를 초기화합니다.
    
    주요 동작:
      - 파일 핸들러: LOG_FILES_DIR 내의 파일에 INFO 레벨 이상의 메시지를 기록.
      - 콘솔 핸들러: 콘솔에 DEBUG 레벨 이상의 메시지를 출력.
      - 큐 핸들러를 통해 비동기 로깅 구성.
      - AggregatingHandler를 추가하여 로그 이벤트 집계 기능 제공.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(CONSOLE_LOG_LEVEL)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 파일 핸들러 생성
    file_handler = LineRotatingFileHandler(
        base_filename=BASE_LOG_FILE,
        max_lines=500,
        encoding="utf-8",
        delay=True
    )
    file_handler.setLevel(FILE_LOG_LEVEL)
    formatter = OneLineFormatter(
        '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_formatter = OneLineFormatter(
        '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # 큐 핸들러 추가 (비동기 로깅)
    q_handler = QueueHandler(log_queue)
    root_logger.addHandler(q_handler)

    global queue_listener
    queue_listener = QueueListener(log_queue, console_handler)
    queue_listener.start()

    # AggregatingHandler 추가
    try:
        agg_handler = AggregatingHandler(level=CONSOLE_LOG_LEVEL)
        agg_handler.setFormatter(formatter)
        root_logger.addHandler(agg_handler)
    except Exception as e:
        logging.getLogger().error("AggregatingHandler 추가 실패: " + str(e), exc_info=True)

    # 외부 라이브러리 로그 레벨 설정 (불필요한 디버그 메시지 제거)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def setup_logger(module_name: str) -> logging.Logger:
    """
    모듈별 로거를 설정하여 반환합니다.
    
    Parameters:
        module_name (str): 모듈을 식별할 이름.
    
    Returns:
        logging.Logger: 설정된 모듈 전용 로거.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(CONSOLE_LOG_LEVEL)
    logger.propagate = True  # 상위 로거로 메시지 전파
    try:
        agg_handler = AggregatingHandler(level=CONSOLE_LOG_LEVEL)
        formatter = OneLineFormatter(
            '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s'
        )
        agg_handler.setFormatter(formatter)
        logger.addHandler(agg_handler)
    except Exception as e:
        logger.error("모듈별 AggregatingHandler 추가 실패: " + str(e), exc_info=True)
    return logger


def shutdown_logging():
    """
    로깅 시스템을 종료합니다.
    
    주요 동작:
      - 모든 핸들러의 flush_aggregation_summary() 메서드를 호출하여 집계 요약 플러시.
      - 큐 리스너를 중지한 후 logging.shutdown() 호출.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if hasattr(handler, 'flush_aggregation_summary'):
            try:
                handler.flush_aggregation_summary()
            except Exception:
                pass
    global queue_listener
    if queue_listener is not None:
        queue_listener.stop()
        queue_listener = None
    logging.shutdown()
