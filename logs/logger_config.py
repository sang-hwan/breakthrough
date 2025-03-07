# logs/logger_config.py
import logging
import os
import queue
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from logs.aggregating_handler import AggregatingHandler

FILE_LOG_LEVEL = logging.INFO         # 파일 로그는 INFO 이상
detail_level = logging.DEBUG           # 콘솔 로그는 DEBUG 이상

LOG_FILES_DIR = os.path.join("logs", "log_files")
if not os.path.exists(LOG_FILES_DIR):
    os.makedirs(LOG_FILES_DIR)
BASE_LOG_FILE = os.path.join(LOG_FILES_DIR, "project.log")

class OneLineFormatter(logging.Formatter):
    def format(self, record):
        formatted = super().format(record)
        return formatted.replace("\n", " | ")

class LineRotatingFileHandler(RotatingFileHandler):
    def __init__(self, base_filename, mode='a', max_lines=500, encoding=None, delay=False):
        self.base_filename = base_filename
        self.current_index = 0
        self._set_current_filename()
        super().__init__(self.current_filename, mode, maxBytes=0, encoding=encoding, delay=delay)
        self.max_lines = max_lines
        self.current_line_count = 0

    def _set_current_filename(self):
        base, ext = os.path.splitext(self.base_filename)
        self.current_filename = self.base_filename if self.current_index == 0 else f"{base}{self.current_index}{ext}"
        self.baseFilename = os.path.abspath(self.current_filename)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.current_index += 1
        self._set_current_filename()
        self.mode = 'w'
        self.stream = self._open()
        self.current_line_count = 0

    def emit(self, record):
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

log_queue = queue.Queue(-1)
queue_listener = None

def initialize_root_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(detail_level)

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 파일 핸들러 (로그 파일은 log_files 디렉토리에 생성됨)
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

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(detail_level)
    console_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))

    q_handler = QueueHandler(log_queue)
    root_logger.addHandler(q_handler)

    global queue_listener
    queue_listener = QueueListener(log_queue, console_handler)
    queue_listener.start()

    # AggregatingHandler 추가
    try:
        aggregator_handler = AggregatingHandler(level=detail_level)
        aggregator_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
        aggregator_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        aggregator_handler.setFormatter(aggregator_formatter)
        root_logger.addHandler(aggregator_handler)
    except Exception as e:
        logging.getLogger().error("Failed to add AggregatingHandler: " + str(e), exc_info=True)

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def setup_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(detail_level)
    logger.propagate = True
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
