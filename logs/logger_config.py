# logs/logger_config.py
import logging
import os
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기 (프로젝트 전체에 통일된 로그 레벨)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
_LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", None)
if _LOG_LEVEL_FROM_ENV:
    level = getattr(logging, _LOG_LEVEL_FROM_ENV.upper(), logging.INFO)
else:
    level = logging.INFO

# 기본 로그 파일 이름 (최초 로그는 이 파일에 기록됨)
BASE_LOG_FILE = os.path.join("logs", "project.log")

class OneLineFormatter(logging.Formatter):
    """
    로그 메시지 내 개행 문자를 제거하여 한 로그 이벤트가 한 줄로 기록되도록 합니다.
    """
    def format(self, record):
        formatted = super().format(record)
        return formatted.replace("\n", " | ")

class LineRotatingFileHandler(RotatingFileHandler):
    """
    지정된 라인 수(max_lines)를 초과하면 현재 로그 파일을 닫고,
    새로운 로그 파일을 'project.log', 'project1.log', 'project2.log', … 
    와 같이 순차적으로 생성하는 핸들러입니다.
    """
    def __init__(self, base_filename, mode='a', max_lines=1000, backupCount=7, encoding=None, delay=False):
        self.base_filename = base_filename
        self.current_index = 0
        self._set_current_filename()
        super().__init__(self.current_filename, mode, maxBytes=0, backupCount=backupCount, encoding=encoding, delay=delay)
        self.max_lines = max_lines
        self.current_line_count = 0

    def _set_current_filename(self):
        base, ext = os.path.splitext(self.base_filename)
        if self.current_index == 0:
            self.current_filename = self.base_filename
        else:
            self.current_filename = f"{base}{self.current_index}{ext}"
        self.baseFilename = os.path.abspath(self.current_filename)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        self.current_index += 1
        if self.backupCount > 0 and self.current_index >= self.backupCount:
            base, ext = os.path.splitext(self.base_filename)
            oldest_index = self.current_index - self.backupCount
            if oldest_index == 0:
                old_file = self.base_filename
            else:
                old_file = f"{base}{oldest_index}{ext}"
            if os.path.exists(old_file):
                os.remove(old_file)
        self._set_current_filename()
        self.mode = 'w'
        self.stream = self._open()
        self.current_line_count = 0

    def emit(self, record):
        try:
            msg = self.format(record)
            lines_in_msg = msg.count("\n") or 1
            if self.current_line_count + lines_in_msg > self.max_lines:
                self.doRollover()
            self.current_line_count += lines_in_msg
            super().emit(record)
        except Exception:
            self.handleError(record)

# 전역 AggregatingHandler를 추가합니다.
try:
    from logs.aggregating_handler import AggregatingHandler
except ImportError:
    AggregatingHandler = None

def initialize_root_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    if not any(isinstance(handler, LineRotatingFileHandler) for handler in root_logger.handlers):
        file_handler = LineRotatingFileHandler(
            base_filename=BASE_LOG_FILE,
            max_lines=1000,
            backupCount=7,
            encoding="utf-8",
            delay=True
        )
        file_handler.setLevel(level)
        formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    if AggregatingHandler is not None and not any(isinstance(handler, AggregatingHandler) for handler in root_logger.handlers):
        global_threshold_str = os.getenv("AGG_THRESHOLD_GLOBAL")
        try:
            global_threshold = int(global_threshold_str) if global_threshold_str and global_threshold_str.isdigit() else 2000
        except Exception:
            global_threshold = 2000
        aggregator_handler = AggregatingHandler(threshold=global_threshold, level=level)
        aggregator_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
        aggregator_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        aggregator_handler.setFormatter(aggregator_formatter)
        root_logger.addHandler(aggregator_handler)
    
    # 민감도 분석 관련 로그 집계를 위한 sensitivity AggregatingHandler 추가
    sensitivity_threshold_str = os.getenv("AGG_THRESHOLD_SENSITIVITY")
    if sensitivity_threshold_str and sensitivity_threshold_str.isdigit():
        sensitivity_threshold = int(sensitivity_threshold_str)
        sensitivity_handler = AggregatingHandler(threshold=global_threshold, level=level, sensitivity_threshold=sensitivity_threshold)
        sensitivity_handler.addFilter(lambda record: getattr(record, "sensitivity_analysis", False))
        sensitivity_formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
        sensitivity_handler.setFormatter(sensitivity_formatter)
        root_logger.addHandler(sensitivity_handler)
    
initialize_root_logger()

def setup_logger(module_name: str) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    mod_key = module_name.split('.')[-1].upper()
    env_threshold = os.getenv(f"AGG_THRESHOLD_{mod_key}")
    if env_threshold is not None and env_threshold.isdigit():
        threshold = int(env_threshold)
        try:
            agg_handler = AggregatingHandler(threshold=threshold, level=level, module_name=mod_key)
            agg_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
            formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
            agg_handler.setFormatter(formatter)
            logger.addHandler(agg_handler)
            logger.propagate = False
        except Exception as e:
            logger.error("모듈별 AggregatingHandler 추가 실패: %s", e)
    else:
        logger.propagate = True
    return logger
