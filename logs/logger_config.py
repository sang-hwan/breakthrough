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
        # maxBytes=0으로 설정하여 크기 기반 롤오버 대신 라인 수 기반 롤오버를 구현합니다.
        super().__init__(self.current_filename, mode, maxBytes=0, backupCount=backupCount, encoding=encoding, delay=delay)
        self.max_lines = max_lines
        # 항상 current_line_count를 0으로 초기화하여 이전 로그 파일 내용의 영향을 받지 않도록 수정함.
        self.current_line_count = 0

    def _set_current_filename(self):
        """
        현재 인덱스에 따라 로그 파일 이름을 결정하고, self.baseFilename을 업데이트합니다.
        """
        base, ext = os.path.splitext(self.base_filename)
        if self.current_index == 0:
            self.current_filename = self.base_filename
        else:
            self.current_filename = f"{base}{self.current_index}{ext}"
        self.baseFilename = os.path.abspath(self.current_filename)

    def doRollover(self):
        """
        현재 로그 파일의 라인 수가 max_lines를 초과하면 호출됩니다.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        self.current_index += 1
        # backupCount 이상의 파일이 쌓이면 가장 오래된 로그 파일(프로젝트 로그 포함)을 삭제합니다.
        if self.backupCount > 0 and self.current_index >= self.backupCount:
            base, ext = os.path.splitext(self.base_filename)
            oldest_index = self.current_index - self.backupCount
            # oldest_index가 0이면 base_filename(project.log)이 대상이 됩니다.
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
    """
    루트 로거에 전역 파일 핸들러, 콘솔 핸들러, 그리고 AggregatingHandler를 한 번만 추가합니다.
    모든 모듈은 propagate를 통해 이 핸들러들을 상속받아 통일된 로그 환경에서 동작합니다.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 파일 핸들러 추가 (중복 추가 방지)
    if not any(isinstance(handler, LineRotatingFileHandler) for handler in root_logger.handlers):
        file_handler = LineRotatingFileHandler(
            base_filename=BASE_LOG_FILE,
            max_lines=1000,
            backupCount=7,
            encoding="utf-8",
            delay=True
        )
        file_handler.setLevel(level)
        formatter = OneLineFormatter(
            '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 콘솔 핸들러 추가 (중복 추가 방지)
    if not any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = OneLineFormatter(
            '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # AggregatingHandler 추가 (중복 추가 방지)
    if AggregatingHandler is not None and not any(isinstance(handler, AggregatingHandler) for handler in root_logger.handlers):
        # 전역 로그 레벨과 기본 threshold(1000)를 사용하여 AggregatingHandler를 생성합니다.
        # 단, 환경 변수 AGG_THRESHOLD_GLOBAL 가 있으면 이를 사용합니다.
        global_threshold = os.getenv("AGG_THRESHOLD_GLOBAL")
        try:
            threshold_value = int(global_threshold) if global_threshold and global_threshold.isdigit() else 1000
        except Exception:
            threshold_value = 1000

        aggregator_handler = AggregatingHandler(threshold=threshold_value, level=level)
        # 이미 요약된 로그는 집계하지 않도록 필터 추가
        aggregator_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
        aggregator_formatter = OneLineFormatter(
            '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s'
        )
        aggregator_handler.setFormatter(aggregator_formatter)
        root_logger.addHandler(aggregator_handler)

# 루트 로거 초기화 (이 시점부터 모든 모듈은 루트 로거의 핸들러를 상속받습니다)
initialize_root_logger()

def setup_logger(module_name: str) -> logging.Logger:
    """
    모듈 이름을 기반으로 로거를 반환합니다.
    모든 모듈은 전역 로그 레벨과 핸들러(파일, 콘솔, AggregatingHandler)를 상속받아 통일된 로그 환경에서 동작합니다.
    
    추가로, 만약 환경 변수 AGG_THRESHOLD_<모듈명> (모듈명의 마지막 부분을 대문자로)이 설정되어 있다면,
    해당 로거에는 별도의 AggregatingHandler를 추가하여 모듈별 임계치를 적용합니다.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    # 기본적으로 전역 핸들러에 의해 처리되지만, 모듈별 임계치를 사용하고자 하는 경우 별도의 AggregatingHandler를 추가합니다.
    mod_key = module_name.split('.')[-1].upper()  # 예: "backtester" → "BACKTESTER"
    env_threshold = os.getenv(f"AGG_THRESHOLD_{mod_key}")
    if env_threshold is not None and env_threshold.isdigit():
        threshold = int(env_threshold)
        # 새 AggregatingHandler를 해당 로거에 추가합니다.
        try:
            from logs.aggregating_handler import AggregatingHandler
            agg_handler = AggregatingHandler(threshold=threshold, level=level, module_name=mod_key)
            agg_handler.addFilter(lambda record: not getattr(record, '_is_summary', False))
            formatter = OneLineFormatter('[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s')
            agg_handler.setFormatter(formatter)
            logger.addHandler(agg_handler)
            # 전역 핸들러와 중복되지 않도록 해당 로거는 propagate를 False로 설정할 수 있습니다.
            logger.propagate = False
        except Exception as e:
            logger.error("모듈별 AggregatingHandler 추가 실패: %s", e)
    else:
        logger.propagate = True
    return logger
