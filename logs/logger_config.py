# logs/logger_config.py
import logging
import os
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
_LOG_LEVEL_FROM_ENV = os.getenv("LOG_LEVEL", None)

if _LOG_LEVEL_FROM_ENV:
    level = getattr(logging, _LOG_LEVEL_FROM_ENV.upper(), logging.INFO)
else:
    level = logging.INFO

# 기본 로그 파일 이름 (최초 로그는 이 파일에 기록됨)
BASE_LOG_FILE = os.path.join("logs", "project.log")

# 전역 파일 핸들러 인스턴스 (모듈 간 공유)
_global_file_handler = None

class LineRotatingFileHandler(RotatingFileHandler):
    """
    지정된 라인 수(max_lines)를 초과하면 현재 로그 파일을 닫고,
    새로운 로그 파일을 'project.log', 'project1.log', 'project2.log', … 
    와 같이 순차적으로 생성하는 핸들러입니다.
    
    - 최초 로그는 BASE_LOG_FILE (즉, project.log)에 기록됩니다.
    - rollover 시 내부 카운터를 증가시켜 새 로그 파일 이름을 생성합니다.
    - backupCount가 지정되면, 오래된 로그 파일은 삭제합니다.
    """
    def __init__(self, base_filename, mode='a', max_lines=1000, backupCount=7, encoding=None, delay=False):
        self.base_filename = base_filename
        # current_index 0이면 파일명은 base_filename, 0보다 크면 base + str(current_index) + ext
        self.current_index = 0
        self._set_current_filename()
        # _global_file_handler나 부모 초기화 시점에서 사용하는 filename을 현재 파일명으로 지정
        super().__init__(self.current_filename, mode, maxBytes=0, backupCount=backupCount, encoding=encoding, delay=delay)
        self.max_lines = max_lines
        # 기존 파일이 존재하면 현재 줄 수를 계산
        if os.path.exists(self.current_filename):
            try:
                with open(self.current_filename, 'r', encoding=encoding) as f:
                    self.current_line_count = sum(1 for _ in f)
            except Exception:
                self.current_line_count = 0
        else:
            self.current_line_count = 0

    def _set_current_filename(self):
        """현재 인덱스에 따라 로그 파일 이름을 결정하고, self.baseFilename을 업데이트합니다."""
        base, ext = os.path.splitext(self.base_filename)
        if self.current_index == 0:
            self.current_filename = self.base_filename
        else:
            self.current_filename = f"{base}{self.current_index}{ext}"
        self.baseFilename = os.path.abspath(self.current_filename)

    def doRollover(self):
        """
        현재 로그 파일의 라인 수가 max_lines를 초과하면 호출됩니다.
        내부 인덱스를 1 증가시키고, 새 로그 파일(현재 인덱스에 해당하는 파일)을 생성합니다.
        또한 backupCount를 초과하는 경우 가장 오래된 로그 파일을 삭제합니다.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # 증가된 인덱스를 적용하여 새 로그 파일명을 설정
        self.current_index += 1

        # backupCount를 초과하면, (현재 인덱스 - backupCount) 번호의 파일 삭제
        if self.backupCount > 0 and self.current_index > self.backupCount:
            base, ext = os.path.splitext(self.base_filename)
            # 만약 (current_index - backupCount)가 0이면 기본 파일명, 아니면 번호가 붙은 이름
            oldest_index = self.current_index - self.backupCount
            old_file = self.base_filename if oldest_index == 0 else f"{base}{oldest_index}{ext}"
            if os.path.exists(old_file):
                os.remove(old_file)

        # 설정된 인덱스에 따라 새 로그 파일명을 적용
        self._set_current_filename()
        # 새 로그 파일을 열고, 현재 줄 수를 0으로 초기화
        self.mode = 'w'
        self.stream = self._open()
        self.current_line_count = 0

    def emit(self, record):
        try:
            msg = self.format(record)
            # 메시지에 포함된 줄바꿈 수를 카운트(없으면 1줄로 간주)
            lines_in_msg = msg.count("\n") or 1
            self.current_line_count += lines_in_msg
            if self.current_line_count >= self.max_lines:
                self.doRollover()
                # 새 파일에 기록될 때 이미 msg의 줄 수를 포함하므로 초기화
                self.current_line_count = lines_in_msg
            super().emit(record)
        except Exception:
            self.handleError(record)

def get_global_file_handler():
    """
    전역 파일 핸들러를 생성(최초 호출 시)하거나 이미 생성된 핸들러를 반환합니다.
    delay=True 옵션을 사용하여 실제 로그 기록 시에만 파일을 열도록 합니다.
    """
    global _global_file_handler
    if _global_file_handler is None:
        _global_file_handler = LineRotatingFileHandler(
            base_filename=BASE_LOG_FILE,
            max_lines=1000,
            backupCount=7,
            encoding="utf-8",
            delay=True  # 실제 로그 기록 시에만 파일이 열림
        )
        _global_file_handler.setLevel(level)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s'
        )
        _global_file_handler.setFormatter(formatter)
    return _global_file_handler

def setup_logger(module_name: str) -> logging.Logger:
    """
    모듈 이름을 기반으로 로거를 반환합니다.
    이미 해당 로거에 핸들러가 등록되어 있다면(즉, 이전에 설정된 경우)
    추가 등록 없이 기존 핸들러를 그대로 사용합니다.
    모든 로거는 전역 파일 핸들러(하나의 인스턴스)를 공유하며, 콘솔 핸들러도 함께 추가합니다.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # 전역 파일 핸들러 추가
        logger.addHandler(get_global_file_handler())
        
        # 콘솔 핸들러 생성 및 추가
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(funcName)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    logger.propagate = False
    return logger
