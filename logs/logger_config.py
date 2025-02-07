# logs/logger_config.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler

def setup_logger(module_name: str) -> logging.Logger:
    """
    - 기본 로그 레벨을 INFO로 설정하여 핵심 이벤트, 오류, 경고 등의 정보만 출력하도록 합니다.
    - 개발 단계에서는 환경변수 LOG_LEVEL을 통해 필요 시 DEBUG 레벨로 전환할 수 있습니다.
    - TimedRotatingFileHandler를 사용해 매일(또는 지정 시간마다) 새 로그 파일로 분리하며, 최대 7일 분량만 보관합니다.
    - 로그 메시지 생략 기능은 제거되어, 모든 로그 메시지가 온전히 출력됩니다.
    """
    # 환경변수를 통해 로그 레벨 설정 (기본은 INFO)
    log_level_str = os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # 로그 저장 폴더 생성 (존재하지 않을 경우)
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    
    # 기존 핸들러가 있다면 제거 (중복 로그 방지)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 파일 핸들러 설정: 매일 자정마다 로그 파일 롤링, 최대 7일 보관
    file_handler = TimedRotatingFileHandler(
        filename=f"logs/{module_name}.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(funcName)s: %(message)s')
    file_handler.setFormatter(formatter)
    # 메시지 생략 필터 제거: 별도의 필터를 추가하지 않습니다.
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러 설정 (개발 중 필요 시)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    # 메시지 생략 필터 제거: 별도의 필터를 추가하지 않습니다.
    logger.addHandler(console_handler)
    
    return logger
