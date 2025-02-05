# logs/logger_config.py
import logging
import os

def setup_logger(module_name: str) -> logging.Logger:
    # logs 디렉토리가 없으면 생성
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    # 이미 핸들러가 있으면 그대로 반환
    if logger.handlers:
        return logger
    # 파일 핸들러: 파일명에 모듈명을 포함, 인코딩을 utf-8로 설정
    file_handler = logging.FileHandler(f"logs/{module_name}.log", encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s:%(name)s:%(funcName)s: %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # 콘솔 핸들러도 추가 (인코딩은 시스템에 따라 자동 처리됨)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
