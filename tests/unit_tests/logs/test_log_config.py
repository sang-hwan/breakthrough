# tests/unit_tests/logs/test_log_config.py
import time
import logging
from logs.log_config import initialize_root_logger, setup_logger, shutdown_logging

def test_logging_setup():
    # 루트 로거 초기화
    initialize_root_logger()
    
    # 모듈 전용 로거 생성
    logger = setup_logger("test_module")
    
    # 다양한 레벨의 로그 기록
    logger.debug("테스트 디버그 메시지")
    logger.info("테스트 정보 메시지")
    logger.warning("테스트 경고 메시지")
    logger.error("테스트 에러 메시지")
    logger.critical("테스트 치명적 메시지")
    
    # 잠시 대기하여 비동기 로그가 처리되도록 함
    time.sleep(1)
    
    # 로깅 시스템 종료 (집계 요약 플러시 포함)
    shutdown_logging()
