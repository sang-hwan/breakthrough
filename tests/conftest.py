# tests/conftest.py
import os
import glob
import logging
import pytest
from logs.logger_config import initialize_root_logger

@pytest.fixture(autouse=True, scope="session")
def clear_logs():
    """
    테스트 실행 전에 logs 디렉토리 내 모든 .log 파일을 삭제하고,
    기존 로거를 종료한 후 루트 로거를 재초기화합니다.
    이 fixture는 세션 전체에 대해 한 번 실행됩니다.
    """
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    log_pattern = os.path.join(log_dir, "*.log")
    log_files = glob.glob(log_pattern)
    for log_file in log_files:
        try:
            os.remove(log_file)
            print(f"Deleted log file: {log_file}")
        except Exception as e:
            print(f"Failed to delete {log_file}: {e}")
    
    # 기존 로거 종료 후, 최신 AggregatingHandler 설정이 반영되도록 루트 로거 재초기화
    logging.shutdown()
    initialize_root_logger()
