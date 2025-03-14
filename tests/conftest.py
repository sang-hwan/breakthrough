# tests/conftest.py
# 이 모듈은 테스트 실행 전후에 로깅 환경을 관리하는 설정들을 포함합니다.
# 기존 로그 파일을 삭제하고 루트 로거를 초기화한 후, 테스트 종료 시 로깅을 종료합니다.

import os  # 파일 및 경로 관련 작업을 위한 모듈
import glob  # 파일 패턴 검색용 모듈
import logging  # 로깅 기능 제공
import pytest  # 테스트 프레임워크용
from logs.logger_config import initialize_root_logger, shutdown_logging  # 로거 초기화 및 종료 함수

@pytest.fixture(autouse=True, scope="session")
def manage_logs():
    """
    테스트 세션 전반에 걸쳐 로깅을 관리하는 fixture입니다.
    
    테스트 실행 전에 로그 디렉토리 내의 모든 로그 파일을 삭제하고, 루트 로거를 초기화합니다.
    테스트 종료 후에는 로깅 시스템을 종료합니다.
    
    Yields:
        None: 이 fixture는 부수 효과(로깅 관리)를 위해 사용됩니다.
    """
    # 현재 파일 기준 상위 logs 디렉토리 경로 설정
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    log_pattern = os.path.join(log_dir, "*.log")  # 로그 파일 패턴 지정
    log_files = glob.glob(log_pattern)  # 로그 파일 목록 검색
    # 각 로그 파일 삭제 시도
    for log_file in log_files:
        try:
            os.remove(log_file)
            print(f"Deleted log file: {log_file}")
        except Exception as e:
            print(f"Failed to delete {log_file}: {e}")
    
    logging.shutdown()  # 기존 로깅 시스템 종료
    initialize_root_logger()  # 새로운 로깅 설정 초기화
    
    yield  # 테스트 실행 대기
    
    shutdown_logging()  # 테스트 종료 후 로깅 종료
