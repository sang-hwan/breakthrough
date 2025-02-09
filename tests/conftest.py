# tests/conftest.py
import os
import glob
import pytest

@pytest.fixture(autouse=True, scope="session")
def clear_logs():
    """
    테스트 실행 전에 logs 디렉토리 내 모든 .log 파일을 삭제합니다.
    이 fixture는 세션 전체에 대해 한 번 실행됩니다.
    """
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    log_pattern = os.path.join(log_dir, "*.log")
    log_files = glob.glob(log_pattern)
    for log_file in log_files:
        try:
            os.remove(log_file)
            print(f"Deleted log file: {log_file}")
        except Exception as e:
            print(f"Failed to delete {log_file}: {e}")
