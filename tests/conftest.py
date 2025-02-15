# tests/conftest.py
import os
import glob
import logging
import pytest
from logs.logger_config import initialize_root_logger, shutdown_logging

@pytest.fixture(autouse=True, scope="session")
def manage_logs():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    log_pattern = os.path.join(log_dir, "*.log")
    log_files = glob.glob(log_pattern)
    for log_file in log_files:
        try:
            os.remove(log_file)
            print(f"Deleted log file: {log_file}")
        except Exception as e:
            print(f"Failed to delete {log_file}: {e}")
    
    logging.shutdown()
    initialize_root_logger()
    
    yield
    
    shutdown_logging()
