# data/data_utils.py
"""
데이터베이스 연결 및 공통 유틸리티 함수를 제공합니다.
"""

from sqlalchemy import create_engine
from typing import Dict, Any
from data.db_config import DATABASE

def create_db_engine(db_config: Dict[str, Any] = DATABASE) -> Any:
    """
    주어진 데이터베이스 설정으로 SQLAlchemy 엔진을 생성합니다.

    Parameters:
        db_config (dict): 데이터베이스 접속 정보

    Returns:
        SQLAlchemy Engine 객체
    """
    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
        pool_pre_ping=True
    )
    return engine
