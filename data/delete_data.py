# data/delete_data.py
"""
이 모듈은 PostgreSQL의 public 스키마 내 모든 테이블을 삭제하는 기능을 제공합니다.
테스트 환경 초기화 시, 스크립트가 아니라 별도의 호출(예: scripts 모듈에서)을 통해 실행됩니다.
"""

from sqlalchemy import text
from typing import Dict, Any
from data.db_config import DATABASE
from data.data_utils import create_db_engine
from logs.log_config import setup_logger

logger = setup_logger(__name__)

def drop_all_tables(db_config: Dict[str, Any] = DATABASE) -> None:
    """
    데이터베이스 내의 모든 테이블을 삭제합니다.
    
    Parameters:
        db_config (dict): 데이터베이스 접속 정보.
        
    Returns:
        None
    """
    try:
        engine = create_db_engine(db_config)
        with engine.begin() as conn:
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = [row[0] for row in result]
            if not tables:
                logger.debug("No tables found in the database.")
                return
            for table in tables:
                logger.debug(f"Dropping table {table}...")
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            logger.info("All tables dropped successfully.")
    except Exception as e:
        logger.error(f"Error dropping tables: {e}", exc_info=True)
        raise
