# data/store_data.py
"""
이 모듈은 수집된 OHLCV 데이터를 데이터베이스에 저장하는 기능만 제공합니다.
데이터 저장 시 중복 삽입 방지를 위해 'ON CONFLICT DO NOTHING' 전략을 사용합니다.
"""

from sqlalchemy import text
from psycopg2.extras import execute_values
import pandas as pd
from typing import Any, Iterable, Dict
from data.db_config import DATABASE
from data.data_utils import create_db_engine
from logs.log_config import setup_logger

logger = setup_logger(__name__)

def insert_on_conflict(table: Any, conn: Any, keys: list, data_iter: Iterable) -> None:
    """
    'timestamp' 컬럼에서 중복이 발생하면 아무 작업도 수행하지 않고 데이터를 삽입합니다.

    Parameters:
        table (Any): 대상 테이블 객체.
        conn (Any): SQLAlchemy 연결 객체.
        keys (list): 삽입할 컬럼 이름 리스트.
        data_iter (Iterable): 삽입할 데이터 목록.
    """
    try:
        raw_conn = conn.connection
        cur = raw_conn.cursor()
        values = list(data_iter)
        columns = ", ".join(keys)
        sql_str = f"INSERT INTO {table.name} ({columns}) VALUES %s ON CONFLICT (timestamp) DO NOTHING"
        execute_values(cur, sql_str, values)
        cur.close()
    except Exception as e:
        logger.error(f"insert_on_conflict 에러: {e}", exc_info=True)

def insert_ohlcv_records(
    df: pd.DataFrame,
    table_name: str = 'ohlcv_data',
    conflict_action: str = "DO NOTHING",
    db_config: Dict[str, Any] = None,
    chunk_size: int = 10000
) -> None:
    """
    OHLCV 데이터를 지정된 테이블에 삽입합니다.

    Parameters:
        df (pd.DataFrame): 삽입할 데이터.
        table_name (str): 대상 테이블 이름.
        conflict_action (str): 중복 처리 방식 (현재는 "DO NOTHING" 사용).
        db_config (dict): 데이터베이스 접속 정보.
        chunk_size (int): 한 번에 삽입할 행의 개수.
    """
    if db_config is None:
        db_config = DATABASE

    engine = create_db_engine(db_config)
    create_table_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp TIMESTAMP NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            PRIMARY KEY (timestamp)
        );
    """)
    try:
        with engine.begin() as conn:
            conn.execute(create_table_sql)
    except Exception as e:
        logger.error(f"테이블 생성 에러 ({table_name}): {e}", exc_info=True)
        return

    try:
        df_to_insert = df.copy()
        df_to_insert.reset_index(inplace=True)
        df_to_insert.to_sql(
            table_name,
            engine,
            if_exists='append',
            index=False,
            method=insert_on_conflict,
            chunksize=chunk_size
        )
    except Exception as e:
        logger.error(f"데이터 저장 에러 ({table_name}): {e}", exc_info=True)
