# data_collection/db_manager.py
import logging
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
import pandas as pd
from data_collection.db_config import DATABASE
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def insert_on_conflict(table, conn, keys, data_iter):
    """
    데이터를 삽입할 때, timestamp 컬럼을 기준으로 중복 발생 시 삽입하지 않습니다.
    pandas.to_sql() 의 method 인자로 사용되며, 각 chunk 단위로 호출됩니다.
    """
    try:
        raw_conn = conn.connection
        cur = raw_conn.cursor()
        values = list(data_iter)
        columns = ", ".join(keys)
        sql = f"INSERT INTO {table.name} ({columns}) VALUES %s ON CONFLICT (timestamp) DO NOTHING"
        execute_values(cur, sql, values)
        cur.close()
        # 각 청크 삽입이 완료되면 DEBUG 레벨 로그로 기록하여, 대용량 데이터 처리 시 불필요한 INFO 로그를 줄임.
        logger.debug(f"[DB] insert_on_conflict: {len(values)} records processed for table {table.name}")
    except Exception as e:
        logger.error(f"insert_on_conflict 에러: {e}", exc_info=True)

def insert_ohlcv_records(df: pd.DataFrame, table_name: str = 'ohlcv_data', conflict_action: str = "DO NOTHING", db_config: dict = None, chunk_size: int = 10000) -> None:
    """
    OHLCV 데이터를 지정된 테이블에 저장합니다.
    - 대용량 데이터 처리를 위해 chunk_size 단위로 나누어 저장합니다.
    - 저장 성공 시 총 행수를 INFO 레벨 로그로 남기며, 에러 발생 시 ERROR 레벨로 기록합니다.
    """
    if db_config is None:
        db_config = DATABASE

    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
        pool_pre_ping=True
    )

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
        df = df.copy()
        df.reset_index(inplace=True)
        # 대용량 데이터를 chunk 단위로 저장 (각 청크 관련 로그는 insert_on_conflict에서 DEBUG 레벨로 처리됨)
        df.to_sql(
            table_name,
            engine,
            if_exists='append',
            index=False,
            method=insert_on_conflict,
            chunksize=chunk_size
        )
        logger.info(f"데이터 저장 완료: {table_name} (총 {len(df)} 행)")
    except Exception as e:
        logger.error(f"데이터 저장 에러 ({table_name}): {e}", exc_info=True)

def fetch_ohlcv_records(table_name: str = 'ohlcv_data', start_date: str = None, end_date: str = None, db_config: dict = None) -> pd.DataFrame:
    """
    지정된 테이블에서 OHLCV 데이터를 읽어옵니다.
    - 에러 발생 시 빈 DataFrame을 반환하며, 상세 에러 내용을 ERROR 레벨로 기록합니다.
    """
    if db_config is None:
        db_config = DATABASE

    try:
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            pool_pre_ping=True
        )
    except Exception as e:
        logger.error(f"DB 엔진 생성 에러: {e}", exc_info=True)
        return pd.DataFrame()

    query = f"SELECT * FROM {table_name} WHERE 1=1"
    params = {}
    if start_date:
        query += " AND timestamp >= :start_date"
        params['start_date'] = start_date
    if end_date:
        query += " AND timestamp <= :end_date"
        params['end_date'] = end_date
    query += " ORDER BY timestamp"
    query = text(query)
    try:
        df = pd.read_sql(query, engine, params=params, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"데이터 로드 완료: {table_name} (총 {len(df)} 행)")
        return df
    except Exception as e:
        logger.error(f"데이터 로드 에러 ({table_name}): {e}", exc_info=True)
        return pd.DataFrame()
