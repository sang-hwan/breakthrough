# data/db/db_manager.py
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
import pandas as pd
from typing import Any, Iterable, Dict
from data.db.db_config import DATABASE
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def insert_on_conflict(table: Any, conn: Any, keys: list, data_iter: Iterable) -> None:
    """
    Custom insertion method for pandas to_sql that handles conflicts on the 'timestamp' column.
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

def insert_ohlcv_records(df: pd.DataFrame, table_name: str = 'ohlcv_data', conflict_action: str = "DO NOTHING",
                         db_config: Dict[str, Any] = None, chunk_size: int = 10000) -> None:
    """
    Insert OHLCV records into the specified table, creating the table if it doesn't exist.
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
        df.to_sql(
            table_name,
            engine,
            if_exists='append',
            index=False,
            method=insert_on_conflict,
            chunksize=chunk_size
        )
    except Exception as e:
        logger.error(f"데이터 저장 에러 ({table_name}): {e}", exc_info=True)

def fetch_ohlcv_records(table_name: str = 'ohlcv_data', start_date: str = None, end_date: str = None,
                        db_config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Fetch OHLCV records from the specified table within the given date range.
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
        return df
    except Exception as e:
        logger.error(f"데이터 로드 에러 ({table_name}): {e}", exc_info=True)
        return pd.DataFrame()

def get_unique_symbol_list(db_config: Dict[str, Any] = None) -> list:
    """
    DB 내 public 스키마에서 테이블 이름을 조회하고,
    "ohlcv_{symbol}_{timeframe}" 형식에 맞는 테이블들을 분석하여 고유 symbol (예: BTC/USDT)을 반환합니다.
    만약 db_config가 제공되지 않으면 기본 DATABASE 설정을 사용합니다.
    """
    if db_config is None:
        db_config = DATABASE
    try:
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            pool_pre_ping=True
        )
        with engine.connect() as conn:
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = [row[0] for row in result]
    except Exception as e:
        logger.error(f"Error fetching table names: {e}", exc_info=True)
        return []

    symbol_set = set()
    for table in tables:
        # 테이블 이름이 ohlcv_ 로 시작하면 형식: ohlcv_{symbol}_{timeframe} 임을 가정
        if table.startswith("ohlcv_"):
            parts = table.split("_")
            if len(parts) >= 3:
                symbol_key = parts[1]  # 예: 'btcusdt'
                symbol_set.add(symbol_key)

    # symbol_key를 표준 형식(BTC/USDT)으로 변환 (간단 규칙: 뒤에 'usdt'가 있으면)
    symbols = []
    for s in symbol_set:
        if s.endswith("usdt"):
            base = s[:-4].upper()
            symbol = f"{base}/USDT"
            symbols.append(symbol)
        else:
            symbols.append(s.upper())
    return list(sorted(symbols))
