# data_collection/db_ohlcv_manager.py

from sqlalchemy import create_engine
import psycopg2
import pandas as pd
from typing import Optional
from config.db_config import DATABASE

def insert_ohlcv_records(
    df: pd.DataFrame,
    table_name: str = 'ohlcv_data',
    conflict_action: str = "DO NOTHING",  # 중복키 충돌 시 액션
    db_config: Optional[dict] = None
) -> None:
    """
    OHLCV DataFrame을 PostgreSQL에 저장합니다.
    conflict_action: 중복키 충돌 시 수행할 액션 (기본 'DO NOTHING')
    db_config: DB 접속 정보를 담은 dict (없으면 DATABASE 사용)
    """
    if db_config is None:
        db_config = DATABASE

    conn = psycopg2.connect(
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port'],
        dbname=db_config['dbname']
    )
    cur = conn.cursor()

    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        timestamp TIMESTAMP NOT NULL,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume DOUBLE PRECISION,
        PRIMARY KEY (timestamp)
    );
    """
    cur.execute(create_table_query)
    conn.commit()

    for index, row in df.iterrows():
        insert_query = f"""
        INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp) 
        {conflict_action};
        """
        cur.execute(insert_query, (
            index.to_pydatetime(),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume'])
        ))
    conn.commit()

    cur.close()
    conn.close()

def fetch_ohlcv_records(
    table_name: str = 'ohlcv_data',
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_config: Optional[dict] = None
) -> pd.DataFrame:
    """
    PostgreSQL에서 OHLCV 데이터를 읽어옵니다.
    """
    if db_config is None:
        db_config = DATABASE

    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )

    query = f"SELECT * FROM {table_name}"
    where_clauses = []
    if start_date:
        where_clauses.append(f"timestamp >= '{start_date}'")
    if end_date:
        where_clauses.append(f"timestamp <= '{end_date}'")
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY timestamp"
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, engine, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

def delete_ohlcv_tables_by_symbol(
    symbol: str,
    schema: str = 'public',
    db_config: Optional[dict] = None
) -> None:
    """
    지정 심볼과 매칭되는 모든 테이블을 삭제합니다.
    schema: 사용할 스키마 (기본 'public')
    """
    if db_config is None:
        db_config = DATABASE

    symbol_for_table = symbol.replace("/", "").lower()

    conn = psycopg2.connect(
        user=db_config['user'],
        password=db_config['password'],
        host=db_config['host'],
        port=db_config['port'],
        dbname=db_config['dbname']
    )
    cur = conn.cursor()

    find_tables_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{schema}'
          AND table_name LIKE 'ohlcv_{symbol_for_table}_%';
    """
    cur.execute(find_tables_query)
    tables = cur.fetchall()

    for (table_name,) in tables:
        drop_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
        print(f"[*] Dropping table: {table_name}")
        cur.execute(drop_query)

    conn.commit()
    cur.close()
    conn.close()
