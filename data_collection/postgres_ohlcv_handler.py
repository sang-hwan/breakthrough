# data_collection/postgres_ohlcv_handler.py
# 이 파일은 PostgreSQL과의 입출력을 담당합니다.
# SQLAlchemy와 psycopg2를 사용합니다.

from sqlalchemy import create_engine
import psycopg2
import pandas as pd
from typing import Optional
from config.db_config import DATABASE  # DB 접속 정보를 담고 있는 설정


def save_ohlcv_to_postgres(df: pd.DataFrame, table_name: str = 'ohlcv_data') -> None:
    """
    OHLCV 형태의 DataFrame을 PostgreSQL 테이블에 저장합니다.
    - table_name이 없다면 'ohlcv_data'라는 이름을 기본 사용
    - timestamp 컬럼을 PK로 설정해 중복 데이터를 막습니다.
    """

    # (1) psycopg2로 DB 연결
    conn = psycopg2.connect(
        user=DATABASE['user'],
        password=DATABASE['password'],
        host=DATABASE['host'],
        port=DATABASE['port'],
        dbname=DATABASE['dbname']
    )
    cur = conn.cursor()

    # (2) 테이블이 없으면 생성
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

    # (3) DataFrame 행을 돌면서 INSERT (ON CONFLICT DO NOTHING은 중복 timestamp면 삽입 무시)
    for index, row in df.iterrows():
        insert_query = f"""
        INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp) 
        DO NOTHING;
        """
        cur.execute(insert_query, (
            index.to_pydatetime(),  # 인덱스가 날짜이므로 datetime 객체로 변환
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume'])
        ))
    conn.commit()

    cur.close()
    conn.close()


def load_ohlcv_from_postgres(
    table_name: str = 'ohlcv_data',
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    PostgreSQL 테이블에서 OHLCV 데이터를 읽어옵니다.
    - 기간(start_date, end_date)나 limit를 걸 수도 있습니다.
    - 결과를 DataFrame으로 반환.
    """

    # SQLAlchemy의 create_engine()으로 연결 문자열 구성
    engine = create_engine(
        f"postgresql://{DATABASE['user']}:{DATABASE['password']}@"
        f"{DATABASE['host']}:{DATABASE['port']}/{DATABASE['dbname']}"
    )

    # 기본 SELECT 쿼리
    query = f"SELECT * FROM {table_name}"

    # WHERE 절을 위해 조건들을 모을 리스트
    where_clauses = []
    if start_date:
        where_clauses.append(f"timestamp >= '{start_date}'")
    if end_date:
        where_clauses.append(f"timestamp <= '{end_date}'")

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    # 시간 순서대로 정렬
    query += " ORDER BY timestamp"

    # limit(최대 행 수) 지정이 있다면
    if limit:
        query += f" LIMIT {limit}"

    # 쿼리 실행 후 DataFrame으로 변환, timestamp를 datetime 형식으로 처리
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    return df


def delete_ohlcv_tables_by_symbol(symbol: str) -> None:
    """
    BTC/USDT => ohlcv_btcusdt_... 처럼 테이블 이름이 matching되는 걸 전부 삭제하는 함수.
    예) "ohlcv_btcusdt_4h", "ohlcv_btcusdt_1d" 등등
    """

    symbol_for_table = symbol.replace("/", "").lower()

    conn = psycopg2.connect(
        user=DATABASE['user'],
        password=DATABASE['password'],
        host=DATABASE['host'],
        port=DATABASE['port'],
        dbname=DATABASE['dbname']
    )
    cur = conn.cursor()

    # 해당 심볼 패턴의 테이블 목록을 찾아옴
    find_tables_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name LIKE 'ohlcv_{symbol_for_table}_%';
    """
    cur.execute(find_tables_query)
    tables = cur.fetchall()

    # 조회된 테이블을 순회하며 삭제
    for (table_name,) in tables:
        drop_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
        print(f"[*] Dropping table: {table_name}")
        cur.execute(drop_query)

    conn.commit()
    cur.close()
    conn.close()
