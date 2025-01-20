# db_manager.py

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd


# ------------------------------------------------------------------------------
# 1) DB 연결정보 설정
#    (실제 운영 시에는 .env 파일, 환경 변수, 별도 설정파일 등을 활용하세요)
# ------------------------------------------------------------------------------
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'my_database',
    'user': 'my_user',
    'password': 'my_password'
}


# ------------------------------------------------------------------------------
# 2) OHLCV 데이터를 저장할 테이블 생성 함수
# ------------------------------------------------------------------------------
def create_table():
    """
    OHLCV 데이터 저장용 테이블을 생성합니다.
    symbol, timeframe, timestamp 에 대해 UNIQUE 제약조건을 두어
    동일 시간대 중복 데이터가 들어오지 않도록 처리합니다.
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS ohlcv_data (
        id        SERIAL PRIMARY KEY,
        symbol    VARCHAR(50) NOT NULL,
        timeframe VARCHAR(10) NOT NULL,
        ts        TIMESTAMP NOT NULL,
        open      DOUBLE PRECISION,
        high      DOUBLE PRECISION,
        low       DOUBLE PRECISION,
        close     DOUBLE PRECISION,
        volume    DOUBLE PRECISION,
        UNIQUE (symbol, timeframe, ts)
    );
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(create_table_sql)
        conn.commit()
        cur.close()
        print("Table 'ohlcv_data' created/verified successfully.")
    except Exception as e:
        print("Error creating table:", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


# ------------------------------------------------------------------------------
# 3) OHLCV 데이터 저장 함수
# ------------------------------------------------------------------------------
def store_ohlcv_to_db(df: pd.DataFrame, symbol: str, timeframe: str):
    """
    OHLCV DataFrame을 받아서 PostgreSQL 테이블에 저장합니다.
    (기존 데이터와 중복 시 'ON CONFLICT DO NOTHING' 처리)
    
    Parameters
    ----------
    df : pd.DataFrame
        반드시 index가 timestamp(또는 datetime)이어야 하며,
        open, high, low, close, volume 컬럼이 존재해야 함.
    symbol : str
        예: 'BTC/USDT'
    timeframe : str
        예: '4h', '1d' 등
    """
    # DataFrame에서 DB에 넣을 형태의 list of tuple로 변환
    # (symbol, timeframe, ts, open, high, low, close, volume)
    records = []
    for ts, row in df.iterrows():
        records.append((
            symbol,
            timeframe,
            ts,                  # 인덱스가 datetime 형식이라고 가정
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume']
        ))
    
    insert_sql = """
    INSERT INTO ohlcv_data 
        (symbol, timeframe, ts, open, high, low, close, volume)
    VALUES %s
    ON CONFLICT (symbol, timeframe, ts) 
    DO NOTHING;
    """
    
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # psycopg2의 execute_values를 활용하면 대량 insert를 빠르게 처리 가능
        execute_values(cur, insert_sql, records)
        
        conn.commit()
        cur.close()
        print(f"Inserted {len(records)} rows for {symbol} / {timeframe}.")
    except Exception as e:
        print("Error inserting data:", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


# ------------------------------------------------------------------------------
# 4) OHLCV 데이터 조회 함수
# ------------------------------------------------------------------------------
def load_ohlcv_from_db(symbol: str, timeframe: str,
                       start_time=None, end_time=None) -> pd.DataFrame:
    """
    PostgreSQL에서 지정된 심볼+타임프레임의 OHLCV 데이터를 조회하여
    pandas DataFrame으로 반환합니다.
    
    Parameters
    ----------
    symbol : str
        조회할 코인 심볼 (예: 'BTC/USDT')
    timeframe : str
        조회할 타임프레임 (예: '4h', '1d' 등)
    start_time : datetime or str
        조회 시작 시각 (없으면 전체)
    end_time : datetime or str
        조회 종료 시각 (없으면 전체)
    
    Returns
    -------
    pd.DataFrame
        index가 timestamp이고, open/high/low/close/volume 컬럼을 가지는 DF
    """
    where_clauses = ["symbol = %s", "timeframe = %s"]
    params = [symbol, timeframe]
    
    if start_time:
        where_clauses.append("ts >= %s")
        params.append(start_time)
    if end_time:
        where_clauses.append("ts <= %s")
        params.append(end_time)
    
    where_sql = " AND ".join(where_clauses)
    
    query_sql = f"""
    SELECT ts, open, high, low, close, volume
      FROM ohlcv_data
     WHERE {where_sql}
     ORDER BY ts ASC;
    """
    
    conn = None
    df = pd.DataFrame()
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(query_sql, params)
        rows = cur.fetchall()
        cur.close()
        
        if rows:
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.set_index('timestamp', inplace=True)
            df = df.astype(float, errors='ignore')
        else:
            print("No data found for the given query.")
    except Exception as e:
        print("Error loading data:", e)
    finally:
        if conn:
            conn.close()
    
    return df
