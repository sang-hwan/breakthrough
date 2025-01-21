# data_collection/save_to_postgres.py

from sqlalchemy import create_engine
import psycopg2        # PostgreSQL에 연결하기 위한 파이썬 라이브러리
import pandas as pd
from typing import Optional
from config.db_config import DATABASE  # 데이터베이스 접속 정보를 담고 있는 설정 파일


def save_ohlcv_to_postgres(df: pd.DataFrame, table_name: str = 'ohlcv_data') -> None:
    """
    OHLCV DataFrame을 PostgreSQL에 저장하는 예시 함수입니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        timestamp(인덱스), open, high, low, close, volume 컬럼을 갖는 DataFrame.
    table_name : str
        데이터를 저장할 테이블 이름 (기본 'ohlcv_data').
    """
    
    # 1) DB 연결
    #    - 'psycopg2.connect'에 접속 정보를 넣어 PostgreSQL에 연결.
    conn = psycopg2.connect(
        user=DATABASE['user'],
        password=DATABASE['password'],
        host=DATABASE['host'],
        port=DATABASE['port'],
        dbname=DATABASE['dbname']
    )
    #    - 연결된 상태에서 SQL 쿼리를 실행하기 위해 cursor(커서)를 얻어옴
    cur = conn.cursor()

    # 2) 테이블 생성(없으면 새로 만들기)
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
    #    - 테이블을 만드는 SQL 쿼리 실행
    cur.execute(create_table_query)
    #    - 쿼리 적용(커밋)
    conn.commit()

    # 3) DataFrame 레코드를 한 줄씩 INSERT
    #    - 대량 삽입 시 COPY 문법 등 더 빠른 방법도 사용 가능.
    for index, row in df.iterrows():
        #   - timestamp, open, high, low, close, volume 순으로 값 매핑
        insert_query = f"""
        INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp)
        DO NOTHING;
        """
        #   - index는 DataFrame의 인덱스(즉, timestamp)
        #   - to_pydatetime()로 Python의 datetime 객체로 변환
        #   - float(row['open']) 등으로 실수 형태 변환
        cur.execute(insert_query, (
            index.to_pydatetime(),
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['volume'])
        ))
    #    - INSERT 이후 최종 커밋
    conn.commit()
    
    #    - 커서와 연결 종료
    cur.close()
    conn.close()


def load_ohlcv_from_postgres(table_name: str = 'ohlcv_data', limit: Optional[int] = None) -> pd.DataFrame:
    """
    Postgres에서 OHLCV 데이터를 읽어오는 함수입니다.
    
    Parameters
    ----------
    table_name : str
        데이터를 가져올 테이블 이름 (기본 'ohlcv_data').
    limit : Optional[int]
        가져올 레코드 개수를 제한하고 싶을 때 사용 (기본 None: 제한 없음).
        
    Returns
    -------
    pd.DataFrame
        timestamp(인덱스), open, high, low, close, volume 컬럼을 갖는 DataFrame.
    """
    
    # 1) SQLAlchemy 엔진 생성
    user     = DATABASE['user']
    password = DATABASE['password']
    host     = DATABASE['host']
    port     = DATABASE['port']
    dbname   = DATABASE['dbname']
    
    # postgresql://user:password@host:port/dbname
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")
    
    # 2) 필요한 SQL 쿼리 생성
    #    - timestamp 기준으로 정렬하여 전부 가져옴
    query = f"SELECT * FROM {table_name} ORDER BY timestamp"
    #    - limit 파라미터가 있으면, 해당 개수만큼만 가져옴
    if limit:
        query += f" LIMIT {limit}"

    # 3) pandas의 read_sql을 통해 SQL 결과를 바로 DataFrame으로 변환
    #    - parse_dates 옵션으로 timestamp 컬럼을 datetime으로 파싱
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])
    
    # 4) timestamp 컬럼을 인덱스로 설정
    df.set_index('timestamp', inplace=True)

    # 가공한 DataFrame 반환
    return df
