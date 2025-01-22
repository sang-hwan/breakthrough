# data_collection/postgres_ohlcv_handler.py

from sqlalchemy import create_engine # SQL 작업을 쉽고 직관적으로 처리하는 라이브러리
import psycopg2 # PostgreSQL 데이터베이스와 직접 통신할 때 사용.
import pandas as pd # 데이터 분석 및 조작을 위한 라이브러리.
from typing import Optional # 함수 매개변수의 타입을 명시하기 위한 라이브러리.
from config.db_config import DATABASE # 데이터베이스 연결 정보를 포함한 설정 파일에서 DATABASE 정보를 가져옵니다.

# OHLCV 데이터를 PostgreSQL에 저장하는 함수입니다.
def save_ohlcv_to_postgres(df: pd.DataFrame, table_name: str = 'ohlcv_data') -> None:
    """
    OHLCV 데이터를 PostgreSQL 데이터베이스에 저장합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터를 담고 있는 pandas DataFrame입니다.
        필요한 컬럼: timestamp(인덱스), open, high, low, close, volume.
    table_name : str
        데이터를 저장할 데이터베이스 테이블 이름. 기본값은 'ohlcv_data'입니다.
    """
    
    # 1) 데이터베이스 연결 생성
    # 데이터베이스 접속 정보를 사용하여 PostgreSQL에 연결합니다.
    conn = psycopg2.connect(
        user=DATABASE['user'],          # 사용자 이름
        password=DATABASE['password'],  # 비밀번호
        host=DATABASE['host'],          # 데이터베이스 호스트 주소
        port=DATABASE['port'],          # 데이터베이스 포트 번호
        dbname=DATABASE['dbname']       # 데이터베이스 이름
    )
    cur = conn.cursor()  # SQL 작업을 수행할 커서 생성

    # 2) 테이블 생성
    # 테이블이 존재하지 않으면 새로 생성합니다.
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        timestamp TIMESTAMP NOT NULL,       -- 데이터의 시간 정보
        open DOUBLE PRECISION,             -- 시가
        high DOUBLE PRECISION,             -- 고가
        low DOUBLE PRECISION,              -- 저가
        close DOUBLE PRECISION,            -- 종가
        volume DOUBLE PRECISION,           -- 거래량
        PRIMARY KEY (timestamp)            -- 기본 키로 timestamp를 사용
    );
    """
    cur.execute(create_table_query)  # SQL 쿼리 실행
    conn.commit()  # 변경사항 저장

    # 3) DataFrame 데이터를 테이블에 삽입
    # DataFrame의 각 행(row)을 데이터베이스에 추가합니다.
    for index, row in df.iterrows():
        insert_query = f"""
        INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp)
        DO NOTHING;  -- 중복된 timestamp 데이터는 무시
        """
        cur.execute(insert_query, (
            index.to_pydatetime(),  # 인덱스를 datetime 형식으로 변환
            float(row['open']),    # 시가
            float(row['high']),    # 고가
            float(row['low']),     # 저가
            float(row['close']),   # 종가
            float(row['volume'])   # 거래량
        ))
    conn.commit()  # 변경사항 저장

    # 4) 데이터베이스 연결 종료
    cur.close()  # 커서 닫기
    conn.close()  # 연결 종료

# PostgreSQL에서 OHLCV 데이터를 읽어오는 함수입니다.
def load_ohlcv_from_postgres(table_name: str = 'ohlcv_data', limit: Optional[int] = None) -> pd.DataFrame:
    """
    PostgreSQL 데이터베이스에서 OHLCV 데이터를 가져옵니다.
    
    Parameters
    ----------
    table_name : str
        데이터를 가져올 테이블 이름. 기본값은 'ohlcv_data'입니다.
    limit : Optional[int]
        가져올 데이터의 최대 개수. 기본값은 제한 없음(None)입니다.
        
    Returns
    -------
    pd.DataFrame
        OHLCV 데이터를 담고 있는 pandas DataFrame을 반환합니다.
    """
    
    # 1) SQLAlchemy 엔진 생성
    # 데이터베이스 연결 정보를 이용하여 엔진을 생성합니다.
    engine = create_engine(f"postgresql://{DATABASE['user']}:{DATABASE['password']}@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['dbname']}")

    # 2) SQL 쿼리 생성
    query = f"SELECT * FROM {table_name} ORDER BY timestamp"  # 데이터를 timestamp 기준으로 정렬
    if limit:  # limit 값이 있으면 최대 limit 개수만 가져옴
        query += f" LIMIT {limit}"

    # 3) SQL 결과를 DataFrame으로 변환
    # pandas의 read_sql()을 사용하여 SQL 결과를 읽어옵니다.
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])  # timestamp를 datetime 형식으로 변환

    # 4) DataFrame 인덱스 설정
    df.set_index('timestamp', inplace=True)  # timestamp를 인덱스로 설정

    # 5) DataFrame 반환
    return df
