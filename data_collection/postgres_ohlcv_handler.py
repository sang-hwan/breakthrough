# data_collection/postgres_ohlcv_handler.py

from sqlalchemy import create_engine  # 데이터베이스와 연결을 쉽게 도와주는 도구
import psycopg2  # PostgreSQL 데이터베이스와 직접 소통하기 위한 라이브러리
import pandas as pd  # 데이터를 분석하고 조작하는 데 필요한 강력한 도구
from typing import Optional  # 매개변수의 타입을 선택적으로 정의하기 위해 사용
from config.db_config import DATABASE  # 데이터베이스 연결 설정 정보를 가져오기 위한 설정 파일

# OHLCV 데이터를 PostgreSQL 데이터베이스에 저장하는 함수
def save_ohlcv_to_postgres(df: pd.DataFrame, table_name: str = 'ohlcv_data') -> None:
    """
    OHLCV 데이터를 PostgreSQL 데이터베이스에 저장합니다.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 데이터를 담은 pandas DataFrame 객체.
        필요한 컬럼: timestamp(인덱스), open, high, low, close, volume.
    table_name : str
        데이터를 저장할 테이블 이름. 기본값은 'ohlcv_data'.
    """
    
    # 데이터베이스에 연결합니다.
    conn = psycopg2.connect(
        user=DATABASE['user'],          # 데이터베이스 사용자 이름
        password=DATABASE['password'],  # 데이터베이스 비밀번호
        host=DATABASE['host'],          # 데이터베이스 호스트 (IP 주소 또는 도메인)
        port=DATABASE['port'],          # 데이터베이스 포트 번호
        dbname=DATABASE['dbname']       # 데이터베이스 이름
    )
    cur = conn.cursor()  # 데이터베이스 작업을 수행하기 위한 커서를 만듭니다.

    # 데이터를 저장할 테이블을 생성합니다. 이미 존재하면 넘어갑니다.
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        timestamp TIMESTAMP NOT NULL,       -- 데이터가 수집된 시점
        open DOUBLE PRECISION,             -- 시가 (주식이 시작된 가격)
        high DOUBLE PRECISION,             -- 고가 (가장 높은 가격)
        low DOUBLE PRECISION,              -- 저가 (가장 낮은 가격)
        close DOUBLE PRECISION,            -- 종가 (마감 가격)
        volume DOUBLE PRECISION,           -- 거래량 (거래된 총 수량)
        PRIMARY KEY (timestamp)            -- 기본 키는 timestamp (중복 방지)
    );
    """
    cur.execute(create_table_query)  # 테이블 생성 SQL 실행
    conn.commit()  # 데이터베이스에 변경 사항 적용

    # DataFrame 데이터를 하나씩 읽어서 테이블에 삽입합니다.
    for index, row in df.iterrows():  # DataFrame의 각 행을 반복
        insert_query = f"""
        INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp) 
        DO NOTHING;  -- timestamp가 중복되면 데이터를 삽입하지 않음
        """
        cur.execute(insert_query, (
            index.to_pydatetime(),  # timestamp를 datetime 형식으로 변환
            float(row['open']),    # 시가 값
            float(row['high']),    # 고가 값
            float(row['low']),     # 저가 값
            float(row['close']),   # 종가 값
            float(row['volume'])   # 거래량 값
        ))
    conn.commit()  # 삽입 작업 후 데이터베이스 적용

    # 데이터베이스 연결을 종료합니다.
    cur.close()  # 커서를 닫음
    conn.close()  # 연결 종료

# PostgreSQL 데이터베이스에서 OHLCV 데이터를 읽어오는 함수
def load_ohlcv_from_postgres(
    table_name: str = 'ohlcv_data',
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    PostgreSQL 데이터베이스에서 OHLCV 데이터를 가져옵니다.

    Parameters
    ----------
    table_name : str
        데이터를 가져올 테이블 이름. 기본값은 'ohlcv_data'.
    limit : Optional[int]
        가져올 데이터의 최대 행 수. 기본값은 제한 없음.
    start_date : Optional[str]
        데이터를 가져올 시작 날짜. 예: '2021-01-01 00:00:00'.
    end_date : Optional[str]
        데이터를 가져올 종료 날짜. 예: '2021-12-31 23:59:59'.

    Returns
    -------
    pd.DataFrame
        가져온 데이터를 담은 pandas DataFrame 객체.
    """
    
    # SQLAlchemy를 사용해 데이터베이스와 연결합니다.
    engine = create_engine(f"postgresql://{DATABASE['user']}:{DATABASE['password']}@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['dbname']}")

    # 기본 SQL 쿼리 작성
    query = f"SELECT * FROM {table_name}"

    # 시작일과 종료일에 따른 조건을 추가합니다.
    where_clauses = []
    if start_date:
        where_clauses.append(f"timestamp >= '{start_date}'")  # 시작 날짜 조건
    if end_date:
        where_clauses.append(f"timestamp <= '{end_date}'")  # 종료 날짜 조건

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)  # 조건 추가

    query += " ORDER BY timestamp"  # 데이터를 시간 순으로 정렬

    if limit:
        query += f" LIMIT {limit}"  # 행 수 제한

    # 쿼리를 실행하고 결과를 DataFrame으로 변환
    df = pd.read_sql(query, engine, parse_dates=['timestamp'])  # timestamp를 datetime 형식으로 변환
    df.set_index('timestamp', inplace=True)  # timestamp를 인덱스로 설정
    return df

# 특정 자산(symbol)과 관련된 테이블 삭제
def delete_ohlcv_tables_by_symbol(symbol: str) -> None:
    """
    특정 자산(symbol)과 관련된 모든 데이터를 삭제합니다.
    예: symbol="BTC/USDT" -> "ohlcv_btcusdt_..." 패턴의 테이블 모두 삭제.
    """
    # 심볼을 데이터베이스 테이블 이름에 맞게 가공
    symbol_for_table = symbol.replace("/", "").lower()

    conn = psycopg2.connect(
        user=DATABASE['user'],
        password=DATABASE['password'],
        host=DATABASE['host'],
        port=DATABASE['port'],
        dbname=DATABASE['dbname']
    )
    cur = conn.cursor()

    # 삭제 대상 테이블 목록 조회
    find_tables_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name LIKE 'ohlcv_{symbol_for_table}_%';
    """
    cur.execute(find_tables_query)
    tables = cur.fetchall()  # 테이블 이름 목록 가져오기

    # 조회된 테이블들을 순차적으로 삭제
    for (table_name,) in tables:
        drop_query = f"DROP TABLE IF EXISTS {table_name} CASCADE;"  # CASCADE: 연관된 데이터도 삭제
        print(f"[*] Dropping table: {table_name}")  # 삭제되는 테이블을 출력
        cur.execute(drop_query)

    conn.commit()  # 삭제 작업 저장
    cur.close()  # 커서 닫기
    conn.close()  # 연결 종료
