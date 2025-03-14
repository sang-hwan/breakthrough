# data/db/db_manager.py

# 필요한 라이브러리 및 모듈을 임포트합니다.
from sqlalchemy import create_engine, text   # SQLAlchemy: 데이터베이스 엔진 생성 및 SQL 실행
from psycopg2.extras import execute_values   # psycopg2 헬퍼 함수: 다중 레코드 삽입 최적화
import pandas as pd                            # 데이터프레임(DataFrame) 처리를 위한 pandas
from typing import Any, Iterable, Dict          # 타입 힌트를 제공하기 위한 모듈
from data.db.db_config import DATABASE         # 데이터베이스 접속 설정 정보를 불러옵니다.
from logs.logger_config import setup_logger     # 로깅 설정을 불러와 로그 기록에 사용합니다.

# 로깅 객체(logger)를 전역 변수로 생성합니다.
# 모듈명을 기준으로 로거를 설정하여 이후 에러 및 실행 정보를 기록할 때 사용됩니다.
logger = setup_logger(__name__)

def insert_on_conflict(table: Any, conn: Any, keys: list, data_iter: Iterable) -> None:
    """
    pandas의 to_sql 메서드와 함께 사용되는 커스텀 삽입 함수입니다.
    'timestamp' 컬럼에서 중복(Conflict)이 발생할 경우 아무 작업도 수행하지 않고 건너뜁니다.
    
    Parameters:
        table (Any): 데이터베이스 테이블 객체로, 삽입 대상 테이블을 의미합니다.
        conn (Any): SQLAlchemy 연결 객체로, 실제 데이터베이스 연결 정보를 포함합니다.
        keys (list): 삽입할 데이터의 컬럼 이름 리스트입니다.
        data_iter (Iterable): 삽입할 데이터가 담긴 반복 가능한 객체입니다.
    
    Returns:
        None: 함수는 데이터 삽입 작업만 수행하며 반환값은 없습니다.
    
    주요 로직:
        - SQLAlchemy 연결 객체에서 psycopg2의 원시 연결(raw connection)을 추출합니다.
        - 커서를 생성하고, 데이터를 리스트로 변환한 후 삽입할 컬럼 이름들을 문자열로 만듭니다.
        - INSERT SQL문에 'ON CONFLICT (timestamp) DO NOTHING' 구문을 추가하여 timestamp 중복 시 삽입하지 않습니다.
        - 다중 레코드를 한 번에 삽입한 후 커서를 닫습니다.
    """
    try:
        # SQLAlchemy 연결 객체에서 psycopg2의 원시 커넥션을 가져옵니다.
        raw_conn = conn.connection
        # SQL 명령을 실행할 커서를 생성합니다.
        cur = raw_conn.cursor()
        # 삽입할 데이터를 리스트 형태로 변환합니다.
        values = list(data_iter)
        # 삽입할 컬럼 이름들을 쉼표로 구분하여 하나의 문자열로 만듭니다.
        columns = ", ".join(keys)
        # SQL INSERT 문을 생성합니다.
        # 'ON CONFLICT (timestamp) DO NOTHING'은 timestamp 컬럼에서 충돌 시 아무 작업도 하지 않도록 합니다.
        sql_str = f"INSERT INTO {table.name} ({columns}) VALUES %s ON CONFLICT (timestamp) DO NOTHING"
        # psycopg2의 execute_values를 사용하여 다중 레코드를 효율적으로 삽입합니다.
        execute_values(cur, sql_str, values)
        # 사용이 끝난 커서를 닫아 리소스를 해제합니다.
        cur.close()
    except Exception as e:
        # 예외 발생 시 에러 메시지와 스택 트레이스를 로깅합니다.
        logger.error(f"insert_on_conflict 에러: {e}", exc_info=True)

def insert_ohlcv_records(df: pd.DataFrame, table_name: str = 'ohlcv_data', conflict_action: str = "DO NOTHING",
                         db_config: Dict[str, Any] = None, chunk_size: int = 10000) -> None:
    """
    OHLCV (시가, 고가, 저가, 종가, 거래량) 데이터를 지정된 테이블에 삽입합니다.
    만약 해당 테이블이 존재하지 않으면 새로 생성하며, 데이터 삽입 시 timestamp 충돌 발생 시 아무 작업도 하지 않습니다.
    
    Parameters:
        df (pd.DataFrame): 삽입할 OHLCV 데이터를 담은 pandas DataFrame.
        table_name (str): 데이터를 삽입할 테이블의 이름. 기본값은 'ohlcv_data'입니다.
        conflict_action (str): 데이터 삽입 시 충돌 발생 처리 방식. 현재 "DO NOTHING" 방식만 사용합니다.
        db_config (Dict[str, Any]): 데이터베이스 접속 설정 딕셔너리. 제공되지 않으면 기본 DATABASE를 사용합니다.
        chunk_size (int): 한 번에 삽입할 데이터 행(row)의 최대 개수. 기본값은 10000입니다.
    
    Returns:
        None: 데이터 삽입 작업을 수행하며, 별도의 반환값은 없습니다.
    
    주요 로직:
        - 데이터베이스 연결 엔진을 생성합니다.
        - 테이블이 존재하지 않을 경우, OHLCV 데이터를 위한 테이블을 생성하는 SQL문을 실행합니다.
        - DataFrame의 인덱스를 초기화하여 삽입에 적합한 형태로 변환한 후, to_sql 메서드로 데이터를 삽입합니다.
        - 삽입 과정에서는 커스텀 함수(insert_on_conflict)를 이용해 timestamp 충돌을 처리합니다.
    """
    # db_config가 제공되지 않으면 기본 DATABASE 설정을 사용합니다.
    if db_config is None:
        db_config = DATABASE

    # SQLAlchemy 엔진 생성: 데이터베이스와의 연결을 담당합니다.
    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
        pool_pre_ping=True  # 연결 유효성 검사를 활성화하여 끊긴 연결을 감지합니다.
    )

    # 테이블이 존재하지 않을 경우 생성할 SQL 문을 작성합니다.
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
        # 트랜잭션 블록 내에서 테이블 생성 SQL을 실행합니다.
        with engine.begin() as conn:
            conn.execute(create_table_sql)
    except Exception as e:
        # 테이블 생성 중 오류가 발생하면 에러 메시지를 로깅하고 함수를 종료합니다.
        logger.error(f"테이블 생성 에러 ({table_name}): {e}", exc_info=True)
        return

    try:
        # 원본 DataFrame을 복사하여 작업 시 데이터 무결성을 유지합니다.
        df = df.copy()
        # DataFrame의 인덱스를 컬럼으로 변환합니다.
        df.reset_index(inplace=True)
        # pandas의 to_sql 메서드를 사용하여 데이터를 데이터베이스에 삽입합니다.
        # if_exists='append' 옵션을 사용해 기존 데이터에 추가하며,
        # method 인자로 커스텀 충돌 처리 함수(insert_on_conflict)를 지정합니다.
        df.to_sql(
            table_name,
            engine,
            if_exists='append',
            index=False,
            method=insert_on_conflict,
            chunksize=chunk_size
        )
    except Exception as e:
        # 데이터 삽입 중 오류가 발생하면 에러 메시지를 로깅합니다.
        logger.error(f"데이터 저장 에러 ({table_name}): {e}", exc_info=True)

def fetch_ohlcv_records(table_name: str = 'ohlcv_data', start_date: str = None, end_date: str = None,
                        db_config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    지정된 테이블에서 OHLCV 데이터를 조회하여, 선택한 날짜 범위 내의 데이터를 pandas DataFrame으로 반환합니다.
    
    Parameters:
        table_name (str): 조회할 테이블의 이름. 기본값은 'ohlcv_data'입니다.
        start_date (str): 조회 시작 날짜 (포함). "YYYY-MM-DD" 또는 "YYYY-MM-DD HH:MM:SS" 형식.
        end_date (str): 조회 종료 날짜 (포함). "YYYY-MM-DD" 또는 "YYYY-MM-DD HH:MM:SS" 형식.
        db_config (Dict[str, Any]): 데이터베이스 접속 설정 딕셔너리. 제공되지 않으면 기본 DATABASE 사용.
    
    Returns:
        pd.DataFrame: 조회된 OHLCV 데이터를 담은 DataFrame.
                      오류 발생 시 빈 DataFrame을 반환합니다.
    
    주요 로직:
        - 데이터베이스 연결 엔진을 생성합니다.
        - 날짜 조건(start_date, end_date)이 있을 경우 SQL WHERE 절에 조건을 추가합니다.
        - SQLAlchemy의 text() 함수를 사용해 쿼리를 구성하고, pandas의 read_sql로 실행 결과를 DataFrame으로 변환합니다.
        - 'timestamp' 컬럼을 인덱스로 설정하여 반환합니다.
    """
    # db_config가 제공되지 않으면 기본 DATABASE 설정을 사용합니다.
    if db_config is None:
        db_config = DATABASE

    try:
        # 데이터베이스 연결 엔진을 생성합니다.
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            pool_pre_ping=True
        )
    except Exception as e:
        # 엔진 생성 실패 시 에러를 로깅하고 빈 DataFrame 반환.
        logger.error(f"DB 엔진 생성 에러: {e}", exc_info=True)
        return pd.DataFrame()

    # 기본 SQL 쿼리문: WHERE 1=1은 추가 조건을 쉽게 붙이기 위한 트릭입니다.
    query = f"SELECT * FROM {table_name} WHERE 1=1"
    params = {}
    # 시작 날짜 조건이 제공되면 쿼리에 추가합니다.
    if start_date:
        query += " AND timestamp >= :start_date"
        params['start_date'] = start_date
    # 종료 날짜 조건이 제공되면 쿼리에 추가합니다.
    if end_date:
        query += " AND timestamp <= :end_date"
        params['end_date'] = end_date
    # 결과를 timestamp 기준으로 정렬합니다.
    query += " ORDER BY timestamp"
    # SQLAlchemy의 text() 함수를 사용하여 SQL 텍스트 객체로 변환합니다.
    query = text(query)
    try:
        # pandas의 read_sql 함수를 이용해 SQL 쿼리 결과를 DataFrame으로 읽어옵니다.
        # parse_dates를 통해 'timestamp' 컬럼을 날짜형으로 파싱합니다.
        df = pd.read_sql(query, engine, params=params, parse_dates=['timestamp'])
        # 'timestamp' 컬럼을 인덱스로 설정합니다.
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        # 데이터 조회 중 오류 발생 시 에러 메시지를 로깅하고 빈 DataFrame 반환.
        logger.error(f"데이터 로드 에러 ({table_name}): {e}", exc_info=True)
        return pd.DataFrame()

def get_unique_symbol_list(db_config: Dict[str, Any] = None) -> list:
    """
    데이터베이스의 public 스키마에서 테이블 이름을 조회하고,
    테이블 이름이 "ohlcv_{symbol}_{timeframe}" 형식을 따르는 경우 고유한 symbol 목록(예: BTC/USDT)을 반환합니다.
    
    Parameters:
        db_config (Dict[str, Any]): 데이터베이스 접속 설정 딕셔너리.
                                    제공되지 않으면 기본 DATABASE 사용.
    
    Returns:
        list: 표준 형식으로 변환된 고유한 symbol 목록을 문자열 리스트로 반환합니다.
              예: ["BTC/USDT", "ETH/USDT", ...]
    
    주요 로직:
        - 데이터베이스 연결을 통해 public 스키마의 모든 테이블 이름을 조회합니다.
        - 테이블 이름이 'ohlcv_'로 시작하는 경우, 언더스코어('_')를 기준으로 분할하여 symbol 정보를 추출합니다.
        - 추출한 symbol이 'usdt'로 끝나면, 이를 'BTC/USDT'와 같이 표준 형식으로 변환합니다.
        - 고유한 symbol을 정렬하여 반환합니다.
    """
    # db_config가 제공되지 않으면 기본 DATABASE 설정을 사용합니다.
    if db_config is None:
        db_config = DATABASE
    try:
        # 데이터베이스 연결 엔진을 생성합니다.
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            pool_pre_ping=True
        )
        # 엔진을 통해 데이터베이스에 접속하여 public 스키마의 테이블 이름을 조회합니다.
        with engine.connect() as conn:
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = [row[0] for row in result]
    except Exception as e:
        # 테이블 이름 조회 중 오류가 발생하면 에러 메시지를 로깅하고 빈 리스트 반환.
        logger.error(f"Error fetching table names: {e}", exc_info=True)
        return []

    # 중복 제거를 위한 집합을 생성합니다.
    symbol_set = set()
    for table in tables:
        # 테이블 이름이 'ohlcv_'로 시작하는 경우에만 처리합니다.
        if table.startswith("ohlcv_"):
            # '_'를 기준으로 분할하면 형식은: ohlcv_{symbol}_{timeframe}이 됩니다.
            parts = table.split("_")
            if len(parts) >= 3:
                # 두 번째 요소(parts[1])가 symbol 정보입니다. 예: 'btcusdt'
                symbol_key = parts[1]
                symbol_set.add(symbol_key)

    symbols = []
    # 추출한 symbol_key를 표준 형식으로 변환합니다.
    for s in symbol_set:
        # 만약 symbol_key가 'usdt'로 끝나면, 예: 'btcusdt' -> 'BTC/USDT'
        if s.endswith("usdt"):
            base = s[:-4].upper()  # 마지막 4글자('usdt') 제거 후 대문자로 변환
            symbol = f"{base}/USDT"
            symbols.append(symbol)
        else:
            # 그 외의 경우 단순히 대문자로 변환합니다.
            symbols.append(s.upper())
    # 정렬된 symbol 리스트를 반환합니다.
    return list(sorted(symbols))

def get_date_range(table_name: str, db_config: dict = None) -> tuple:
    """
    지정된 테이블에서 가장 오래된(timestamp 최소값) 날짜와 최신(timestamp 최대값) 날짜를 조회하여 반환합니다.
    
    Parameters:
        table_name (str): 날짜 범위를 조회할 테이블의 이름.
        db_config (dict): 데이터베이스 접속 설정 딕셔너리. 제공되지 않으면 기본 DATABASE 사용.
    
    Returns:
        tuple: (start_date, end_date) 형식의 튜플을 반환합니다.
               각 날짜는 "YYYY-MM-DD HH:MM:SS" 형식의 문자열입니다.
               조회에 실패하면 (None, None)을 반환합니다.
    
    주요 로직:
        - 데이터베이스 연결 엔진을 생성합니다.
        - SQL 쿼리를 통해 해당 테이블의 최소(timestamp) 및 최대(timestamp) 값을 조회합니다.
        - 조회된 날짜를 지정한 문자열 포맷으로 변환 후 반환합니다.
    """
    # db_config가 제공되지 않으면 기본 DATABASE 설정을 사용합니다.
    if db_config is None:
        db_config = DATABASE
    try:
        # 데이터베이스 연결 엔진 생성
        engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['dbname']}",
            pool_pre_ping=True
        )
    except Exception as e:
        # 엔진 생성 실패 시 에러 메시지 로깅 후 (None, None) 반환.
        logger.error(f"DB 엔진 생성 에러: {e}", exc_info=True)
        return None, None

    # 최소 및 최대 timestamp 값을 조회하는 SQL 쿼리문을 작성합니다.
    query = f"SELECT MIN(timestamp) AS start_date, MAX(timestamp) AS end_date FROM {table_name}"
    query = text(query)
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            # 유효한 결과가 있는 경우 날짜를 지정한 형식의 문자열로 변환합니다.
            if row and row[0] and row[1]:
                start_date = row[0].strftime("%Y-%m-%d %H:%M:%S")
                end_date = row[1].strftime("%Y-%m-%d %H:%M:%S")
                return start_date, end_date
            else:
                # 유효한 날짜 범위를 찾지 못하면 에러 메시지를 로깅합니다.
                logger.error(f"테이블 {table_name}에서 날짜 범위를 찾을 수 없습니다.")
                return None, None
    except Exception as e:
        # SQL 실행 중 오류 발생 시 에러 메시지 로깅 후 (None, None) 반환.
        logger.error(f"날짜 범위 조회 에러 ({table_name}): {e}", exc_info=True)
        return None, None
