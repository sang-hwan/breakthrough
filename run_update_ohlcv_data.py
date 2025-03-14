# run_update_ohlcv_data.py
"""
이 스크립트는 거래 데이터(OHLCV: Open, High, Low, Close, Volume)를
최신 상태로 업데이트하기 위한 작업을 수행합니다.

주요 기능:
  1. .env에 정의된 데이터베이스가 없으면 생성
  2. 상위 거래량 심볼 및 해당 심볼의 최초 온보딩 날짜를 조회하여
     공통 시작 날짜를 결정
  3. 각 심볼과 각 시간 프레임에 대해 기존 데이터를 조회하고,
     새로운 데이터를 API 등으로 가져와서 데이터베이스에 삽입
"""

import sys  # 시스템 종료를 위해 사용
from datetime import datetime, timedelta, timezone  # 날짜 및 시간 처리를 위한 모듈
import pandas as pd  # 데이터프레임 처리를 위한 라이브러리
from dotenv import load_dotenv  # 환경변수 로드를 위한 라이브러리
import psycopg2  # PostgreSQL 데이터베이스 연결 라이브러리
from psycopg2 import sql  # SQL 쿼리 작성 시 안전하게 식별자 처리를 위한 모듈

from data.db.db_config import DATABASE  # 데이터베이스 접속 정보를 담은 설정 객체
from data.db.db_manager import fetch_ohlcv_records, insert_ohlcv_records  # 기존 데이터 조회 및 삽입 함수
from data.ohlcv.ohlcv_fetcher import fetch_historical_ohlcv_data, get_top_volume_symbols, get_latest_onboard_date  # OHLCV 데이터 관련 함수들
from logs.logger_config import initialize_root_logger, setup_logger  # 로깅 초기화 및 설정 함수

# 환경변수 로드 및 로깅 초기화
load_dotenv()
initialize_root_logger()
logger = setup_logger(__name__)

def create_database_if_not_exists(db_config):
    """
    .env에 명시된 데이터베이스가 존재하지 않을 경우,
    PostgreSQL의 기본 데이터베이스("postgres")에 접속하여 해당 데이터베이스를 생성합니다.
    
    Parameters:
        db_config (dict): 데이터베이스 접속 정보를 담은 딕셔너리 
                          (예: {'dbname': ..., 'user': ..., 'password': ..., 'host': ..., 'port': ...})
    
    Returns:
        None: 데이터베이스 생성 여부에 따라 로그를 기록하며, 에러 발생 시 프로그램을 종료합니다.
    """
    dbname = db_config.get('dbname')
    user = db_config.get('user')
    password = db_config.get('password')
    host = db_config.get('host')
    port = db_config.get('port')
    try:
        # PostgreSQL 기본 데이터베이스인 "postgres"에 접속합니다.
        conn = psycopg2.connect(dbname="postgres", user=user, password=password, host=host, port=port)
        conn.autocommit = True  # 자동 커밋 모드를 활성화합니다.
        cur = conn.cursor()
        # 타겟 데이터베이스의 존재 여부를 확인합니다.
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        exists = cur.fetchone()
        if not exists:
            logger.debug(f"Database '{dbname}' does not exist. Creating database.")
            # 안전한 SQL 식별자 처리를 위해 psycopg2.sql 모듈 사용
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
        else:
            logger.debug(f"Database '{dbname}' already exists.")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error checking/creating database: {e}", exc_info=True)
        sys.exit(1)

def run_update_ohlcv_data():
    """
    최신 OHLCV 데이터를 가져와서 데이터베이스에 업데이트하는 메인 함수입니다.
    
    주요 단계:
      1. 데이터베이스가 존재하지 않으면 생성
      2. 상위 거래량 심볼(예: Binance에서 USDT 마켓 상위 심볼)과 각 심볼의 첫 거래 날짜를 조회
      3. 공통 시작 날짜(모든 심볼에 적용)를 결정
      4. 각 심볼과 각 시간 프레임(1d, 4h, 1h, 15m)에 대해:
            - 기존 데이터를 조회하고,
            - 데이터가 있다면 마지막 타임스탬프 이후부터, 없으면 공통 시작 날짜부터
            - 새로운 OHLCV 데이터를 API로 조회 후,
            - 조회된 데이터가 있으면 데이터베이스에 삽입
      5. 각 단계에서 에러 발생 시 로깅 처리 후 다음 처리로 진행 또는 종료
      
    Parameters:
        없음
    
    Returns:
        None
    """
    # 데이터베이스가 존재하지 않으면 생성합니다.
    create_database_if_not_exists(DATABASE)
    
    # 상위 거래량 심볼 조회: (심볼, 첫 거래일) 튜플 리스트 반환
    symbols_with_onboard = get_top_volume_symbols(exchange_id='binance', quote_currency='USDT', count=3)
    if not symbols_with_onboard:
        logger.error("No valid symbols found from Binance.")
        sys.exit(1)
    logger.debug(f"Top symbols (with onboard date): {symbols_with_onboard}")
    
    # 모든 심볼에 대해 공통 시작 날짜를 결정 (가장 늦은 온보딩 날짜 사용)
    global_start_date = get_latest_onboard_date(symbols_with_onboard, exchange_id='binance')
    logger.debug(f"Unified start date for all symbols: {global_start_date}")
    
    # 심볼 이름만 추출하여 리스트 생성
    symbols = [item[0] for item in symbols_with_onboard]
    # 데이터 업데이트할 다양한 시간 프레임 설정
    timeframes = ["1d", "4h", "1h", "15m"]
    
    # 업데이트 종료 시점을 현재 UTC 시간으로 설정
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    # 각 심볼과 각 시간 프레임에 대해 데이터 업데이트를 수행합니다.
    for symbol in symbols:
        symbol_key = symbol.replace("/", "").lower()  # 테이블명 생성을 위한 심볼 변환
        logger.debug(f"Processing {symbol} (table prefix: ohlcv_{symbol_key}_)")
        for tf in timeframes:
            table_name = f"ohlcv_{symbol_key}_{tf}"
            logger.debug(f"Processing {symbol} - {tf} (table: {table_name})")
            
            try:
                # 데이터베이스에서 기존 OHLCV 데이터를 조회합니다.
                df_existing = fetch_ohlcv_records(table_name=table_name)
            except Exception as e:
                logger.error(f"Error fetching existing data for table {table_name}: {e}", exc_info=True)
                df_existing = pd.DataFrame()  # 에러 발생 시 빈 DataFrame 사용
            
            if not df_existing.empty:
                # 기존 데이터가 있다면 마지막 타임스탬프 이후부터 새로운 데이터 조회
                last_timestamp = df_existing.index.max()
                new_start_dt = last_timestamp + timedelta(seconds=1)
                new_start_date = new_start_dt.strftime("%Y-%m-%d %H:%M:%S")
                logger.debug(f"Existing data found in {table_name}. Fetching new data from {new_start_date} to {end_date}.")
            else:
                # 데이터가 없으면 공통 시작 날짜부터 조회
                new_start_date = global_start_date
                logger.debug(f"No existing data in {table_name}. Fetching data from {new_start_date} to {end_date}.")
            
            try:
                # 지정된 기간 동안 OHLCV 데이터를 조회합니다.
                df_new = fetch_historical_ohlcv_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=new_start_date,
                    exchange_id='binance'
                )
                if df_new.empty:
                    logger.debug(f"No new data fetched for {symbol} - {tf}.")
                    continue
                else:
                    logger.debug(f"Fetched {len(df_new)} new rows for {symbol} - {tf}.")
            except Exception as e:
                logger.error(f"Error fetching OHLCV data for {symbol} - {tf}: {e}", exc_info=True)
                continue
            
            try:
                # 조회된 새로운 데이터를 데이터베이스 테이블에 삽입합니다.
                insert_ohlcv_records(df_new, table_name=table_name)
                logger.debug(f"Inserted new data into table {table_name}.")
            except Exception as e:
                logger.error(f"Error inserting data into table {table_name}: {e}", exc_info=True)

if __name__ == "__main__":
    run_update_ohlcv_data()
