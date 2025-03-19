# data/fetch_data.py
"""
이 모듈은 데이터베이스에 저장된 OHLCV 데이터를 조회하는 기능을 제공합니다.
"""

import pandas as pd
from sqlalchemy import text
from typing import Dict, Any, Tuple, List
from data.db_config import DATABASE
from data.data_utils import create_db_engine
from logs.log_config import setup_logger

logger = setup_logger(__name__)

def fetch_ohlcv_records(
    table_name: str = 'ohlcv_data',
    start_date: str = None,
    end_date: str = None,
    db_config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    지정된 테이블에서 OHLCV 데이터를 조회하여 DataFrame으로 반환합니다.

    Parameters:
        table_name (str): 조회할 테이블 이름.
        start_date (str): 조회 시작 날짜 ("YYYY-MM-DD" 혹은 "YYYY-MM-DD HH:MM:SS").
        end_date (str): 조회 종료 날짜 ("YYYY-MM-DD" 혹은 "YYYY-MM-DD HH:MM:SS").
        db_config (dict): 데이터베이스 접속 정보.

    Returns:
        pd.DataFrame: 조회된 데이터. 오류 발생 시 빈 DataFrame 반환.
    """
    if db_config is None:
        db_config = DATABASE

    try:
        engine = create_db_engine(db_config)
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


def get_unique_symbol_list(db_config: Dict[str, Any] = None) -> List[str]:
    """
    public 스키마 내 테이블 이름을 통해 고유한 OHLCV 심볼 목록을 반환합니다.

    Returns:
        List[str]: 예) ["BTC/USDT", "ETH/USDT", ...]
    """
    if db_config is None:
        db_config = DATABASE
    try:
        engine = create_db_engine(db_config)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            tables = [row[0] for row in result]
    except Exception as e:
        logger.error(f"테이블 이름 조회 에러: {e}", exc_info=True)
        return []

    symbol_set = set()
    for table in tables:
        if table.startswith("ohlcv_"):
            parts = table.split("_")
            if len(parts) >= 3:
                symbol_set.add(parts[1])
    symbols = []
    for s in symbol_set:
        if s.endswith("usdt"):
            base = s[:-4].upper()
            symbols.append(f"{base}/USDT")
        else:
            symbols.append(s.upper())
    return sorted(list(symbols))

def get_date_range(table_name: str, db_config: Dict[str, Any] = None) -> Tuple[str, str]:
    """
    지정된 테이블에서 최소 및 최대 timestamp 값을 조회합니다.

    Returns:
        Tuple[str, str]: (start_date, end_date) 형식의 문자열.
                         조회 실패 시 (None, None)을 반환.
    """
    if db_config is None:
        db_config = DATABASE
    try:
        engine = create_db_engine(db_config)
    except Exception as e:
        logger.error(f"DB 엔진 생성 에러: {e}", exc_info=True)
        return None, None

    query = f"SELECT MIN(timestamp) AS start_date, MAX(timestamp) AS end_date FROM {table_name}"
    query = text(query)
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if row and row[0] and row[1]:
                start_date = row[0].strftime("%Y-%m-%d %H:%M:%S")
                end_date = row[1].strftime("%Y-%m-%d %H:%M:%S")
                return start_date, end_date
            else:
                logger.error(f"테이블 {table_name}에서 날짜 범위를 찾을 수 없습니다.")
                return None, None
    except Exception as e:
        logger.error(f"날짜 범위 조회 에러 ({table_name}): {e}", exc_info=True)
        return None, None
