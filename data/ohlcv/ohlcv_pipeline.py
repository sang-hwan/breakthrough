# data/ohlcv/ohlcv_pipeline.py

import time  # API 호출 간 지연 시간 적용을 위한 모듈
import threading  # 멀티스레딩 및 동기화를 위한 모듈
from typing import List, Optional  # 타입 힌트를 위한 모듈
import concurrent.futures  # 스레드 풀을 사용한 병렬 처리 모듈

from logs.logger_config import setup_logger  # 로깅 설정 모듈
from data.ohlcv.ohlcv_fetcher import (
    fetch_historical_ohlcv_data,
    fetch_latest_ohlcv_data
)
from data.db.db_manager import insert_ohlcv_records  # 수집된 데이터를 데이터베이스에 저장하기 위한 함수

# 전역 변수: 로깅 설정 객체
logger = setup_logger(__name__)

# 전역 변수: API 호출 결과 캐싱을 위한 in-memory 캐시와 해당 접근을 위한 락
_cache_lock = threading.Lock()  # 멀티스레딩 환경에서 캐시 동시 접근 제어용 락
_ohlcv_cache: dict = {}         # 이미 호출한 OHLCV 데이터를 저장하여 중복 API 호출 방지

def collect_and_store_ohlcv_data(
    symbols: List[str],
    timeframes: List[str],
    use_historical: bool = True,
    start_date: Optional[str] = '2018-01-01 00:00:00',
    limit_per_request: int = 1000,
    latest_limit: int = 500,
    pause_sec: float = 1.0,
    table_name_format: str = "ohlcv_{symbol}_{timeframe}",
    exchange_id: str = 'binance',
    time_offset_ms: int = 1
) -> None:
    """
    지정된 심볼과 시간 간격에 대해 OHLCV 데이터를 수집하고 데이터베이스에 저장합니다.
    - use_historical가 True이면 과거 데이터를, False이면 최신 데이터만 수집합니다.
    - in-memory 캐시를 사용하여 중복 API 호출을 피하고, 동시 실행을 위해 스레드 풀을 사용합니다.
    
    Parameters:
        symbols (List[str]): 거래 심볼 리스트 (예: ['BTC/USDT', 'ETH/USDT']).
        timeframes (List[str]): 시간 간격 리스트 (예: ['1d', '1h']).
        use_historical (bool): 과거 데이터 수집 여부.
        start_date (Optional[str]): 과거 데이터 수집 시작일 (use_historical True 시 필수).
        limit_per_request (int): 과거 데이터 수집 시 요청당 데이터 제한.
        latest_limit (int): 최신 데이터 수집 시 요청 제한 개수.
        pause_sec (float): API 호출 간 대기 시간.
        table_name_format (str): 데이터 저장 시 테이블 이름 형식. {symbol}과 {timeframe}이 포맷팅됨.
        exchange_id (str): 사용 거래소 ID (기본 'binance').
        time_offset_ms (int): 데이터 수집 시 타임스탬프 오프셋 (밀리초 단위).
    
    Returns:
        None
    """
    
    # 내부 함수: 각 심볼과 시간 간격(timeframe) 조합에 대해 데이터를 처리 및 저장합니다.
    def process_symbol_tf(symbol: str, tf: str) -> None:
        # 캐시 키 구성: 파라미터 조합을 튜플로 묶어 고유 식별자 역할 수행
        key = (symbol, tf, use_historical, start_date, limit_per_request, latest_limit, exchange_id, time_offset_ms)
        with _cache_lock:
            # 캐시에서 이미 데이터가 존재하는지 확인
            if key in _ohlcv_cache:
                df = _ohlcv_cache[key]
                logger.debug(f"Cache hit for {symbol} {tf}")
            else:
                logger.debug(f"Cache miss for {symbol} {tf}, fetching data")
                # 과거 데이터를 수집하는 경우 반드시 start_date가 필요함
                if use_historical:
                    if not start_date:
                        raise ValueError("과거 데이터 수집 시 start_date가 필요합니다.")
                    df = fetch_historical_ohlcv_data(
                        symbol=symbol,
                        timeframe=tf,
                        start_date=start_date,
                        limit_per_request=limit_per_request,
                        pause_sec=pause_sec,
                        exchange_id=exchange_id,
                        single_fetch=False,
                        time_offset_ms=time_offset_ms
                    )
                else:
                    # 최신 데이터 수집
                    df = fetch_latest_ohlcv_data(
                        symbol=symbol,
                        timeframe=tf,
                        limit=latest_limit,
                        exchange_id=exchange_id
                    )
                # 수집한 데이터를 캐시에 저장하여 후속 호출 시 재사용
                _ohlcv_cache[key] = df
        
        if df.empty:
            logger.warning(f"[OHLCV PIPELINE] {symbol} - {tf} 데이터가 없습니다. 저장 건너뜁니다.")
            return
        
        # 테이블 이름 구성: 심볼의 슬래시 제거 및 소문자 변환, 시간 간격 포함
        table_name = table_name_format.format(symbol=symbol.replace('/', '').lower(), timeframe=tf)
        try:
            # 수집한 OHLCV 데이터를 데이터베이스에 저장하는 함수 호출
            insert_ohlcv_records(df, table_name=table_name)
            logger.debug(f"Data inserted for {symbol} {tf} into table {table_name}")
        except Exception as e:
            logger.error(f"[OHLCV PIPELINE] 데이터 저장 에러 ({table_name}): {e}", exc_info=True)
        time.sleep(pause_sec)  # API rate limit 준수를 위해 짧은 대기 시간 적용

    tasks = []  # 스레드풀에 제출할 작업들을 저장하는 리스트
    # 최대 10개의 스레드로 동시 실행 (병렬 처리)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for symbol in symbols:
            for tf in timeframes:
                # 각 심볼-시간간격 조합에 대해 process_symbol_tf 함수를 스레드풀에 제출
                tasks.append(executor.submit(process_symbol_tf, symbol, tf))
        # 모든 제출된 작업이 완료될 때까지 대기
        concurrent.futures.wait(tasks)
