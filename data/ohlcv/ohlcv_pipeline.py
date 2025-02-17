# data/ohlcv/ohlcv_pipeline.py
import time
import threading
from typing import List, Optional
import concurrent.futures
from logs.logger_config import setup_logger
from data.ohlcv.ohlcv_fetcher import (
    fetch_historical_ohlcv_data,
    fetch_latest_ohlcv_data
)
from data.db.db_manager import insert_ohlcv_records

logger = setup_logger(__name__)

# In-memory cache for fetched OHLCV data (to avoid duplicate API calls)
_cache_lock = threading.Lock()
_ohlcv_cache: dict = {}

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
    Collect OHLCV data for given symbols and timeframes, and store them into the database.
    Uses threading to fetch data concurrently.
    """
    def process_symbol_tf(symbol: str, tf: str) -> None:
        # Create a cache key including all parameters that affect the API call
        key = (symbol, tf, use_historical, start_date, limit_per_request, latest_limit, exchange_id, time_offset_ms)
        with _cache_lock:
            if key in _ohlcv_cache:
                df = _ohlcv_cache[key]
                logger.debug(f"Cache hit for {symbol} {tf}")
            else:
                logger.debug(f"Cache miss for {symbol} {tf}, fetching data")
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
                    df = fetch_latest_ohlcv_data(
                        symbol=symbol,
                        timeframe=tf,
                        limit=latest_limit,
                        exchange_id=exchange_id
                    )
                _ohlcv_cache[key] = df
        if df.empty:
            logger.warning(f"[OHLCV PIPELINE] {symbol} - {tf} 데이터가 없습니다. 저장 건너뜁니다.")
            return
        table_name = table_name_format.format(symbol=symbol.replace('/', '').lower(), timeframe=tf)
        try:
            insert_ohlcv_records(df, table_name=table_name)
            logger.debug(f"Data inserted for {symbol} {tf} into table {table_name}")
        except Exception as e:
            logger.error(f"[OHLCV PIPELINE] 데이터 저장 에러 ({table_name}): {e}", exc_info=True)
        time.sleep(pause_sec)  # 짧은 대기 시간으로 API rate limit 준수

    tasks = []
    # 최대 동시 작업 수는 상황에 맞게 조정 (여기서는 최대 10개)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for symbol in symbols:
            for tf in timeframes:
                tasks.append(executor.submit(process_symbol_tf, symbol, tf))
        # 모든 작업이 완료될 때까지 대기
        concurrent.futures.wait(tasks)
