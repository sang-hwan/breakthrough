# data/ohlcv/ohlcv_pipeline.py
import time
from typing import List, Optional
from logs.logger_config import setup_logger
from data.ohlcv.ohlcv_fetcher import fetch_historical_ohlcv_data, fetch_latest_ohlcv_data
from data.db.db_manager import insert_ohlcv_records  # 경로 변경됨

logger = setup_logger(__name__)

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
    심볼과 타임프레임에 따라 OHLCV 데이터를 수집한 후, 데이터베이스에 저장합니다.
    """
    for symbol in symbols:
        for tf in timeframes:
            logger.debug(f"[OHLCV PIPELINE] Fetching {symbol} - {tf} data...")
            try:
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
            except Exception as e:
                logger.error(f"[OHLCV PIPELINE] 데이터 수집 에러 ({symbol} - {tf}): {e}", exc_info=True)
                continue

            logger.debug(f"[OHLCV PIPELINE] -> Total Rows Fetched for {symbol} - {tf}: {len(df)}")
            if df.empty:
                logger.warning(f"[OHLCV PIPELINE] -> {symbol} - {tf} 데이터가 없습니다. 저장 건너뜁니다.")
                continue

            table_name = table_name_format.format(symbol=symbol.replace('/', '').lower(), timeframe=tf)
            try:
                insert_ohlcv_records(df, table_name=table_name)
                logger.debug(f"[OHLCV PIPELINE] -> Saved to table: {table_name}")
            except Exception as e:
                logger.error(f"[OHLCV PIPELINE] 데이터 저장 에러 ({table_name}): {e}", exc_info=True)
            time.sleep(pause_sec)
