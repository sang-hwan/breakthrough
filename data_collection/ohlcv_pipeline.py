# data_collection/ohlcv_pipeline.py

import time
from typing import List, Optional

from data_collection.ohlcv_data_fetcher import (
    fetch_historical_ohlcv_data,
    fetch_latest_ohlcv_data
)
from data_collection.db_ohlcv_manager import insert_ohlcv_records

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
    지정한 심볼과 타임프레임에 대해 데이터를 수집하여 PostgreSQL에 저장합니다.
    table_name_format: 예) "ohlcv_{symbol}_{timeframe}" 형식
    """
    for symbol in symbols:
        for tf in timeframes:
            print(f"\n[*] Fetching {symbol} - {tf} data...")

            if use_historical:
                if not start_date:
                    raise ValueError("start_date는 과거 데이터 수집 시 반드시 필요합니다.")
                df = fetch_historical_ohlcv_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    limit_per_request=limit_per_request,
                    pause_sec=pause_sec,
                    exchange_id=exchange_id,
                    time_offset_ms=time_offset_ms
                )
            else:
                df = fetch_latest_ohlcv_data(
                    symbol=symbol,
                    timeframe=tf,
                    limit=latest_limit,
                    exchange_id=exchange_id
                )

            # 테이블 이름 생성 (table_name_format 활용)
            table_name = table_name_format.format(symbol=symbol.replace('/', '').lower(), timeframe=tf)
            print(f"    -> Total Rows Fetched: {len(df)}")

            insert_ohlcv_records(df, table_name=table_name)
            print(f"    -> Saved to table: {table_name}")

            time.sleep(pause_sec)
