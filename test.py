# test.py

import pandas as pd
from data_collection.fetch_binance_data import fetch_binance_historical_ohlcv
from data_collection.postgres_ohlcv_handler import (
    save_ohlcv_to_postgres,
    delete_ohlcv_tables_by_symbol
)

if __name__ == "__main__":
    # (1) 기존 ETH/USDT 관련 테이블 모두 삭제
    print("[*] Deleting existing ETH/USDT tables...")
    delete_ohlcv_tables_by_symbol("ETH/USDT")
    
    # BTC/USDT 설정
    symbol = "BTC/USDT"
    timeframes = ["1d", "4h", "1h"]
    start_date = "2018-01-01 00:00:00"
    end_date = pd.to_datetime("2025-01-21 00:00:00")

    # (2) BTC/USDT에 대해 1d, 4h, 1h 데이터를 수집하여 DB 저장
    for tf in timeframes:
        print(f"\n[*] Fetching historical data: {symbol}, {tf}, start={start_date}")

        # (a) 바이낸스에서 과거 데이터 수집
        df = fetch_binance_historical_ohlcv(
            symbol=symbol,
            timeframe=tf,
            start_date=start_date,
            limit_per_request=1000,
            pause_sec=1.0
        )

        # (b) 2025-01-21 까지만 데이터 필터링
        df = df[df.index <= end_date]
        print(f"    -> Rows (filtered by end_date): {len(df)}")

        # (c) PostgreSQL 저장 (테이블명 예: ohlcv_btcusdt_1d)
        table_name = f"ohlcv_{symbol.replace('/', '').lower()}_{tf}"
        save_ohlcv_to_postgres(df, table_name=table_name)
        print(f"    -> Data saved to table: {table_name}")
