# collect_data.py

import datetime
import time
import pandas as pd
from db_manager import store_ohlcv_to_db
from fetch_binance_data import fetch_binance_ohlcv


def collect_data_for_symbols(symbols, timeframes, limit=1000):
    """
    여러 심볼/타임프레임에 대해 바이낸스의 OHLCV 데이터를 수집하여 PostgreSQL에 저장하는 함수.
    """
    for sym in symbols:
        for tf in timeframes:
            print(f"Fetching {sym} / {tf}")
            df = fetch_binance_ohlcv(sym, timeframe=tf, limit=limit)
            if not df.empty:
                store_ohlcv_to_db(df, symbol=sym, timeframe=tf)
            else:
                print(f"No data fetched for {sym} / {tf}.")
            time.sleep(0.5)  # API 호출 시 짧은 텀(쿨다운)을 주는 것을 권장


if __name__ == "__main__":
    symbols_list = ["BTC/USDT", "ETH/USDT"]  # 예시
    timeframes_list = ["1h", "4h", "1d"]    # 예시
    
    # 예시: 1000개 캔들씩만 가져오는 단순 버전
    collect_data_for_symbols(symbols_list, timeframes_list, limit=1000)

    # * 더 많은 데이터가 필요하다면,
    #   '시작 시점'부터 '끝 시점'까지 반복해서 fetch_ohlcv를 호출하는 로직을 추가해야 합니다.
