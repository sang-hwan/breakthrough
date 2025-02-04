# insert_ohlcv.py

import time
from datetime import datetime
import pandas as pd
from data_collection.db_manager import insert_ohlcv_records
from data_collection.ohlcv_fetcher import fetch_historical_ohlcv_data, get_top_market_cap_symbols

# 수집할 타임프레임 목록 (Binance의 경우, 올바른 인터벌은 "15m"임)
TIMEFRAMES = ["1d", "4h", "1h", "15m"]
START_DATE = "2018-06-01"

def insert_ohlcv_for_symbol(symbol: str, timeframes: list, start_date: str, pause_sec: float = 1.0, exchange_id: str = 'binance'):
    for tf in timeframes:
        # 데이터 수집 시, Binance에서는 "15m" 그대로 사용해야 함.
        fetch_tf = tf  
        print(f"\n{symbol}의 {tf} 데이터 수집 시작 (시작일: {start_date})...")
        df = fetch_historical_ohlcv_data(symbol, fetch_tf, start_date, exchange_id=exchange_id, pause_sec=pause_sec)
        if df.empty:
            print(f"{symbol}의 {tf} 데이터가 없습니다. 해당 타임프레임은 스킵합니다.")
            continue
        # 테이블명은 원래 타임프레임 값("15m")을 사용
        table_name = f"ohlcv_{symbol.replace('/', '').lower()}_{tf}"
        try:
            insert_ohlcv_records(df, table_name=table_name)
            print(f"{symbol}의 {tf} 데이터가 '{table_name}' 테이블에 저장되었습니다.")
        except Exception as e:
            print(f"{symbol}의 {tf} 데이터를 '{table_name}'에 저장하는 중 에러 발생: {e}")
        time.sleep(pause_sec)

def main():
    print("시가총액(대용: 거래량 기준) 상위 심볼 중 2018-06-01 이전 데이터가 있는 심볼을 찾습니다...")
    valid_symbols = get_top_market_cap_symbols(required_start_date=START_DATE, count=3)
    
    if not valid_symbols or len(valid_symbols) < 3:
        print(f"{START_DATE} 이전 데이터가 있는 유효한 심볼이 5개 미만입니다. 수집을 중단합니다.")
        return
    
    print("최종 유효 심볼:", valid_symbols)
    
    for symbol in valid_symbols:
        insert_ohlcv_for_symbol(symbol, TIMEFRAMES, START_DATE)
    
    print("\n모든 심볼의 OHLCV 데이터 삽입 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
