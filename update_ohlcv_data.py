# update_ohlcv_data.py
import time
import pandas as pd
from datetime import datetime, timedelta

# DB 관련 모듈 및 데이터 수집 모듈 import
from data_collection.db_manager import fetch_ohlcv_records, insert_ohlcv_records
from data_collection.ohlcv_fetcher import fetch_historical_ohlcv_data

# 업데이트할 심볼 및 시간 프레임 (이전에 시총 상위 3종, 예: BTC/USDT, ETH/USDT, BNB/USDT)
symbols = ["ETH/USDT", "BTC/USDT", "XRP/USDT"]
# 시간 프레임: 1일, 4시간, 1시간, 15분
timeframes = ["1d", "4h", "1h", "15min"]

def update_data_for_symbol_timeframe(symbol: str, timeframe: str) -> None:
    # 테이블명은 시간 프레임에서 'min'을 'm'으로 변경하여 사용
    table_name = f"ohlcv_{symbol.replace('/', '').lower()}_{timeframe.replace('min', 'm')}"
    print(f"\n[*] {symbol} - {timeframe} 데이터 업데이트를 시작합니다...")
    
    # 기존 데이터 조회: DB에 저장된 마지막 timestamp를 가져옴
    try:
        df_existing = fetch_ohlcv_records(table_name=table_name)
    except Exception as e:
        print(f"  [!] {table_name} 데이터 조회 중 오류 발생: {e}")
        df_existing = pd.DataFrame()
    
    if not df_existing.empty:
        last_timestamp = df_existing.index.max()
        if timeframe.endswith('d'):
            delta = timedelta(days=int(timeframe[:-1]))
        elif timeframe.endswith('h'):
            delta = timedelta(hours=int(timeframe[:-1]))
        elif timeframe.endswith('min'):
            delta = timedelta(minutes=int(timeframe[:-3]))
        else:
            delta = timedelta(0)
            
        new_start = last_timestamp + delta
        if new_start > datetime.now():
            print(f"  [*] {table_name}는 이미 최신 데이터입니다. (마지막 시간: {last_timestamp})")
            return
        start_date_str = new_start.strftime("%Y-%m-%d %H:%M:%S")
    else:
        start_date_str = "2018-01-01 00:00:00"
    
    print(f"  [*] {table_name}의 최신 데이터를 {start_date_str}부터 현재({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})까지 가져옵니다.")
    df_new = fetch_historical_ohlcv_data(symbol=symbol, timeframe=timeframe, start_date=start_date_str)
    
    if df_new.empty:
        print(f"  [*] {table_name}에 추가할 새로운 데이터가 없습니다.")
        return
    
    print(f"  [*] {table_name}에 {len(df_new)}건의 신규 데이터가 조회되었습니다. DB에 삽입합니다...")
    try:
        insert_ohlcv_records(df_new, table_name=table_name)
        print(f"  [*] {table_name} 업데이트 완료.")
    except Exception as e:
        print(f"  [!] {table_name} 데이터 삽입 중 오류 발생: {e}")
    
    time.sleep(1)

def main():
    for symbol in symbols:
        for tf in timeframes:
            update_data_for_symbol_timeframe(symbol, tf)

if __name__ == "__main__":
    main()
