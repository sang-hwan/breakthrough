# data_collection/ohlcv_data_fetcher.py

import ccxt
import pandas as pd
import time
from datetime import datetime

def fetch_historical_ohlcv_data(
    symbol: str,
    timeframe: str = '4h',
    start_date: str = '2021-01-01 00:00:00',
    limit_per_request: int = 1000,
    pause_sec: float = 1.0,
    exchange_id: str = 'binance',
    time_offset_ms: int = 1
) -> pd.DataFrame:
    """
    거래소에서 과거 가격 데이터를 여러 번 나눠 요청하여 대량으로 수집합니다.
    추가 매개변수:
      - exchange_id: 사용할 거래소 ID (기본 'binance')
      - time_offset_ms: 마지막 데이터 이후 요청 시 오프셋 (기본 1ms)
    """
    exchange = getattr(ccxt, exchange_id)()
    since_ms = exchange.parse8601(start_date)
    all_ohlcv = []

    while True:
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=limit_per_request
        )

        if not ohlcv:
            break

        all_ohlcv += ohlcv
        last_ts = ohlcv[-1][0]
        since_ms = last_ts + time_offset_ms  # 시간 오프셋 적용

        time.sleep(pause_sec)

        if len(ohlcv) < limit_per_request:
            break

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df[~df.index.duplicated()]
    df.dropna(inplace=True)

    return df

def fetch_latest_ohlcv_data(
    symbol: str,
    timeframe: str = '4h',
    limit: int = 500,
    exchange_id: str = 'binance'
) -> pd.DataFrame:
    """
    거래소에서 가장 최근 시세 데이터를 가져옵니다.
    추가 매개변수:
      - exchange_id: 사용할 거래소 ID (기본 'binance')
    """
    exchange = getattr(ccxt, exchange_id)()
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df = df[~df.index.duplicated()]
    df.dropna(inplace=True)

    return df
