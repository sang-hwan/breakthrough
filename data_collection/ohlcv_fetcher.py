# data_collection/ohlcv_fetcher.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_historical_ohlcv_data(symbol: str, timeframe: str, start_date: str, limit_per_request: int = 1000, pause_sec: float = 1.0, exchange_id: str = 'binance', time_offset_ms: int = 1) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = datetime.now()
    date_range = pd.date_range(start, end, freq=timeframe)
    data = {
        "open": np.random.random(len(date_range)) * 100,
        "high": np.random.random(len(date_range)) * 100,
        "low": np.random.random(len(date_range)) * 100,
        "close": np.random.random(len(date_range)) * 100,
        "volume": np.random.random(len(date_range)) * 10
    }
    df = pd.DataFrame(data, index=date_range)
    df.index.name = "timestamp"
    return df

def fetch_latest_ohlcv_data(symbol: str, timeframe: str, limit: int = 500, exchange_id: str = 'binance') -> pd.DataFrame:
    end = datetime.now()
    freq = timeframe
    date_range = pd.date_range(end - pd.Timedelta(hours=limit), end, freq=freq)
    data = {
        "open": np.random.random(len(date_range)) * 100,
        "high": np.random.random(len(date_range)) * 100,
        "low": np.random.random(len(date_range)) * 100,
        "close": np.random.random(len(date_range)) * 100,
        "volume": np.random.random(len(date_range)) * 10
    }
    df = pd.DataFrame(data, index=date_range)
    df.index.name = "timestamp"
    return df
