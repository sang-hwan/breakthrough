# data/collect_data.py
"""
이 모듈은 ccxt 라이브러리를 사용하여 거래소에서 OHLCV 데이터를 수집하는 기능을 제공합니다.
수집된 데이터는 pandas DataFrame으로 반환되며, lru_cache를 통해 동일 호출 시 중복 API 요청을 방지합니다.
"""

import ccxt
import pandas as pd
from datetime import datetime
import time
from functools import lru_cache
from typing import Optional
from logs.log_config import setup_logger

logger = setup_logger(__name__)

@lru_cache(maxsize=32)
def collect_historical_ohlcv_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    limit_per_request: int = 1000,
    pause_sec: float = 1.0,
    exchange_id: str = 'binance',
    single_fetch: bool = False,
    time_offset_ms: int = 1,
    max_retries: int = 3,
    exchange_instance: Optional[ccxt.Exchange] = None
) -> pd.DataFrame:
    """
    거래소에서 지정 심볼과 시간 간격에 대해 시작 날짜부터의 OHLCV 데이터를 수집합니다.

    Parameters:
        symbol (str): 거래 심볼 (예: 'BTC/USDT').
        timeframe (str): 시간 간격 (예: '1d', '1h').
        start_date (str): 데이터 수집 시작일 ("YYYY-MM-DD" 혹은 "YYYY-MM-DD HH:MM:SS").
        limit_per_request (int): 한 번 호출 시 가져올 데이터 수.
        pause_sec (float): API 호출 사이의 대기 시간.
        exchange_id (str): 거래소 ID (기본 'binance').
        single_fetch (bool): 단일 호출 여부.
        time_offset_ms (int): 다음 호출 시 타임스탬프 오프셋.
        max_retries (int): 최대 재시도 횟수.
        exchange_instance (Optional[ccxt.Exchange]): 재사용할 거래소 인스턴스.

    Returns:
        pd.DataFrame: 수집된 OHLCV 데이터를 담은 DataFrame.
    """
    if exchange_instance is None:
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
            logger.debug(f"{exchange_id}: Loading markets...")
            exchange.load_markets()
            logger.debug(f"{exchange_id}: Markets loaded successfully.")
        except Exception as e:
            logger.error(f"Exchange '{exchange_id}' 초기화 에러: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        exchange = exchange_instance

    try:
        # 날짜 형식이 YYYY-MM-DD이면 시간 정보를 추가합니다.
        if len(start_date.strip()) == 10:
            start_date += " 00:00:00"
        since = exchange.parse8601(datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").isoformat())
    except Exception as e:
        logger.error(f"start_date ({start_date}) 파싱 에러: {e}", exc_info=True)
        return pd.DataFrame()

    ohlcv_list = []
    retry_count = 0
    while True:
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_request)
        except Exception as e:
            logger.error(f"{symbol}의 {timeframe} 데이터 수집 에러: {e}", exc_info=True)
            retry_count += 1
            if retry_count >= max_retries:
                logger.error(f"최대 재시도({max_retries}) 초과 - {symbol} {timeframe} 데이터 수집 중단")
                break
            time.sleep(pause_sec)
            continue

        if not ohlcvs:
            break

        ohlcv_list.extend(ohlcvs)
        
        if single_fetch:
            break

        last_timestamp = ohlcvs[-1][0]
        since = last_timestamp + time_offset_ms

        if last_timestamp >= exchange.milliseconds():
            break

        time.sleep(pause_sec)

    if ohlcv_list:
        try:
            df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df.copy()
        except Exception as e:
            logger.error(f"DataFrame 변환 에러: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.warning(f"{symbol} {timeframe}에 대한 데이터가 없습니다.")
        return pd.DataFrame()

@lru_cache(maxsize=32)
def collect_latest_ohlcv_data(
    symbol: str,
    timeframe: str,
    limit: int = 500,
    exchange_id: str = 'binance'
) -> pd.DataFrame:
    """
    거래소에서 지정 심볼과 시간 간격에 대해 최신 OHLCV 데이터를 수집합니다.

    Parameters:
        symbol (str): 거래 심볼.
        timeframe (str): 시간 간격.
        limit (int): 가져올 데이터 수 제한.
        exchange_id (str): 거래소 ID.

    Returns:
        pd.DataFrame: 최신 OHLCV 데이터를 담은 DataFrame.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
        exchange.load_markets()
    except Exception as e:
        logger.error(f"Exchange '{exchange_id}' 초기화 에러: {e}", exc_info=True)
        return pd.DataFrame()
    
    try:
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        logger.error(f"{symbol}의 {timeframe} 최신 데이터 수집 에러: {e}", exc_info=True)
        return pd.DataFrame()
    
    if ohlcvs:
        try:
            df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df.copy()
        except Exception as e:
            logger.error(f"DataFrame 변환 에러: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.warning(f"{symbol} {timeframe}에 대한 최신 데이터가 없습니다.")
        return pd.DataFrame()
