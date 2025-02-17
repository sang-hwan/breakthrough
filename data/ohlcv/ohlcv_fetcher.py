# data/ohlcv/ohlcv_fetcher.py
import ccxt
import pandas as pd
from datetime import datetime
import time
from logs.logger_config import setup_logger
from functools import lru_cache

logger = setup_logger(__name__)

@lru_cache(maxsize=32)
def fetch_historical_ohlcv_data(symbol: str, timeframe: str, start_date: str, 
                                limit_per_request: int = 1000, pause_sec: float = 1.0, 
                                exchange_id: str = 'binance', single_fetch: bool = False,
                                time_offset_ms: int = 1, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for the specified symbol and timeframe starting from start_date.
    Uses caching to reduce duplicate API calls.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        exchange.load_markets()
    except Exception as e:
        logger.error(f"Exchange '{exchange_id}' 초기화 에러: {e}", exc_info=True)
        return pd.DataFrame()

    try:
        since = exchange.parse8601(datetime.strptime(start_date, "%Y-%m-%d").isoformat())
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
                logger.error(f"최대 재시도({max_retries}) 초과로 {symbol} {timeframe} 데이터 수집 중단")
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
            return df.copy()  # Return a copy to prevent external modifications
        except Exception as e:
            logger.error(f"DataFrame 변환 에러: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.warning(f"{symbol} {timeframe}에 대한 데이터가 없습니다.")
        return pd.DataFrame()

@lru_cache(maxsize=32)
def fetch_latest_ohlcv_data(symbol: str, timeframe: str, limit: int = 500, exchange_id: str = 'binance') -> pd.DataFrame:
    """
    Fetch the latest OHLCV data for the specified symbol and timeframe.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
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

def get_top_market_cap_symbols(exchange_id: str = 'binance', quote_currency: str = 'USDT', 
                               required_start_date: str = "2018-01-01", count: int = 5, 
                               pause_sec: float = 1.0) -> list:
    """
    Retrieve top market cap symbols based on trading volume that have historical data
    starting from the required_start_date.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        markets = exchange.load_markets()
    except Exception as e:
        logger.error(f"{exchange_id}에서 마켓 로드 에러: {e}", exc_info=True)
        return []

    usdt_symbols = [symbol for symbol in markets if symbol.endswith('/' + quote_currency)]
    
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        logger.error(f"티커 수집 에러: {e}", exc_info=True)
        tickers = {}

    symbol_volumes = []
    for symbol in usdt_symbols:
        ticker = tickers.get(symbol, {})
        volume = ticker.get('quoteVolume', 0)
        symbol_volumes.append((symbol, volume))
    
    symbol_volumes.sort(key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
    
    valid_symbols = []
    for symbol, volume in symbol_volumes:
        df = fetch_historical_ohlcv_data(symbol, '1d', required_start_date, 
                                         limit_per_request=1, pause_sec=pause_sec, 
                                         exchange_id=exchange_id, single_fetch=True)
        if df.empty:
            continue
        first_timestamp = df.index.min()
        if first_timestamp > pd.to_datetime(required_start_date):
            continue
        valid_symbols.append(symbol)
        if len(valid_symbols) >= count:
            break

    if len(valid_symbols) < count:
        logger.warning(f"경고: {required_start_date} 이전 데이터가 있는 유효 심볼이 {len(valid_symbols)}개 밖에 없습니다.")
    return valid_symbols
