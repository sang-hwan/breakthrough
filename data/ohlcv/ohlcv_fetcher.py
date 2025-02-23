# data/ohlcv/ohlcv_fetcher.py
import ccxt
import pandas as pd
from datetime import datetime
import time
from logs.logger_config import setup_logger
from functools import lru_cache

logger = setup_logger(__name__)

def extract_onboard_date(market_info) -> str:
    """
    Extract onboardDate from the given market_info dictionary.
    Returns a string in "%Y-%m-%d %H:%M:%S" format.
    If unavailable or error occurs, returns the default "2018-01-01 00:00:00".
    """
    default_date = "2018-01-01 00:00:00"
    if market_info:
        onboard = market_info.get('info', {}).get('onboardDate')
        if onboard:
            try:
                onboard_dt = datetime.fromtimestamp(int(onboard) / 1000)
                return onboard_dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return default_date
    return default_date

@lru_cache(maxsize=32)
def fetch_historical_ohlcv_data(symbol: str, timeframe: str, start_date: str, 
                                limit_per_request: int = 1000, pause_sec: float = 1.0, 
                                exchange_id: str = 'binance', single_fetch: bool = False,
                                time_offset_ms: int = 1, max_retries: int = 3,
                                exchange_instance=None) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for the specified symbol and timeframe starting from start_date.
    Uses caching to reduce duplicate API calls.
    
    If exchange_instance is provided, it will be reused instead of creating a new instance.
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
        # 만약 start_date가 "YYYY-MM-DD" 형식이면 시간 부분(" 00:00:00")을 추가합니다.
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

def get_top_volume_symbols(exchange_id: str = 'binance', quote_currency: str = 'USDT', 
                           required_start_date: str = "2018-01-01", count: int = 5, 
                           pause_sec: float = 1.0) -> list:
    """
    Retrieve the top symbols based on trading volume (quoteVolume) that have historical data
    available since the required_start_date.
    
    This function creates a single ccxt exchange instance to load market data and tickers,
    sorts the USDT symbols by their quoteVolume in descending order, and filters out symbols
    without data prior to the required_start_date. It returns a list of tuples, where each tuple
    contains (symbol, onboard_date) as a string in "%Y-%m-%d %H:%M:%S" format.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
        logger.debug(f"{exchange_id}: Loading markets for top symbol retrieval...")
        markets = exchange.load_markets()
        logger.debug(f"{exchange_id}: Markets loaded, total symbols: {len(markets)}")
    except Exception as e:
        logger.error(f"{exchange_id}에서 마켓 로드 에러: {e}", exc_info=True)
        return []

    usdt_symbols = [symbol for symbol in markets if symbol.endswith('/' + quote_currency)]
    logger.debug(f"Found {len(usdt_symbols)} USDT symbols.")

    try:
        logger.debug("Fetching tickers...")
        tickers = exchange.fetch_tickers()
        logger.debug("Tickers fetched successfully.")
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
                                         exchange_id=exchange_id, single_fetch=True,
                                         exchange_instance=exchange)
        if df.empty:
            continue
        first_timestamp = df.index.min()
        if first_timestamp > pd.to_datetime(required_start_date):
            continue
        # onboardDate 추출
        market_info = exchange.markets.get(symbol)
        onboard_str = extract_onboard_date(market_info)
        valid_symbols.append((symbol, onboard_str))
        if len(valid_symbols) >= count:
            break

    if len(valid_symbols) < count:
        logger.warning(f"경고: {required_start_date} 이전 데이터가 있는 유효 심볼이 {len(valid_symbols)}개 밖에 없습니다.")
    return valid_symbols

def get_latest_onboard_date(symbols: list, exchange_id: str = 'binance') -> str:
    """
    For the given list of symbols, retrieve each symbol's onboardDate using extract_onboard_date,
    and return the latest date as a string in "%Y-%m-%d %H:%M:%S" format.
    If retrieval fails, returns the default "2018-01-01 00:00:00".
    """
    onboard_dates = []
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
        exchange.load_markets()
        for item in symbols:
            # symbols may be a list of tuples (symbol, onboard_date) or just symbol strings.
            if isinstance(item, tuple):
                symbol = item[0]
            else:
                symbol = item
            market_info = exchange.markets.get(symbol)
            onboard_str = extract_onboard_date(market_info)
            try:
                dt = datetime.strptime(onboard_str, "%Y-%m-%d %H:%M:%S")
                onboard_dates.append(dt)
            except Exception:
                onboard_dates.append(datetime.strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"))
        if onboard_dates:
            latest_date = max(onboard_dates)
            return latest_date.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error in get_latest_onboard_date: {e}", exc_info=True)
    return "2018-01-01 00:00:00"
