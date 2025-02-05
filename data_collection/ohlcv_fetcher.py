# data_collection/ohlcv_fetcher.py
import ccxt
import pandas as pd
from datetime import datetime
import time
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def fetch_historical_ohlcv_data(symbol: str, timeframe: str, start_date: str, 
                                limit_per_request: int = 1000, pause_sec: float = 1.0, 
                                exchange_id: str = 'binance', single_fetch: bool = False,
                                time_offset_ms: int = 1):
    """
    ccxt를 이용해 지정한 심볼, 타임프레임, 시작일로부터 OHLCV 데이터를 수집합니다.
    만약 single_fetch=True이면, 한 번의 요청만 수행합니다.
    
    반환된 DataFrame은 'open', 'high', 'low', 'close', 'volume' 컬럼을 가지며, 
    인덱스는 timestamp입니다.
    
    [time_offset_ms 설명]
    → 마지막으로 수집한 타임스탬프에 time_offset_ms(밀리초 단위)를 더하여,
       중복된 데이터 수집을 방지하기 위해 사용됩니다.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        exchange.load_markets()
    except Exception as e:
        logger.error(f"Exchange '{exchange_id}' 초기화 에러: {e}")
        return pd.DataFrame()

    try:
        since = exchange.parse8601(datetime.strptime(start_date, "%Y-%m-%d").isoformat())
    except Exception as e:
        logger.error(f"start_date ({start_date}) 파싱 에러: {e}")
        return pd.DataFrame()

    ohlcv_list = []
    while True:
        try:
            ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_request)
        except Exception as e:
            logger.error(f"{symbol}의 {timeframe} 데이터 수집 에러: {e}")
            break

        if not ohlcvs:
            break

        ohlcv_list.extend(ohlcvs)
        
        # single_fetch 옵션이 True이면 한 번의 요청 후 종료
        if single_fetch:
            break

        last_timestamp = ohlcvs[-1][0]
        since = last_timestamp + time_offset_ms  # 중복 방지를 위해 offset 적용

        # 현재 시간보다 과거 데이터가 모두 수집되었으면 종료
        if last_timestamp >= exchange.milliseconds():
            break

        time.sleep(pause_sec)

    if ohlcv_list:
        df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        return pd.DataFrame()
    
def fetch_latest_ohlcv_data(symbol: str, timeframe: str, limit: int = 500, exchange_id: str = 'binance'):
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        exchange.load_markets()
    except Exception as e:
        logger.error(f"Exchange '{exchange_id}' 초기화 에러: {e}")
        return pd.DataFrame()
    
    try:
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        logger.error(f"{symbol}의 {timeframe} 최신 데이터 수집 에러: {e}")
        return pd.DataFrame()
    
    if ohlcvs:
        df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        return pd.DataFrame()
    
def get_top_market_cap_symbols(exchange_id: str = 'binance', quote_currency: str = 'USDT', 
                               required_start_date: str = "2018-01-01", count: int = 5, 
                               pause_sec: float = 1.0):
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        markets = exchange.load_markets()
    except Exception as e:
        logger.error(f"{exchange_id}에서 마켓 로드 에러: {e}")
        return []

    usdt_symbols = [symbol for symbol in markets if symbol.endswith('/' + quote_currency)]
    
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        logger.error(f"티커 수집 에러: {e}")
        tickers = {}

    symbol_volumes = []
    for symbol in usdt_symbols:
        ticker = tickers.get(symbol, {})
        volume = ticker.get('quoteVolume', 0)
        symbol_volumes.append((symbol, volume))
    
    symbol_volumes.sort(key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
    
    valid_symbols = []
    for symbol, volume in symbol_volumes:
        logger.info(f"심볼 {symbol}의 데이터 가용성 확인 중 (시작일: {required_start_date})...")
        df = fetch_historical_ohlcv_data(symbol, '1d', required_start_date, 
                                         limit_per_request=1, pause_sec=pause_sec, 
                                         exchange_id=exchange_id, single_fetch=True)
        if df.empty:
            logger.info(f"  → {symbol}은(는) {required_start_date} 이후 데이터만 존재하거나 데이터가 없음. 스킵합니다.")
            continue
        first_timestamp = df.index.min()
        if first_timestamp > pd.to_datetime(required_start_date):
            logger.info(f"  → {symbol}은(는) {required_start_date} 이후 상장됨 (최초 데이터: {first_timestamp}). 스킵합니다.")
            continue
        valid_symbols.append(symbol)
        if len(valid_symbols) >= count:
            break

    if len(valid_symbols) < count:
        logger.warning(f"경고: {required_start_date} 이전 데이터가 있는 유효 심볼이 {len(valid_symbols)}개 밖에 없습니다.")
    return valid_symbols
