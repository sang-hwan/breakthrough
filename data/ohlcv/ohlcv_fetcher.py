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
                                time_offset_ms: int = 1, max_retries: int = 3,
                                exchange_instance=None) -> pd.DataFrame:
    """
    지정 심볼과 timeframe에 대해 start_date부터의 역사적 OHLCV 데이터를 가져옵니다.
    exchange_instance가 제공되면 이를 재사용합니다.
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
            return df.copy()
        except Exception as e:
            logger.error(f"DataFrame 변환 에러: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.warning(f"{symbol} {timeframe}에 대한 데이터가 없습니다.")
        return pd.DataFrame()

@lru_cache(maxsize=32)
def fetch_latest_ohlcv_data(symbol: str, timeframe: str, limit: int = 500, exchange_id: str = 'binance') -> pd.DataFrame:
    """
    지정 심볼과 timeframe에 대해 최신 OHLCV 데이터를 가져옵니다.
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
                           count: int = 5, pause_sec: float = 1.0) -> list:
    """
    Binance의 경우 거래량(quoteVolume)을 시총 대용 지표로 사용하여,
    거래소 내에서 유효한 심볼 중 상위 'count' 개를 선택합니다.
    스테이블 코인(예: USDT, BUSD, USDC 등)은 모두 제외합니다.
    
    각 심볼에 대해 fetch_historical_ohlcv_data() (limit=1)를 호출하여 최초 제공일을 구한 후,
    (symbol, first_available_date) 튜플 리스트로 반환합니다.
    
    (파일 간 호환: 기존 required_start_date 매개변수가 제거되었습니다.)
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

    # USDT 심볼 추출
    usdt_symbols = [symbol for symbol in markets if symbol.endswith('/' + quote_currency)]
    
    stable_coins = {
        # 기존 주요 스테이블 코인
        "USDT", "BUSD", "USDC", "DAI", "TUSD", "PAX", "USDP", "GUSD", "MIM", "UST",
        
        # 최신 주요 스테이블 코인 추가
        "USDe",  # Ethena USD (synthetic dollar)
        "FDUSD",  # First Digital USD
        "PYUSD",  # PayPal USD
        "USD0",  # Usual USD
        "FRAX",  # Frax Finance Stablecoin
        "USDY",  # Ondo USD Yield
        "USDD",  # Tron-based stablecoin
        "EURS",  # Stasis Euro stablecoin

        # 추가된 주요 스테이블 코인 (2023~2025년 신규)
        "RLUSD",  # Ripple USD
        "GHO",  # Aave Stablecoin
        "crvUSD",  # Curve Finance Stablecoin
        "LUSD",  # Liquity USD (ETH-backed stablecoin)
        "XAU₮",  # Tether Gold (Gold-backed stablecoin)
        "PAXG",  # Paxos Gold
        "EUROC",  # Circle Euro Coin
    }
    
    filtered_symbols = [symbol for symbol in usdt_symbols if symbol.split('/')[0] not in stable_coins]
    logger.debug(f"Filtered symbols (excluding stablecoins): {len(filtered_symbols)} out of {len(usdt_symbols)}")

    # Binance의 경우 거래량을 시총 대신 사용합니다.
    if exchange_id == 'binance':
        try:
            tickers = exchange.fetch_tickers()
        except Exception as e:
            logger.error(f"Error fetching tickers from {exchange_id}: {e}", exc_info=True)
            tickers = {}
        symbol_volumes = []
        for symbol in filtered_symbols:
            ticker = tickers.get(symbol, {})
            vol = ticker.get('quoteVolume', 0)
            try:
                vol = float(vol)
            except Exception:
                vol = 0
            symbol_volumes.append((symbol, vol))
        symbol_volumes.sort(key=lambda x: x[1], reverse=True)
        sorted_symbols = [item[0] for item in symbol_volumes]
    else:
        # Binance가 아닌 경우, 기존 marketCap 정보를 사용합니다.
        symbol_caps = []
        for symbol in filtered_symbols:
            market_info = markets.get(symbol, {})
            cap = market_info.get('info', {}).get('marketCap')
            if cap is None:
                cap = 0
            try:
                cap = float(cap)
            except Exception:
                cap = 0
            symbol_caps.append((symbol, cap))
        symbol_caps.sort(key=lambda x: x[1], reverse=True)
        sorted_symbols = [item[0] for item in symbol_caps]

    valid_symbols = []
    # 모든 심볼에 대해 첫 데이터 제공일을 가져옵니다.
    for symbol in sorted_symbols:
        # 확인용 start_date는 "2018-01-01"으로 고정합니다.
        df = fetch_historical_ohlcv_data(symbol, '1d', "2018-01-01", 
                                         limit_per_request=1, pause_sec=pause_sec, 
                                         exchange_id=exchange_id, single_fetch=True,
                                         exchange_instance=exchange)
        if df.empty:
            continue
        first_timestamp = df.index.min()
        first_date_str = first_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        valid_symbols.append((symbol, first_date_str))
        if len(valid_symbols) >= count:
            break

    if len(valid_symbols) < count:
        logger.warning(f"경고: 유효한 데이터가 있는 심볼이 {len(valid_symbols)}개 밖에 없습니다.")
    return valid_symbols

def get_latest_onboard_date(symbols: list, exchange_id: str = 'binance') -> str:
    """
    전달된 심볼 리스트(튜플 또는 단순 심볼 문자열)에 대해 최초 데이터 제공일(=onboardDate)을 구합니다.
    Binance의 경우 get_top_volume_symbols에서 (symbol, first_available_date) 튜플로 전달할 수 있으며,
    이를 이용하여 가장 늦은(최근) 날짜를 구합니다.
    
    (파일 간 호환: 기존 get_latest_onboard_date의 매개변수 타입과 동작이 변경되었습니다.)
    """
    onboard_dates = []
    try:
        for item in symbols:
            if isinstance(item, tuple):
                onboard_str = item[1]
            else:
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
                exchange.load_markets()
                df = fetch_historical_ohlcv_data(item, '1d', "2018-01-01", 
                                                 limit_per_request=1, single_fetch=True, 
                                                 exchange_instance=exchange)
                if df.empty:
                    onboard_str = "2018-01-01 00:00:00"
                else:
                    onboard_str = df.index.min().strftime("%Y-%m-%d %H:%M:%S")
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
