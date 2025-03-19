# data/fetcher/fetch_data.py

import ccxt  # 암호화폐 거래소 API 연동을 위한 라이브러리
import pandas as pd
from datetime import datetime
import time  # 시간 지연을 위한 모듈
from logs.log_config import setup_logger  # 로깅 설정 모듈
from functools import lru_cache  # 함수 결과 캐싱을 위한 모듈

# 전역 변수: 로깅 설정 객체
logger = setup_logger(__name__)

@lru_cache(maxsize=32)  # 캐시를 통해 동일 파라미터 호출 시 중복 API 호출 방지
def fetch_historical_ohlcv_data(symbol: str, timeframe: str, start_date: str, 
                                limit_per_request: int = 1000, pause_sec: float = 1.0, 
                                exchange_id: str = 'binance', single_fetch: bool = False,
                                time_offset_ms: int = 1, max_retries: int = 3,
                                exchange_instance=None) -> pd.DataFrame:
    """
    지정된 심볼(symbol)과 시간 간격(timeframe)에 대해, start_date부터의 역사적 OHLCV 데이터를 ccxt를 통해 수집합니다.
    선택적으로 exchange_instance를 재사용하여 API 호출 횟수를 줄입니다.
    
    Parameters:
        symbol (str): 거래 심볼 (예: 'BTC/USDT').
        timeframe (str): 시간 간격 (예: '1d', '1h').
        start_date (str): 데이터 수집 시작일 ("YYYY-MM-DD" 또는 "YYYY-MM-DD HH:MM:SS").
        limit_per_request (int): 한 번 호출 시 가져올 데이터 개수 (기본값 1000).
        pause_sec (float): API 호출 간 대기 시간 (기본값 1.0초).
        exchange_id (str): 사용 거래소 ID (기본값 'binance').
        single_fetch (bool): 한 번의 호출만 수행할지 여부.
        time_offset_ms (int): 다음 호출 시 타임스탬프 오프셋 (밀리초 단위).
        max_retries (int): 최대 재시도 횟수 (기본값 3).
        exchange_instance: 재사용할 ccxt 거래소 인스턴스 (옵션).
    
    Returns:
        pd.DataFrame: OHLCV 데이터를 포함하는 DataFrame. 컬럼은 ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    # exchange_instance가 제공되지 않은 경우, 새로운 ccxt 거래소 인스턴스 생성 및 마켓 로드
    if exchange_instance is None:
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
            logger.debug(f"{exchange_id}: Loading markets...")
            exchange.load_markets()  # 해당 거래소의 마켓 정보를 불러옴
            logger.debug(f"{exchange_id}: Markets loaded successfully.")
        except Exception as e:
            logger.error(f"Exchange '{exchange_id}' 초기화 에러: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        exchange = exchange_instance

    try:
        # 시작일이 "YYYY-MM-DD" 형식이면 " 00:00:00"을 추가하여 완전한 datetime 문자열로 만듦
        if len(start_date.strip()) == 10:
            start_date += " 00:00:00"
        # 시작일을 ISO 포맷으로 변환한 후, ccxt의 parse8601를 이용하여 밀리초 단위 timestamp로 변환
        since = exchange.parse8601(datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").isoformat())
    except Exception as e:
        logger.error(f"start_date ({start_date}) 파싱 에러: {e}", exc_info=True)
        return pd.DataFrame()

    ohlcv_list = []  # 수집된 OHLCV 데이터를 저장할 리스트
    retry_count = 0  # 재시도 횟수 초기화
    while True:
        try:
            # 지정된 심볼, 시간 간격, 시작 시점(since), 요청당 데이터 제한을 기준으로 OHLCV 데이터 호출
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
            # 더 이상 데이터가 없으면 반복문 종료
            break

        ohlcv_list.extend(ohlcvs)
        
        if single_fetch:
            # 한 번의 호출만 필요한 경우 반복 종료
            break

        # 마지막 데이터의 타임스탬프를 확인하여, 다음 호출 시 시작점을 업데이트
        last_timestamp = ohlcvs[-1][0]
        since = last_timestamp + time_offset_ms

        # 만약 마지막 타임스탬프가 현재 시간(밀리초) 이상이면 더 이상 데이터를 요청하지 않음
        if last_timestamp >= exchange.milliseconds():
            break

        time.sleep(pause_sec)  # API rate limit 준수를 위한 대기

    if ohlcv_list:
        try:
            # 리스트를 DataFrame으로 변환하고, 컬럼명을 지정
            df = pd.DataFrame(ohlcv_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # 타임스탬프 컬럼을 datetime 형식으로 변환 (밀리초 단위)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # 인덱스를 timestamp로 설정
            df.set_index('timestamp', inplace=True)
            return df.copy()  # 복사본 반환
        except Exception as e:
            logger.error(f"DataFrame 변환 에러: {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.warning(f"{symbol} {timeframe}에 대한 데이터가 없습니다.")
        return pd.DataFrame()

@lru_cache(maxsize=32)
def fetch_latest_ohlcv_data(symbol: str, timeframe: str, limit: int = 500, exchange_id: str = 'binance') -> pd.DataFrame:
    """
    지정된 심볼(symbol)과 시간 간격(timeframe)에 대해 최신 OHLCV 데이터를 수집합니다.
    
    Parameters:
        symbol (str): 거래 심볼 (예: 'BTC/USDT').
        timeframe (str): 시간 간격 (예: '1d', '1h').
        limit (int): 가져올 데이터 개수 제한 (기본값 500).
        exchange_id (str): 사용 거래소 ID (기본값 'binance').
    
    Returns:
        pd.DataFrame: 최신 OHLCV 데이터를 포함하는 DataFrame.
    """
    try:
        # ccxt 거래소 인스턴스 생성 및 마켓 로드
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
        exchange.load_markets()
    except Exception as e:
        logger.error(f"Exchange '{exchange_id}' 초기화 에러: {e}", exc_info=True)
        return pd.DataFrame()
    
    try:
        # 최신 데이터 호출 (limit 개수만큼)
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        logger.error(f"{symbol}의 {timeframe} 최신 데이터 수집 에러: {e}", exc_info=True)
        return pd.DataFrame()
    
    if ohlcvs:
        try:
            # 수집된 데이터를 DataFrame으로 변환 및 인덱스 설정
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
    Binance의 경우, 거래량(quoteVolume)을 시총 대용 지표로 사용하여 거래소 내 유효 심볼 중 상위 count개를 선택합니다.
    스테이블 코인은 제외하며, 각 심볼의 최초 데이터 제공일(온보딩 날짜)을 함께 반환합니다.
    
    Parameters:
        exchange_id (str): 사용 거래소 ID (기본 'binance').
        quote_currency (str): 기준 통화 (예: 'USDT').
        count (int): 반환할 심볼 개수 (기본 5).
        pause_sec (float): API 호출 사이의 대기 시간 (기본 1.0초).
    
    Returns:
        list: (symbol, first_available_date) 튜플의 리스트.
    """
    try:
        # ccxt 거래소 인스턴스 생성 및 마켓 정보 로드
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True, 'timeout': 30000})
        logger.debug(f"{exchange_id}: Loading markets for top symbol retrieval...")
        markets = exchange.load_markets()
        logger.debug(f"{exchange_id}: Markets loaded, total symbols: {len(markets)}")
    except Exception as e:
        logger.error(f"{exchange_id}에서 마켓 로드 에러: {e}", exc_info=True)
        return []

    # USDT 마켓에 해당하는 심볼 필터링 (예: 'BTC/USDT')
    usdt_symbols = [symbol for symbol in markets if symbol.endswith('/' + quote_currency)]
    
    # 스테이블 코인 목록: 해당 코인들은 제외
    stable_coins = {
        "USDT", "BUSD", "USDC", "DAI", "TUSD", "PAX", "USDP", "GUSD", "MIM", "UST",
        "USDe",  # Ethena USD (synthetic dollar)
        "FDUSD",  # First Digital USD
        "PYUSD",  # PayPal USD
        "USD0",  # Usual USD
        "FRAX",  # Frax Finance Stablecoin
        "USDY",  # Ondo USD Yield
        "USDD",  # Tron-based stablecoin
        "EURS",  # Stasis Euro stablecoin
        "RLUSD",  # Ripple USD
        "GHO",  # Aave Stablecoin
        "crvUSD",  # Curve Finance Stablecoin
        "LUSD",  # Liquity USD (ETH-backed stablecoin)
        "XAU₮",  # Tether Gold (Gold-backed stablecoin)
        "PAXG",  # Paxos Gold
        "EUROC",  # Circle Euro Coin
    }
    
    # 스테이블 코인 제외: 심볼의 기본 통화가 stable_coins에 속하지 않는 경우만 필터링
    filtered_symbols = [symbol for symbol in usdt_symbols if symbol.split('/')[0] not in stable_coins]
    logger.debug(f"Filtered symbols (excluding stablecoins): {len(filtered_symbols)} out of {len(usdt_symbols)}")

    # 거래소가 Binance인 경우, 거래량을 기준으로 정렬하여 상위 심볼 선택
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
        # 거래량(quoteVolume) 기준 내림차순 정렬
        symbol_volumes.sort(key=lambda x: x[1], reverse=True)
        sorted_symbols = [item[0] for item in symbol_volumes]
    else:
        # Binance가 아닌 경우, marketCap 정보를 사용하여 정렬
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
    # 각 심볼에 대해 첫 데이터 제공일(온보딩 날짜)을 확인
    for symbol in sorted_symbols:
        # 시작 날짜를 "2018-01-01"로 고정하여 데이터 호출
        df = fetch_historical_ohlcv_data(symbol, '1d', "2018-01-01", 
                                         limit_per_request=1, pause_sec=pause_sec, 
                                         exchange_id=exchange_id, single_fetch=True,
                                         exchange_instance=exchange)
        if df.empty:
            # 데이터가 없으면 해당 심볼 건너뜀
            continue
        first_timestamp = df.index.min()
        first_date_str = first_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        valid_symbols.append((symbol, first_date_str))
        if len(valid_symbols) >= count:
            # 원하는 개수만큼 수집되면 종료
            break

    if len(valid_symbols) < count:
        logger.warning(f"경고: 유효한 데이터가 있는 심볼이 {len(valid_symbols)}개 밖에 없습니다.")
    return valid_symbols

def get_latest_onboard_date(symbols: list, exchange_id: str = 'binance') -> str:
    """
    전달된 심볼 리스트(튜플 또는 단순 심볼 문자열)에 대해 각 심볼의 최초 데이터 제공일(온보딩 날짜)을 구하여,
    가장 늦은(최신) 날짜를 반환합니다.
    
    Parameters:
        symbols (list): 심볼 문자열 또는 (symbol, first_available_date) 튜플 리스트.
        exchange_id (str): 사용 거래소 ID (기본 'binance').
    
    Returns:
        str: 가장 늦은 온보딩 날짜를 "YYYY-MM-DD HH:MM:SS" 형식으로 반환.
    """
    onboard_dates = []
    try:
        for item in symbols:
            if isinstance(item, tuple):
                # 심볼이 튜플인 경우, 두 번째 요소에 첫 데이터 제공일이 포함되어 있음
                onboard_str = item[1]
            else:
                # 단순 심볼인 경우, fetch_historical_ohlcv_data를 호출하여 온보딩 날짜를 확인
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
                # 문자열을 datetime 객체로 변환하여 리스트에 저장
                dt = datetime.strptime(onboard_str, "%Y-%m-%d %H:%M:%S")
                onboard_dates.append(dt)
            except Exception:
                onboard_dates.append(datetime.strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S"))
        if onboard_dates:
            # 가장 최근 날짜를 선택하여 문자열로 반환
            latest_date = max(onboard_dates)
            return latest_date.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error in get_latest_onboard_date: {e}", exc_info=True)
    return "2018-01-01 00:00:00"
