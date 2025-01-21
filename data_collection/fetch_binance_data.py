# data_collection/fetch_binance_data.py

import ccxt
import pandas as pd
import time
from datetime import datetime


def fetch_binance_historical_ohlcv(
    symbol: str,
    timeframe: str = '4h',
    start_date: str = '2021-01-01 00:00:00',
    limit_per_request: int = 1000,
    pause_sec: float = 1.0
) -> pd.DataFrame:
    """
    1) 바이낸스에서 과거 데이터를 대량으로 수집하는 함수입니다.
    - start_date(시작 시점)부터 현재 시점까지,
      'limit_per_request'만큼씩 데이터를 끊어서 반복적으로 불러옵니다.
    - 각 반복 호출 사이에는 'pause_sec'만큼 잠깐 쉬어,
      API 호출 제한에 걸리지 않도록 합니다.

    Parameters
    ----------
    symbol : str
        예) 'BTC/USDT'. (거래쌍: 어떤 코인/토큰을 어떤 기준화폐로 거래하는지 표시)
    timeframe : str
        예) '1m', '5m', '15m', '1h', '4h', '1d' 등 (봉 하나가 몇 분, 시간, 일 단위인지)
    start_date : str
        데이터 수집을 시작할 시점 (UTC 기준 문자열)
        예) '2021-01-01 00:00:00'
    limit_per_request : int
        한 번 API를 호출할 때 가져올 최대 캔들 수(기본값 1000, 바이낸스에서 보통 1000까지 허용)
    pause_sec : float
        연속된 API 호출 사이에 쉬는 시간(초). (기본값 1초)

    Returns
    -------
    pd.DataFrame
        * 인덱스: timestamp (datetime 형태)
        * 컬럼: open, high, low, close, volume
    """

    # 바이낸스 거래소에 연결할 수 있는 'ccxt' 객체를 생성합니다.
    exchange = ccxt.binance()

    # 1) start_date(문자열)를 바이낸스 API에서 요구하는 timestamp(밀리초)로 변환
    since_ms = exchange.parse8601(start_date)

    # 데이터를 담을 빈 리스트 생성
    all_ohlcv = []

    # 무한 반복문: 더 이상 받아올 데이터가 없을 때까지 계속 API를 호출
    while True:
        # 2) fetch_ohlcv: 바이낸스 서버에서 OHLCV(시세데이터) 가져오기
        #    - since=since_ms 부터
        #    - 최대 limit_per_request개
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=limit_per_request
        )

        # 불러온 데이터가 없다면, (더 이상 가져올 것이 없으므로) 반복 중단
        if not ohlcv:
            break

        # 3) 새로 불러온 데이터를 all_ohlcv 리스트에 추가(누적)
        all_ohlcv += ohlcv

        # 4) 새로 불러온 데이터 중 마지막 캔들의 타임스탬프를 구함
        #    ohlcv는 [[ts, open, high, low, close, volume], [...], ...] 형태로 되어있음
        last_ts = ohlcv[-1][0]  # ms 단위 정수값
        # 다음 호출에서는 마지막 캔들의 시각 + 1ms 부터 요청 (겹치지 않게 하기 위함)
        since_ms = last_ts + 1

        # 5) 잠깐 휴식 -> API 서버 호출 제한(rate limit)에 걸리지 않도록 배려
        time.sleep(pause_sec)

        # 6) 만약 이번 호출에서 limit_per_request(기본 1000)보다 적게 가져왔다면,
        #    더 이상 과거 데이터가 없다고 판단하고 반복 중단
        if len(ohlcv) < limit_per_request:
            break

    # 7) 누적한 리스트(all_ohlcv)를 판다스 DataFrame으로 변환
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # timestamp(숫자:ms)을 판다스에서 인식 가능한 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # DataFrame 인덱스를 timestamp 컬럼으로 설정
    df.set_index('timestamp', inplace=True)

    # 시가/고가/저가/종가/거래량을 모두 float(실수)형으로 변환
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # 인덱스(날짜)가 중복된 행이 있다면 제거
    df = df[~df.index.duplicated()]

    # 결측치(NaN) 행이 있다면 제거
    df.dropna(inplace=True)

    # 최종적으로 가공이 완료된 DataFrame 반환
    return df


def fetch_binance_latest_ohlcv(symbol: str, timeframe: str = '4h', limit: int = 500) -> pd.DataFrame:
    """
    2) 바이낸스에서 '최신' 데이터만 빠르게 수집하는 메서드입니다.
    - 가장 최근(limit개)의 OHLCV 정보를 가져와서
      실시간 모니터링 혹은 단기 분석에 활용할 때 사용합니다.

    Parameters
    ----------
    symbol : str
        예) 'BTC/USDT'
    timeframe : str
        예) '1m', '5m', '15m', '1h', '4h', '1d' 등
    limit : int
        불러올 캔들의 개수 (기본값 500)

    Returns
    -------
    pd.DataFrame
        * 인덱스: timestamp (datetime 형태)
        * 컬럼: open, high, low, close, volume
    """

    # 바이낸스 거래소에 연결할 수 있는 'ccxt' 객체 생성
    exchange = ccxt.binance()

    # 이번에는 'since' 없이, 마지막 limit개의 데이터만 바로 불러옴
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # 가져온 2차원 리스트를 DataFrame으로 변환
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # timestamp를 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # DataFrame 인덱스를 timestamp로 설정
    df.set_index('timestamp', inplace=True)

    # 숫자형 컬럼을 float으로 변환
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # 인덱스 중복 제거
    df = df[~df.index.duplicated()]

    # 결측치 제거
    df.dropna(inplace=True)

    # 최신 데이터가 들어있는 DataFrame 반환
    return df

