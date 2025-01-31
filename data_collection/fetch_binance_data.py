# data_collection/fetch_binance_data.py
# 바이낸스(Binance) 거래소의 과거/최신 시세 데이터를 수집하기 위해
# CCXT 라이브러리를 사용하는 예시 코드를 담은 파일입니다.

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
    바이낸스에서 과거 가격 데이터를 여러 번 나눠 요청하여 대량으로 수집합니다.
    
    - symbol: 'BTC/USDT' 처럼 거래쌍을 나타냄
    - timeframe: '1m', '4h', '1d' 처럼 원하는 봉 기간(캔들 간격)
    - start_date: 어떤 시점부터 데이터를 불러올지(YYYY-MM-DD HH:MM:SS)
    - limit_per_request: 한 번의 API 호출로 가져올 봉의 최대 개수
    - pause_sec: API 호출 사이에 쉬는 시간(너무 자주 요청하면 제한에 걸릴 수 있음)

    반환값: 날짜와 OHLCV(시가, 고가, 저가, 종가, 거래량)가 들어 있는 pandas DataFrame
    """

    # ccxt 라이브러리를 통해 바이낸스 API 사용 준비
    exchange = ccxt.binance()

    # '2021-01-01 00:00:00' 문자열을 바이낸스에서 요구하는 밀리초 단위 타임스탬프로 변환
    since_ms = exchange.parse8601(start_date)

    all_ohlcv = []  # 과거 데이터를 모두 모으기 위한 리스트

    while True:
        # fetch_ohlcv: 시세 데이터를 (timestamp, open, high, low, close, volume) 순으로 반환
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=limit_per_request
        )

        # 만약 새로 가져온 데이터가 없다면(빈 리스트), 더 이상 받아올 게 없으므로 멈춤
        if not ohlcv:
            break

        # 받아온 데이터를 큰 리스트에 이어붙임
        all_ohlcv += ohlcv

        # 마지막 데이터의 타임스탬프를 기준으로 다음 요청 시작 지점을 갱신
        last_ts = ohlcv[-1][0]
        since_ms = last_ts + 1  # 1ms 뒤부터 다시 요청

        # API 사용량 제한을 피하기 위해 잠시 쉬어줌
        time.sleep(pause_sec)

        # 만약 이번에 가져온 데이터 개수가 최대치보다 작으면, 추가로 가져올 데이터가 없다고 가정하고 종료
        if len(ohlcv) < limit_per_request:
            break

    # 여기까지 모은 시세 데이터를 DataFrame으로 변환
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # timestamp(ms 단위)를 날짜/시간 형식으로 변환 후 인덱스로 설정
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 숫자값으로 바꿀 컬럼들을 float 타입으로 변환
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # 혹시 중복된 인덱스가 있으면 제거, NaN 값도 제거
    df = df[~df.index.duplicated()]
    df.dropna(inplace=True)

    return df


def fetch_binance_latest_ohlcv(symbol: str, timeframe: str = '4h', limit: int = 500) -> pd.DataFrame:
    """
    바이낸스에서 '가장 최근' 시세 데이터 여러 개를 가져오는 함수입니다.
    - symbol: 거래쌍 (예: "BTC/USDT")
    - timeframe: 캔들 간격
    - limit: 몇 개의 봉(캔들)을 가져올지

    반환값: OHLCV 데이터가 담긴 pandas DataFrame
    """

    # 바이낸스 API에 연결할 ccxt 객체 생성
    exchange = ccxt.binance()

    # 최신 데이터 요청
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # 가져온 데이터를 DataFrame으로 변환
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # timestamp를 사람이 읽을 수 있는 시간 형식으로 바꿈 + 인덱스로 설정
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 숫자 컬럼들을 float로 변환
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # 인덱스 중복 제거, NaN 제거
    df = df[~df.index.duplicated()]
    df.dropna(inplace=True)

    return df
