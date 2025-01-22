# data_collection/fetch_binance_data.py
# 이 파일은 바이낸스(Binance) 거래소에서 시세 데이터를 수집하는 데 사용됩니다.

import ccxt  # 다양한 거래소 API를 쉽게 사용할 수 있게 해주는 라이브러리
import pandas as pd  # 데이터 분석을 위한 라이브러리
import time  # 시간 관련 작업을 처리하기 위한 모듈
from datetime import datetime  # 날짜 및 시간을 처리하기 위한 모듈


def fetch_binance_historical_ohlcv(
    symbol: str,
    timeframe: str = '4h',
    start_date: str = '2021-01-01 00:00:00',
    limit_per_request: int = 1000,
    pause_sec: float = 1.0
) -> pd.DataFrame:
    """
    바이낸스에서 과거 데이터를 대량으로 수집하는 함수입니다.

    주요 기능:
    - 특정 시작 날짜부터 현재까지 데이터를 가져옵니다.
    - 데이터를 한 번에 'limit_per_request'만큼 가져오며,
      반복 호출로 더 많은 데이터를 수집합니다.
    - API 호출 제한을 피하기 위해 요청 사이에 대기 시간을 추가합니다.

    매개변수:
    - symbol: str
        거래 쌍을 나타냅니다. 예: 'BTC/USDT'
    - timeframe: str
        데이터 간격을 설정합니다. 예: '1m'(1분), '4h'(4시간), '1d'(1일)
    - start_date: str
        데이터 수집을 시작할 시점입니다. 예: '2021-01-01 00:00:00'
    - limit_per_request: int
        한 번 호출 시 가져올 데이터 개수입니다. 기본값은 1000입니다.
    - pause_sec: float
        연속 호출 사이에 대기할 시간(초)입니다. 기본값은 1.0초입니다.

    반환값:
    - pd.DataFrame
        날짜와 시세 데이터(open, high, low, close, volume)가 포함된 데이터프레임
    """
    # 바이낸스 거래소 객체 생성
    exchange = ccxt.binance()

    # 시작 날짜를 타임스탬프(ms 단위)로 변환
    since_ms = exchange.parse8601(start_date)

    # 수집한 데이터를 저장할 빈 리스트 생성
    all_ohlcv = []

    # 데이터 수집 반복문
    while True:
        # 바이낸스 API를 사용해 OHLCV(시세 데이터) 수집
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=limit_per_request
        )

        # 불러온 데이터가 없으면 종료
        if not ohlcv:
            break

        # 가져온 데이터를 리스트에 추가
        all_ohlcv += ohlcv

        # 가장 최근 데이터의 타임스탬프를 가져와 다음 요청의 시작점으로 설정
        last_ts = ohlcv[-1][0]
        since_ms = last_ts + 1

        # API 호출 제한을 피하기 위해 대기
        time.sleep(pause_sec)

        # 마지막으로 가져온 데이터 개수가 한 번 호출의 최대 개수보다 적으면 종료
        if len(ohlcv) < limit_per_request:
            break

    # 수집된 데이터를 데이터프레임으로 변환
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # 타임스탬프를 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 타임스탬프를 데이터프레임의 인덱스로 설정
    df.set_index('timestamp', inplace=True)

    # 시세 데이터 컬럼들을 실수(float)형으로 변환
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # 중복된 인덱스와 결측치 제거
    df = df[~df.index.duplicated()]
    df.dropna(inplace=True)

    # 최종 데이터프레임 반환
    return df


def fetch_binance_latest_ohlcv(symbol: str, timeframe: str = '4h', limit: int = 500) -> pd.DataFrame:
    """
    바이낸스에서 가장 최근 시세 데이터를 가져오는 함수입니다.

    주요 기능:
    - 지정된 개수의 최신 데이터를 가져옵니다.
    - 실시간 분석이나 단기 데이터가 필요한 경우 유용합니다.

    매개변수:
    - symbol: str
        거래 쌍을 나타냅니다. 예: 'BTC/USDT'
    - timeframe: str
        데이터 간격을 설정합니다. 예: '1m'(1분), '4h'(4시간), '1d'(1일)
    - limit: int
        가져올 데이터의 최대 개수입니다. 기본값은 500입니다.

    반환값:
    - pd.DataFrame
        날짜와 시세 데이터(open, high, low, close, volume)가 포함된 데이터프레임
    """
    # 바이낸스 거래소 객체 생성
    exchange = ccxt.binance()

    # 최신 데이터를 요청
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # 데이터를 데이터프레임으로 변환
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # 타임스탬프를 datetime 형식으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 타임스탬프를 데이터프레임의 인덱스로 설정
    df.set_index('timestamp', inplace=True)

    # 시세 데이터 컬럼들을 실수(float)형으로 변환
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)

    # 중복된 인덱스와 결측치 제거
    df = df[~df.index.duplicated()]
    df.dropna(inplace=True)

    # 최종 데이터프레임 반환
    return df
