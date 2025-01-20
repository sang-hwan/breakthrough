# fetch_binance_data.py

import ccxt  # 다양한 암호화폐 거래소와의 연동을 위한 라이브러리
import pandas as pd  # 데이터 분석용 라이브러리

def fetch_binance_ohlcv(symbol: str, timeframe='4h', limit=500) -> pd.DataFrame:
    """
    Binance에서 OHLCV(시가, 고가, 저가, 종가, 거래량) 데이터를 수집하고,
    이를 pandas DataFrame으로 반환합니다.
    
    :param symbol: 암호화폐 심볼(예: 'BTC/USDT')
    :param timeframe: 바이낸스가 지원하는 데이터 간격(예: '1m', '5m', '15m', '1h', '4h', '1d' 등)
    :param limit: 한 번에 가져올 캔들의 개수 (바이낸스 기준, 대개 1000개까지 가능)
    :return: 인덱스가 timestamp(날짜/시간)이고,
             컬럼으로 open, high, low, close, volume이 있는 pandas DataFrame
    """
    
    # 1) ccxt를 이용해 binance 거래소 객체 생성
    exchange = ccxt.binance()
    
    # 2) binance 거래소에서 OHLCV 데이터를 가져옵니다.
    #    fetch_ohlcv의 반환형: [[timestamp, open, high, low, close, volume], ...] 형태
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    # 3) 가져온 OHLCV 리스트를 pandas DataFrame으로 변환합니다.
    #    columns 인자로 컬럼 이름을 지정해서 보기 좋게 만듭니다.
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 4) timestamp가 밀리초(ms) 단위로 되어 있으므로, 이를 datetime(YYYY-MM-DD hh:mm:ss) 형태로 변환합니다.
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 5) timestamp 컬럼을 데이터프레임의 인덱스로 설정합니다.
    df.set_index('timestamp', inplace=True)
    
    # 6) open, high, low, close, volume 컬럼을 float 형태로 변환하여, 숫자로 연산 가능하게 만듭니다.
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    # 7) 혹시 중복된 인덱스(같은 timestamp 데이터)가 있다면 제거해줍니다.
    if df.index.duplicated().sum() > 0:
        df = df[~df.index.duplicated()]
    
    # 8) 데이터 중 NaN(결측치)이 있다면 해당 행을 제거하여, 분석에 문제가 없도록 합니다.
    if df.isna().any().any():
        df.dropna(inplace=True)
    
    # 9) 최종적으로 정리된 DataFrame을 반환합니다.
    return df
