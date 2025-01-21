# data_collection/ohlcv_data_pipeline.py

import time
from datetime import datetime
from typing import List, Optional

# 아래는 별도 파일(fetch_binance_data.py, save_to_postgres.py)에 정의된 함수를 불러옵니다.
from data_collection.fetch_binance_data import (
    fetch_binance_historical_ohlcv,
    fetch_binance_latest_ohlcv
)
from data_collection.save_to_postgres import save_ohlcv_to_postgres

def collect_data_for_backtest(
    symbols: List[str],
    timeframes: List[str],
    use_historical: bool = True,
    start_date: Optional[str] = '2018-01-01 00:00:00',
    limit_per_request: int = 1000,
    latest_limit: int = 500,
    pause_sec: float = 1.0
) -> None:
    """
    Binance에서 OHLCV 데이터를 수집 후, PostgreSQL에 저장하는 함수.
    
    생계형 투자를 위해 다음을 고려했습니다:
    1) 주요 타임프레임: 1h, 4h, 1d (신뢰도 높은 신호를 얻기 위함)
    2) 부가적으로 15m를 추가 가능 (세밀한 돌파 시점 확인), 
       그러나 지나치게 짧은 분봉은 과매매로 이어질 수 있으므로 주의
    3) start_date를 5년 전(2018년) 정도로 설정하여 충분히 긴 기간(상승장/하락장 모두 포함) 확보
    
    Parameters
    ----------
    symbols : List[str]
        수집할 암호화폐 심볼 목록 (예: ["BTC/USDT", "ETH/USDT"])
    timeframes : List[str]
        원하는 타임프레임 목록 (예: ["1h", "4h", "1d", "15m"] 등)
    use_historical : bool
        True 면 start_date부터 현재까지 역사적 데이터를 반복 호출로 수집 (기본값: True).
        False 면 최근 latest_limit개의 데이터만 가져옵니다.
    start_date : str, optional
        대량(역사적) 데이터 수집 시작 시점(UTC), 기본값: '2018-01-01 00:00:00' (약 5년치).
        use_historical=False일 경우 사용되지 않음.
    limit_per_request : int
        (역사적 수집 시) fetch_ohlcv 1회당 가져올 최대 캔들 수 (기본 1000).
    latest_limit : int
        (최신 데이터 수집 시) 가져올 캔들의 개수 (기본 500).
    pause_sec : float
        심볼/타임프레임별 호출 후 대기 시간(초). API 과부하 방지.
    """
    
    # (1) 심볼(예: "BTC/USDT", "ETH/USDT") 목록을 하나씩 순회
    for symbol in symbols:
        # (2) 각 심볼에 대해 원하는 타임프레임(예: "1h", "4h", "1d" 등)마다 순회
        for tf in timeframes:
            print(f"\n[*] Fetching {symbol} - {tf} data...")

            # -------------------------------
            # 1) 바이낸스에서 OHLCV 데이터 수집
            # -------------------------------
            if use_historical:
                # 장기간(역사적) 데이터를 한 번에 전부 수집하는 경우
                if not start_date:
                    # 만약 start_date가 지정되지 않았는데 'use_historical=True'라면 오류 처리
                    raise ValueError("start_date must be provided for historical data.")
                
                # fetch_binance_historical_ohlcv 함수를 이용해
                # start_date부터 현재까지 데이터를 모두 불러옴
                df = fetch_binance_historical_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    limit_per_request=limit_per_request,
                    pause_sec=pause_sec
                )
            else:
                # 최근 latest_limit개의 데이터만 가져오고 싶을 때
                df = fetch_binance_latest_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    limit=latest_limit
                )

            # -------------------------------
            # 2) 테이블 이름 정의
            # -------------------------------
            # 예: 심볼 "BTC/USDT" -> "btcusdt", 타임프레임 "4h" -> "4h"
            # 최종 테이블명 "ohlcv_btcusdt_4h"
            table_name = f"ohlcv_{symbol.replace('/', '').lower()}_{tf}"
            print(f"    -> Total Rows Fetched: {len(df)}")
            
            # -------------------------------
            # 3) DB에 저장
            # -------------------------------
            # 위에서 불러온 DataFrame(df)을 지정된 테이블(table_name)에 저장
            save_ohlcv_to_postgres(df, table_name=table_name)
            print(f"    -> Saved to table: {table_name}")

            # -------------------------------
            # 4) 다음 심볼/타임프레임으로 넘어가기 전 대기
            # -------------------------------
            # API 호출을 잇달아 하면 서버 부하 및 제한이 걸릴 수 있으므로 잠시 휴식
            time.sleep(pause_sec)
