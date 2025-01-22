# data_collection/ohlcv_data_pipeline.py

# 시간 관리 및 데이터 타입을 위한 모듈
import time
from datetime import datetime
from typing import List, Optional

# 외부 파일에서 데이터 수집 및 저장 함수 가져오기
from data_collection.fetch_binance_data import (
    fetch_binance_historical_ohlcv,  # 바이낸스에서 역사적 OHLCV 데이터 수집 함수
    fetch_binance_latest_ohlcv       # 바이낸스에서 최신 OHLCV 데이터 수집 함수
)
from data_collection.save_to_postgres import save_ohlcv_to_postgres  # 데이터 저장 함수

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
    Binance에서 OHLCV 데이터를 수집 후 PostgreSQL 데이터베이스에 저장하는 함수입니다.

    Parameters:
        symbols (List[str]): 데이터 수집을 원하는 암호화폐 심볼 리스트. 예: ["BTC/USDT", "ETH/USDT"]
        timeframes (List[str]): 데이터 수집을 원하는 타임프레임 리스트. 예: ["1h", "4h", "1d"]
        use_historical (bool): True면 start_date부터 모든 데이터를 수집, False면 최신 데이터만 수집.
        start_date (str, optional): 역사적 데이터 수집 시작 날짜 (UTC 기준). 기본값: '2018-01-01 00:00:00'
        limit_per_request (int): 한 번의 요청에서 가져올 데이터 수. 기본값: 1000
        latest_limit (int): 최신 데이터 수집 시 가져올 데이터 수. 기본값: 500
        pause_sec (float): 각 요청 사이의 대기 시간(초). 기본값: 1.0

    Returns:
        None
    """
    # (1) 각 암호화폐 심볼에 대해 작업 시작
    for symbol in symbols:
        # (2) 각 타임프레임에 대해 작업 시작
        for tf in timeframes:
            print(f"\n[*] Fetching {symbol} - {tf} data...")

            # 데이터 수집 단계
            if use_historical:
                # (a) 역사적 데이터 수집
                if not start_date:
                    raise ValueError("start_date는 역사적 데이터 수집에 필수입니다.")
                df = fetch_binance_historical_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    limit_per_request=limit_per_request,
                    pause_sec=pause_sec
                )
            else:
                # (b) 최신 데이터 수집
                df = fetch_binance_latest_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    limit=latest_limit
                )

            # 테이블 이름 생성
            table_name = f"ohlcv_{symbol.replace('/', '').lower()}_{tf}"
            print(f"    -> Total Rows Fetched: {len(df)}")

            # 수집한 데이터를 PostgreSQL 데이터베이스에 저장
            save_ohlcv_to_postgres(df, table_name=table_name)
            print(f"    -> Saved to table: {table_name}")

            # 다음 요청 전 잠시 대기 (API 부하 방지)
            time.sleep(pause_sec)
