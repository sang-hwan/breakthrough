# data_collection/ohlcv_data_pipeline.py
# Binance에서 OHLCV 데이터를 수집하여 PostgreSQL에 넣는 과정을 자동화한 파이프라인 코드입니다.

import time
from datetime import datetime
from typing import List, Optional

# 다른 파일에서 필요한 함수를 가져옴
from data_collection.fetch_binance_data import (
    fetch_binance_historical_ohlcv,
    fetch_binance_latest_ohlcv
)
from data_collection.postgres_ohlcv_handler import save_ohlcv_to_postgres


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
    심볼(예: BTC/USDT 등)과 타임프레임(예: 4h, 1d 등)을 지정하면,
    바이낸스에서 OHLCV를 가져와 PostgreSQL에 저장합니다.

    - use_historical=True => 설정된 start_date부터 모든 과거 데이터를 받아옴
    - use_historical=False => 최신 데이터만 제한적으로 받음
    """

    # 선택된 각 심볼을 순회
    for symbol in symbols:
        # 각 타임프레임(봉 간격)도 순회
        for tf in timeframes:
            print(f"\n[*] Fetching {symbol} - {tf} data...")

            # (1) 데이터 수집
            if use_historical:
                # 과거 데이터
                if not start_date:
                    raise ValueError("start_date는 과거 데이터 수집 시 반드시 필요합니다.")
                df = fetch_binance_historical_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    limit_per_request=limit_per_request,
                    pause_sec=pause_sec
                )
            else:
                # 최신 데이터
                df = fetch_binance_latest_ohlcv(
                    symbol=symbol,
                    timeframe=tf,
                    limit=latest_limit
                )

            # DB에 저장할 테이블 이름 예: "ohlcv_btcusdt_4h"
            table_name = f"ohlcv_{symbol.replace('/', '').lower()}_{tf}"
            print(f"    -> Total Rows Fetched: {len(df)}")

            # (2) 수집한 DataFrame을 PostgreSQL에 저장
            save_ohlcv_to_postgres(df, table_name=table_name)
            print(f"    -> Saved to table: {table_name}")

            # (3) 다음 요청 전에 잠깐 대기 (API 부하 방지)
            time.sleep(pause_sec)
