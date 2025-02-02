# init_db.py

import sys
from data_collection.ohlcv_pipeline import collect_and_store_ohlcv_data

def main():
    # 예시: 바이낸스에서 가장 시가총액이 큰 코인 3개
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    # 사용하고자 하는 타임프레임 리스트
    timeframes = ["1d", "4h", "1h", "15m"]
    
    # 2016년부터 최신까지 데이터를 수집 (실제로는 거래소가 지원하는 가장 과거 데이터부터)
    start_date = "2016-01-01 00:00:00"
    
    # 과거 데이터를 모두 받아와서 PostgreSQL에 저장
    collect_and_store_ohlcv_data(
        symbols=symbols,
        timeframes=timeframes,
        use_historical=True,
        start_date=start_date,
        limit_per_request=1000,
        latest_limit=500,  # 최신 데이터 수집 시에만 사용됨
        pause_sec=1.0
    )

    print("\n[완료] 초기 DB 세팅이 끝났습니다.")
    print("지정한 코인들(BTC/USDT, ETH/USDT, BNB/USDT)에 대해")
    print("15m, 1h, 4h, 1d 봉 데이터를 모두 PostgreSQL에 저장했습니다.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[오류 발생] {e}", file=sys.stderr)
        sys.exit(1)
