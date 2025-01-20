# main.py

import pandas as pd

# 1) 데이터 수집 함수 (ccxt 사용)
from fetch_binance_data import fetch_binance_ohlcv

# 2) 시그널 계산 함수들
from signal_generator import (
    calculate_breakout_signals,
    calculate_vcp_pattern
)

# 3) 매매 로직 (손절/익절 가격 계산 등)
from trade_logic import generate_trade_signals

# 4) 백테스트 함수
from backtest import backtest_breakout_strategy

def main():
    # ------------------------------------------------------------
    # 1) 실제 바이낸스 과거 데이터 불러오기
    # ------------------------------------------------------------
    # symbol, timeframe, limit 등은 원하는 대로 설정 가능
    symbol = "BTC/USDT"
    timeframe = "1d"
    limit = 1000  # 예: 최근 500개 4h 봉 => 약 80일 정도
    
    df = fetch_binance_ohlcv(symbol, timeframe, limit)
    # df에는 ['open', 'high', 'low', 'close', 'volume'] 컬럼과
    # 시계열 인덱스(timestamp)가 설정되어 있음.

    # ------------------------------------------------------------
    # 2) 시그널 계산
    # ------------------------------------------------------------
    # (a) 단순 돌파 시그널 + 거래량 조건
    #     - window=20, vol_factor=1.5, confirm_bars=2 등 파라미터 조정
    df = calculate_breakout_signals(
        df,
        window=20,
        vol_factor=1.5,
        confirm_bars=2,
        use_high=False,       # 종가 기준 돌파판단
        breakout_buffer=0.0
    )

    # (b) VCP 패턴(단순 예시)
    df = calculate_vcp_pattern(
        df,
        window_list=[20, 10, 5]
    )

    # ------------------------------------------------------------
    # 3) 매매 로직(손절/익절) 적용
    # ------------------------------------------------------------
    # ATR 손절: (기본값 14, x2 배)
    # 고정 익절: 5% (0.05)
    df = generate_trade_signals(
        df,
        atr_window=14,
        atr_multiplier=2.0,
        profit_ratio=0.05
    )
    # => df에 long_entry, stop_loss_price, take_profit_price, position, ...

    # ------------------------------------------------------------
    # 4) 백테스트 실행
    # ------------------------------------------------------------
    bt_df, trades, summary = backtest_breakout_strategy(
        df,
        initial_capital=10000.0,
        risk_per_trade=1.0,      # 자본 100%씩 진입
        fee_rate=0.0004          # 0.04% 수수료 가정
    )

    # ------------------------------------------------------------
    # 5) 결과 출력
    # ------------------------------------------------------------
    print(bt_df[['close','breakout_signal','confirmed_breakout','long_entry','position','pnl','cum_pnl']].tail(30))
    print("\nTrade History:")
    for t in trades:
        print(t)
    print("\nSummary:", summary)

if __name__ == "__main__":
    main()
