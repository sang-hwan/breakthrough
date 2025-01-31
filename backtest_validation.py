# backtest_validation.py

from backtesting.backtest_advanced import run_advanced_backtest


def main():
    # 백테스트 파라미터 예시 (필요시 수정)
    symbol = "BTC/USDT"
    short_tf = "4h"
    long_tf = "1d"

    # 실제로 DB에 해당 구간 데이터가 있어야 매매가 발생할 수 있습니다.
    start_date = "2018-01-01"
    end_date   = "2025-01-21"

    # 백테스트 실행
    trades_df = run_advanced_backtest(
        symbol=symbol,
        short_timeframe=short_tf,
        long_timeframe=long_tf,
        window=20,               # 돌파 확인 시 사용되는 window
        volume_factor=1.5,
        confirm_bars=2,
        breakout_buffer=0.0,
        atr_window=14,
        atr_multiplier=2.0,
        profit_ratio=0.05,
        account_size=10_000.0,
        risk_per_trade=0.01,
        fee_rate=0.001,
        slippage_rate=0.0005,
        taker_fee_rate=0.001,
        total_splits=3,
        threshold_percent=0.02,
        use_trailing_stop=False,
        trailing_percent=0.05,
        use_trend_exit=False,
        start_date=start_date,
        end_date=end_date,
        use_partial_take_profit=False,    # 부분 익절 사용 여부
        partial_tp_factor=0.03,
        final_tp_factor=0.06
    )

    # 결과 처리
    if trades_df is not None and not trades_df.empty:
        # 결과 DataFrame을 CSV로 저장
        trades_df.to_csv("backtest_validation.csv", index=False)
    else:
        print("매매가 발생하지 않았거나 백테스트 결과가 없습니다.")


if __name__ == "__main__":
    main()
