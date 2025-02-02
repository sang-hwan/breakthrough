# backtest_validation.py

import os
import json
import pandas as pd
from backtesting.backtest_advanced import run_advanced_backtest

def main():
    # 백테스트 기본 설정
    symbol = "BTC/USDT"
    short_tf = "4h"
    long_tf = "1d"
    start_date = "2018-01-01"
    end_date   = "2025-02-01"
    
    # best_params.json 파일 읽기
    best_params_path = os.path.join(os.path.dirname(__file__), "best_params.json")
    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    
    # best_params.json 에 없는 파라미터는 기본값으로 설정합니다.
    default_trend_exit_lookback = 30
    # trailing_percent는 최적화 결과에 포함되지 않을 수 있으므로 기본값 0.0 사용
    trailing_percent = best_params.get("trailing_percent", 0.0)
    
    # 백테스트 실행 시 최적 파라미터들을 적용합니다.
    result = run_advanced_backtest(
        symbol=symbol,
        short_timeframe=short_tf,
        long_timeframe=long_tf,
        window=best_params["lookback_window"],
        volume_factor=best_params["volume_factor"],
        confirm_bars=best_params["confirmation_bars"],
        breakout_buffer=best_params["breakout_buffer"],
        atr_window=best_params["atr_period"],
        atr_multiplier=best_params["atr_multiplier"],
        profit_ratio=best_params["profit_ratio"],
        account_size=10000.0,
        risk_per_trade=best_params["risk_per_trade"],
        fee_rate=0.001,
        slippage_rate=0.0005,
        taker_fee_rate=0.001,
        total_splits=best_params["total_splits"],
        allocation_mode=best_params["allocation_mode"],
        threshold_percent=best_params["scale_in_threshold"],
        use_trailing_stop=best_params["use_trailing_stop"],
        trailing_percent=trailing_percent,
        use_trend_exit=best_params["use_trend_exit"],
        trend_exit_lookback=best_params.get("trend_exit_lookback", default_trend_exit_lookback),
        trend_exit_price_column="close",
        start_date=start_date,
        end_date=end_date,
        use_partial_take_profit=best_params["use_partial_take_profit"],
        partial_tp_factor=best_params["partial_tp_factor"],
        final_tp_factor=best_params["final_tp_factor"],
        # 장기 추세 필터 관련 파라미터
        sma_period=best_params["sma_period"],
        macd_slow_period=best_params["macd_slow_period"],
        macd_fast_period=best_params["macd_fast_period"],
        macd_signal_period=best_params["macd_signal_period"],
        rsi_period=best_params["rsi_period"],
        rsi_threshold=best_params["rsi_threshold"],
        bb_period=best_params["bb_period"],
        bb_std_multiplier=best_params["bb_std_multiplier"],
        # 리테스트 관련 파라미터
        retest_threshold=best_params["retest_threshold"],
        retest_confirmation_bars=best_params["retest_confirmation_bars"],
        # 신호 옵션 (단기 보조지표 사용 여부)
        use_short_term_indicators=best_params["use_short_term_indicators"],
        short_rsi_threshold=best_params["short_rsi_threshold"],
        short_rsi_period=best_params["short_rsi_period"],
        # 신호 컬럼명 (기본값 사용)
        breakout_flag_col="breakout_signal",
        confirmed_breakout_flag_col="confirmed_breakout",
        retest_signal_col="retest_signal",
        long_entry_col="long_entry"
    )
    
    if result is not None:
        trades_df, trade_logs = result
        
        # 거래 요약 결과를 CSV 파일로 저장합니다.
        trades_df.to_csv("backtest_validation.csv", index=False)
        print("백테스트 결과가 'backtest_validation.csv'로 저장되었습니다.")
        
        # 거래 체결 상세 정보를 CSV 파일로 저장합니다.
        if trade_logs:
            trade_logs_df = pd.DataFrame(trade_logs)
            trade_logs_df.to_csv("trade_details.csv", index=False)
            print("거래 체결 상세 정보가 'trade_details.csv'로 저장되었습니다.")
        else:
            print("거래 체결 상세 정보가 없습니다.")
    else:
        print("매매가 발생하지 않았거나 백테스트 결과가 없습니다.")

if __name__ == "__main__":
    main()
