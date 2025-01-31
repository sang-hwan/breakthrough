# param_sweep_test.py

from backtesting.param_sweep import run_param_sweep_advanced

def main():
    # (1) 파라미터 스윕 실행
    result_df = run_param_sweep_advanced(
        symbol="BTC/USDT",
        short_timeframe="4h",
        long_timeframe="1d",
        account_size=10000.0,
        window_list=[10, 20],
        atr_multiplier_list=[1.5, 2.0],
        profit_ratio_list=[0.03, 0.05],
        use_partial_tp_list=[False, True],
        partial_tp_factor=0.02,
        final_tp_factor=0.05,
        start_date="2018-01-01",
        end_date="2025-01-25"
    )

    # (2) CSV 파일로 저장 (현재 폴더에 "param_sweep_result.csv" 생성)
    result_df.to_csv("param_sweep_result.csv", index=False)

if __name__ == "__main__":
    main()