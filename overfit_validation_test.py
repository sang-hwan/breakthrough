# overfit_validation_test.py

import os
import pandas as pd
from backtesting.overfit_validation import walk_forward_analysis

def main():
    # 결과 저장 디렉토리 생성
    output_dir = "walk_forward_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 워크포워드 분석 실행
    results = walk_forward_analysis(
        symbol="BTC/USDT",
        short_timeframe="4h",
        long_timeframe="1d",
        overall_start="2018-01-01 00:00:00",
        overall_end="2025-01-25 23:59:59",
        n_splits=3,          # 예: 3분할
        account_size=10_000,
        train_ratio=0.5      # 각 구간의 절반은 '훈련', 절반은 '테스트'
    )

    # 스플릿별 요약 정보를 저장하기 위한 리스트
    summary_data = []

    for split_res in results:
        idx = split_res['split_index']
        best_params = split_res['best_params'] if split_res['best_params'] else {}
        train_start, train_end = split_res['train_start'], split_res['train_end']
        test_start, test_end = split_res['test_start'], split_res['test_end']

        # 훈련 결과 (DataFrame)
        train_df = split_res['train_results']
        # 테스트 트레이드 (DataFrame)
        test_trades = split_res['test_trades']

        # 1) 스플릿별 훈련 결과를 CSV로 저장
        if train_df is not None and not train_df.empty:
            train_df.to_csv(os.path.join(output_dir, f"train_result_split{idx}.csv"), index=False)

        # 2) 스플릿별 테스트 트레이드를 CSV로 저장
        if test_trades is not None and not test_trades.empty:
            test_trades.to_csv(os.path.join(output_dir, f"test_trades_split{idx}.csv"), index=False)

        # 3) 스플릿별 요약(최적 파라미터 등) 정보를 리스트에 담음
        summary_data.append({
            "split_index": idx,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "best_window": best_params.get("window", None),
            "best_atr_multiplier": best_params.get("atr_multiplier", None),
            "best_profit_ratio": best_params.get("profit_ratio", None),
            "use_partial_tp": best_params.get("use_partial_tp", None)
        })

    # 4) 스플릿별 요약 정보 전체를 하나의 CSV로 저장
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "walk_forward_summary.csv"), index=False)

if __name__ == "__main__":
    main()