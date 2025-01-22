# test.py

import pandas as pd
from backtesting.param_tuning import param_sweep_test


def main():
    # 함수 호출하여 결과 DataFrame 가져옴
    results_df = param_sweep_test()

    if results_df.empty:
        print("\nNo valid results returned.")
        return

    # CSV로 저장
    csv_filename = "param_sweep_results_with_metrics.csv"
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"CSV saved: {csv_filename}")

    # ROI(%)로 정렬 후 상위 5개 출력
    sorted_df = results_df.sort_values(by='ROI(%)', ascending=False)
    print("\n=== Top 5 by ROI ===")
    print(sorted_df.head(5))


if __name__ == "__main__":
    main()
