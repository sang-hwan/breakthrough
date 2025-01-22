# test.py

import pandas as pd  # 데이터 처리를 위한 라이브러리
# 파라미터 조합 테스트 함수 임포트
from backtesting.param_tuning import param_sweep_test

# 메인 함수 정의
def main():
    """
    파라미터 조합 테스트를 실행하고 결과를 CSV로 저장하며,
    ROI 기준 상위 5개의 결과를 출력합니다.
    """

    # -------------------------------
    # 1) 파라미터 조합 테스트 실행
    # -------------------------------
    # param_sweep_test 함수 호출 → 백테스트 실행 및 결과 반환
    results_df = param_sweep_test()

    # 결과가 비어 있으면 처리 중단
    if results_df.empty:
        print("\nNo valid results returned.")  # 유효한 결과 없음 메시지 출력
        return

    # -------------------------------
    # 2) 결과를 CSV 파일로 저장
    # -------------------------------
    csv_filename = "param_sweep_results_with_metrics.csv"  # 저장할 파일 이름
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')  # 파일 저장
    print(f"CSV saved: {csv_filename}")  # 저장 완료 메시지 출력

    # -------------------------------
    # 3) ROI 기준 상위 5개 결과 출력
    # -------------------------------
    # ROI(%) 열을 기준으로 결과를 내림차순 정렬
    sorted_df = results_df.sort_values(by='ROI(%)', ascending=False)
    
    # 정렬된 데이터프레임의 상위 5개 출력
    print("\n=== Top 5 by ROI ===")
    print(sorted_df.head(5))

# 이 스크립트가 직접 실행될 때만 main() 함수를 호출
if __name__ == "__main__":
    main()
