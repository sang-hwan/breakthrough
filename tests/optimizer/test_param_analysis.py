# tests/optimizer/test_param_analysis.py
# 이 파일은 run_sensitivity_analysis 함수를 테스트합니다.
# 파라미터 범위를 설정하여 민감도 분석을 실행하고, 결과가 dict 타입으로 반환되는지 확인합니다.

import numpy as np
from strategies.param_analysis import run_sensitivity_analysis

def test_sensitivity_analysis():
    """
    민감도 분석 함수(run_sensitivity_analysis) 테스트

    목적:
      - profit_ratio와 atr_multiplier에 대해 범위 값을 설정한 후,
        민감도 분석을 실행하여 결과가 dict 형식으로 반환되는지 검증.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문으로 반환된 결과 타입 검증)
    """
    # profit_ratio와 atr_multiplier 파라미터에 대해 3개씩 선형 공간의 값 생성
    param_settings = {
        "profit_ratio": np.linspace(0.07, 0.09, 3),
        "atr_multiplier": np.linspace(2.0, 2.2, 3)
    }
    # run_sensitivity_analysis 함수 호출: 지정한 자산, 타임프레임, 날짜 범위, 기간 정보 사용
    results = run_sensitivity_analysis(
        param_settings,
        assets=["BTC/USDT"],
        short_tf="4h",
        long_tf="1d",
        start_date="2023-01-01",
        end_date="2023-01-10",
        periods=[("2023-01-01", "2023-01-10")]
    )
    # 반환 결과가 dict 타입인지 검증
    assert isinstance(results, dict)
