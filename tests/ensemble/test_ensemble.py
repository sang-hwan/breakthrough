# tests/ensemble/test_ensemble.py
# 이 파일은 Ensemble 모듈의 get_final_signal 메서드 기능을 테스트하기 위한 테스트 케이스들을 포함합니다.
# Ensemble 객체는 여러 트레이딩 시그널을 조합하여 최종 트레이딩 신호를 결정하는 역할을 합니다.

import pytest
import pandas as pd
from signal_calculation.ensemble import Ensemble

# 전역 객체 및 데이터셋 준비: 최소한의 컬럼을 가진 테스트용 데이터프레임 생성
@pytest.fixture
def dummy_data():
    """
    Dummy 데이터프레임 생성 함수

    목적:
      - 테스트를 위해 시가(open), 종가(close), 최고가(high), 최저가(low),
        단순 이동 평균(sma), RSI, 볼린저 밴드 하단(bb_lband) 등 기본 컬럼을 갖는 데이터프레임 생성.
    
    Parameters:
      없음

    Returns:
      pd.DataFrame: 시간 인덱스를 가진 10행의 간단한 가격 데이터
    """
    # 2023-01-01부터 시작하여 10시간 간격의 타임스탬프 생성
    dates = pd.date_range("2023-01-01", periods=10, freq="h")
    df = pd.DataFrame({
        "open": [100] * 10,       # 시가: 모든 값이 100
        "close": [101] * 10,      # 종가: 모든 값이 101
        "high": [102] * 10,       # 최고가: 모든 값이 102
        "low": [99] * 10,         # 최저가: 모든 값이 99
        "sma": [100.5] * 10,      # 단순 이동 평균: 모든 값이 100.5
        "rsi": [30] * 10,         # RSI: 모든 값이 30 (과매도 영역을 의미할 수 있음)
        "bb_lband": [99] * 10     # 볼린저 밴드 하단: 모든 값이 99
    }, index=dates)
    return df

# 주간 데이터 준비: 주간 단위의 추가 지표를 포함하는 데이터프레임 생성
@pytest.fixture
def dummy_weekly_data():
    """
    Dummy 주간 데이터프레임 생성 함수

    목적:
      - 주간 데이터 테스트를 위해 종가(close), 최고가(high), 최저가(low),
        그리고 주간 모멘텀(weekly_momentum) 컬럼을 가진 데이터프레임 생성.
    
    Parameters:
      없음

    Returns:
      pd.DataFrame: 주 단위의 데이터(2행)를 포함하는 데이터프레임
    """
    # 2023-01-01부터 시작하여 주간(월요일) 단위의 타임스탬프 생성 (총 2주)
    dates = pd.date_range("2023-01-01", periods=2, freq="W-MON")
    df = pd.DataFrame({
        "close": [101, 103],
        "high": [102, 104],
        "low": [99, 100],
        "weekly_momentum": [0.6, 0.6]  # 주간 모멘텀 지표 (예시 값)
    }, index=dates)
    return df

def test_get_final_signal(dummy_data, dummy_weekly_data):
    """
    Ensemble 모듈의 get_final_signal 메서드 동작 테스트

    목적:
      - 주어진 시장 상황(market_regime)과 유동성 정보(liquidity_info), 
        그리고 테스트 데이터를 이용해 최종 트레이딩 신호가 올바르게 반환되는지 검증.

    Parameters:
      dummy_data (pd.DataFrame): 시세 관련 데이터 (시간 단위)
      dummy_weekly_data (pd.DataFrame): 주간 단위의 추가 데이터

    Returns:
      없음 (assert 구문을 통해 테스트 통과 여부 확인)
    """
    # Ensemble 인스턴스 생성
    ens = Ensemble()
    # 현재 시간을 dummy_data의 마지막 인덱스로 설정
    current_time = dummy_data.index[-1]
    # get_final_signal 메서드 호출: 'bullish' 시장, 'high' 유동성을 가정
    final_signal = ens.get_final_signal(
        market_regime="bullish", 
        liquidity_info="high", 
        data=dummy_data, 
        current_time=current_time, 
        data_weekly=dummy_weekly_data
    )
    # 최종 신호가 "enter_long", "exit_all", "hold" 중 하나임을 검증
    assert final_signal in ["enter_long", "exit_all", "hold"]
