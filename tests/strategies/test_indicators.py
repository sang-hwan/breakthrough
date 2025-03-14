# tests/strategies/test_indicators.py
# 이 모듈은 기술적 지표(예: SMA, MACD, RSI, Bollinger Bands)의 계산 함수들이 올바르게 동작하는지 검증하는 테스트 케이스들을 포함합니다.

import pytest  # pytest 프레임워크를 사용하여 테스트 케이스를 작성합니다.
import pandas as pd  # 데이터프레임 생성 및 조작을 위해 사용합니다.
import numpy as np  # 수치 연산에 활용합니다.
from trading.indicators import compute_sma, compute_macd, compute_rsi, compute_bollinger_bands  # 테스트할 지표 계산 함수들

@pytest.fixture
def sample_data():
    """
    샘플 거래 데이터를 생성하는 fixture입니다.
    
    30일 동안의 날짜 인덱스를 가지며, 'close', 'high', 'low', 'open', 'volume' 컬럼을 포함하는 DataFrame을 생성합니다.
    이 데이터는 다양한 지표 함수의 입력으로 사용됩니다.
    
    Returns:
        pd.DataFrame: 샘플 거래 데이터.
    """
    dates = pd.date_range("2023-01-01", periods=30, freq="D")  # 2023년 1월 1일부터 30일간의 날짜 생성
    df = pd.DataFrame({
        "close": np.linspace(100, 130, 30),  # 100에서 130까지 선형적으로 증가하는 종가
        "high": np.linspace(101, 131, 30),   # 101에서 131까지 선형적으로 증가하는 고가
        "low": np.linspace(99, 129, 30),     # 99에서 129까지 선형적으로 증가하는 저가
        "open": np.linspace(100, 130, 30),   # 100에서 130까지 선형적으로 증가하는 시가
        "volume": np.random.randint(1000, 5000, 30)  # 1000 ~ 5000 사이의 임의의 거래량
    }, index=dates)
    return df

def test_compute_sma(sample_data):
    """
    단순 이동 평균(SMA) 계산 함수를 테스트합니다.
    
    sample_data의 복사본에 대해 period=5로 SMA를 계산하고, 지정한 출력 컬럼("sma_test")이 추가되었는지 검증합니다.
    """
    df = compute_sma(sample_data.copy(), period=5, output_col="sma_test")  # SMA 계산 실행
    assert "sma_test" in df.columns  # SMA 컬럼이 추가되었는지 확인

def test_compute_macd(sample_data):
    """
    MACD (이동평균 수렴발산 지표) 계산 함수를 테스트합니다.
    
    지정된 기간 값으로 MACD, 시그널, 그리고 차이값 컬럼들이 생성되는지 확인합니다.
    """
    df = compute_macd(sample_data.copy(), slow_period=26, fast_period=12, signal_period=9, prefix="macd_")
    # MACD 관련 컬럼들이 모두 존재하는지 반복문으로 검증
    for col in ["macd_macd", "macd_signal", "macd_diff"]:
        assert col in df.columns

def test_compute_rsi(sample_data):
    """
    RSI (상대강도지수) 계산 함수를 테스트합니다.
    
    period=14로 RSI를 계산한 후, 지정한 출력 컬럼("rsi_test")이 DataFrame에 추가되었는지 검증합니다.
    """
    df = compute_rsi(sample_data.copy(), period=14, output_col="rsi_test")
    assert "rsi_test" in df.columns

def test_compute_bollinger_bands(sample_data):
    """
    Bollinger Bands 계산 함수를 테스트합니다.
    
    period=20, 표준편차 배수 2.0을 적용하여 Bollinger Bands 관련 컬럼들이 생성되는지 확인합니다.
    """
    df = compute_bollinger_bands(sample_data.copy(), period=20, std_multiplier=2.0, prefix="bb_")
    # Bollinger Bands 관련 컬럼들이 모두 존재하는지 검증
    for col in ["bb_mavg", "bb_hband", "bb_lband", "bb_pband", "bb_wband", "bb_hband_ind", "bb_lband_ind"]:
        assert col in df.columns
