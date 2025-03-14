# tests/backtesting/test_ohlcv_variability.py

import pandas as pd
import numpy as np
import pytest
# OHLCV 데이터의 유효성을 검증하는 내부 함수 임포트
from backtesting.steps.data_loader import _validate_and_prepare_df
# 로거 설정 함수 임포트 (로그 메시지 확인에 사용)
from logs.logger_config import setup_logger

# 현재 모듈의 로거 객체를 생성합니다.
logger = setup_logger(__name__)

@pytest.fixture
def constant_ohlcv():
    """
    거의 일정한 OHLCV 데이터를 생성하는 fixture (낮은 변동성).
    
    - 10일치 데이터를 생성하며, 가격은 거의 일정하며 미세한 변동만 있음.
    
    Returns:
        pd.DataFrame: 일정한 OHLCV 데이터 (인덱스는 날짜)
    """
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    data = {
        'open': np.full(10, 100.0),
        'high': np.full(10, 100.01),  # 극히 미세한 변동
        'low': np.full(10, 99.99),
        'close': np.full(10, 100.0),
        'volume': np.full(10, 1000)
    }
    df = pd.DataFrame(data, index=dates)
    return df

@pytest.fixture
def volatile_ohlcv():
    """
    충분한 변동성을 가지는 OHLCV 데이터를 생성하는 fixture.
    
    - 재현성을 위해 난수 시드를 고정하고, 가격에 대한 변동성이 충분히 나타나도록 데이터를 생성합니다.
    
    Returns:
        pd.DataFrame: 변동성이 큰 OHLCV 데이터 (인덱스는 날짜)
    """
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    open_prices = np.linspace(100, 110, 10)
    high_prices = open_prices + np.random.uniform(1, 2, 10)
    low_prices = open_prices - np.random.uniform(1, 2, 10)
    close_prices = open_prices + np.random.uniform(-1, 1, 10)
    volume = np.random.randint(1000, 2000, 10)
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    return df

def test_low_volatility_warning(caplog, constant_ohlcv):
    """
    낮은 변동성 데이터를 사용할 때 경고 메시지가 로깅되는지 테스트합니다.
    
    - caplog을 사용하여 로깅된 경고 메시지를 캡처합니다.
    - _validate_and_prepare_df 함수를 호출 후, "low volatility" 경고 메시지가 포함되었는지 검증합니다.
    
    Parameters:
        caplog: pytest의 로그 캡처 fixture
        constant_ohlcv (pd.DataFrame): 낮은 변동성 데이터
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    caplog.set_level("WARNING")
    # 실제 테이블 명칭 예시: "ohlcv_btcusdt_1d"
    table_name = "ohlcv_btcusdt_1d"
    _validate_and_prepare_df(constant_ohlcv, table_name)
    # 로깅된 메시지 중 "low volatility" 문자열이 포함되었는지 확인
    warning_found = any("low volatility" in record.message for record in caplog.records)
    assert warning_found, "낮은 변동성 데이터에 대해 경고가 발생해야 합니다."

def test_high_volatility_no_warning(caplog, volatile_ohlcv):
    """
    충분한 변동성 데이터를 사용할 때 경고 메시지가 발생하지 않는지 테스트합니다.
    
    - caplog을 사용하여 로깅된 경고 메시지를 캡처한 후, "low volatility" 메시지가 없는지 확인합니다.
    
    Parameters:
        caplog: pytest의 로그 캡처 fixture
        volatile_ohlcv (pd.DataFrame): 변동성이 큰 데이터
        
    Returns:
        없음 (assertion을 통해 테스트 결과를 검증)
    """
    caplog.set_level("WARNING")
    table_name = "ohlcv_btcusdt_1d"
    _validate_and_prepare_df(volatile_ohlcv, table_name)
    # 로깅된 메시지 중 "low volatility" 문자열이 없는지 확인
    warning_found = any("low volatility" in record.message for record in caplog.records)
    assert not warning_found, "충분한 변동성 데이터에서는 경고가 발생하면 안 됩니다."
