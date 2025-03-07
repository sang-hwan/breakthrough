# tests/backtesting/test_ohlcv_variability.py
import pandas as pd
import numpy as np
import pytest
from backtesting.steps.data_loader import _validate_and_prepare_df
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

@pytest.fixture
def constant_ohlcv():
    """
    거의 일정한 OHLCV 데이터를 생성 (낮은 변동성)
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
    충분한 변동성을 가지는 OHLCV 데이터를 생성.
    재현성을 위해 난수 시드를 고정합니다.
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
    caplog.set_level("WARNING")
    # 실제 테이블 이름 규칙을 반영 (예: "ohlcv_btcusdt_1d")
    table_name = "ohlcv_btcusdt_1d"
    _validate_and_prepare_df(constant_ohlcv, table_name)
    # "low volatility" 경고 메시지가 로그에 기록되었는지 확인합니다.
    warning_found = any("low volatility" in record.message for record in caplog.records)
    assert warning_found, "낮은 변동성 데이터에 대해 경고가 발생해야 합니다."

def test_high_volatility_no_warning(caplog, volatile_ohlcv):
    caplog.set_level("WARNING")
    table_name = "ohlcv_btcusdt_1d"
    _validate_and_prepare_df(volatile_ohlcv, table_name)
    # 충분한 변동성 데이터에서는 경고가 발생하지 않아야 합니다.
    warning_found = any("low volatility" in record.message for record in caplog.records)
    assert not warning_found, "충분한 변동성 데이터에서는 경고가 발생하면 안 됩니다."
