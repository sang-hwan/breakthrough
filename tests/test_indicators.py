# tests/test_indicators.py
import pytest
import pandas as pd
import numpy as np
from trading.indicators import compute_sma, compute_macd, compute_rsi, compute_bollinger_bands

@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "close": np.linspace(100, 130, 30),
        "high": np.linspace(101, 131, 30),
        "low": np.linspace(99, 129, 30),
        "open": np.linspace(100, 130, 30),
        "volume": np.random.randint(1000, 5000, 30)
    }, index=dates)
    return df

def test_compute_sma(sample_data):
    df = compute_sma(sample_data.copy(), period=5, output_col="sma_test")
    assert "sma_test" in df.columns

def test_compute_macd(sample_data):
    df = compute_macd(sample_data.copy(), slow_period=26, fast_period=12, signal_period=9, prefix="macd_")
    for col in ["macd_macd", "macd_signal", "macd_diff"]:
        assert col in df.columns

def test_compute_rsi(sample_data):
    df = compute_rsi(sample_data.copy(), period=14, output_col="rsi_test")
    assert "rsi_test" in df.columns

def test_compute_bollinger_bands(sample_data):
    df = compute_bollinger_bands(sample_data.copy(), period=20, std_multiplier=2.0, prefix="bb_")
    for col in ["bb_mavg", "bb_hband", "bb_lband", "bb_pband", "bb_wband", "bb_hband_ind", "bb_lband_ind"]:
        assert col in df.columns
