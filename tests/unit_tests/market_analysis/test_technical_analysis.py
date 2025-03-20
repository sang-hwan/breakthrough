# tests/unit_tests/market_analysis/test_technical_analysis.py
import pytest
import pandas as pd
import numpy as np
from market_analysis.technical_analysis import compute_sma, compute_bollinger_bands, compute_rsi

def test_compute_sma():
    data = pd.Series([1, 2, 3, 4, 5])
    sma = compute_sma(data, window=3)
    # 첫 두 값는 NaN이어야 하므로, 나머지 값에 대해 검증
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sma.dropna().values, expected.dropna().values)

def test_compute_bollinger_bands():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    bb = compute_bollinger_bands(data, window=3, num_std=2)
    # 결과 DataFrame에 필요한 컬럼들이 있는지 확인
    for col in ["SMA", "Upper Band", "Lower Band"]:
        assert col in bb.columns

def test_compute_rsi():
    data = pd.Series(np.linspace(100, 110, 20))
    rsi = compute_rsi(data, window=14)
    # RSI 값이 0~100 범위에 있는지 확인 (NaN은 제외)
    assert rsi.dropna().between(0, 100).all()
