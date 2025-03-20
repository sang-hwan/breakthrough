# tests/unit_tests/market_analysis/test_onchain_analysis.py
import pytest
import math
from market_analysis.onchain_analysis import calculate_mvrv_ratio, analyze_exchange_flow

def test_calculate_mvrv_ratio_normal():
    ratio = calculate_mvrv_ratio(200e9, 150e9)
    expected = 200e9 / 150e9
    assert abs(ratio - expected) < 1e-6

def test_calculate_mvrv_ratio_zero_realized():
    ratio = calculate_mvrv_ratio(200e9, 0)
    assert math.isinf(ratio)

def test_analyze_exchange_flow():
    signal = analyze_exchange_flow(5e8, 3e8)
    assert signal == "distribution"
    signal2 = analyze_exchange_flow(3e8, 5e8)
    assert signal2 == "accumulation"
