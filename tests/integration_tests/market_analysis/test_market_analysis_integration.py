# tests/integration_tests/market_analysis/test_market_analysis_integration.py
import pytest
import pandas as pd
import numpy as np
from market_analysis.analyze_market import aggregate_market_analysis

# 더미 ML 모델: 예측값을 고정값(예, 105.0)으로 반환하여 bullish 신호를 생성
class DummyMLModel:
    def predict(self, ml_input):
        return np.full((ml_input.shape[0], 1), 105.0)

@pytest.fixture
def technical_data():
    dates = pd.date_range("2023-01-01", periods=100)
    price_series = pd.Series(np.linspace(100, 110, 100), index=dates)
    return {"price_series": price_series, "rsi_window": 14}

@pytest.fixture
def onchain_data():
    # mvrv 계산: market_cap=200e9, realized_cap=250e9 → mvrv=0.8 (bullish 조건)
    # 거래소 데이터: exchange_inflow=3e8, exchange_outflow=5e8 → "accumulation" (bullish)
    return {
        "market_cap": 200e9,
        "realized_cap": 250e9,
        "exchange_inflow": 3e8,
        "exchange_outflow": 5e8
    }

@pytest.fixture
def sentiment_texts():
    return ["The market is bullish.", "Investors are positive and optimistic."]

@pytest.fixture
def ml_data():
    dummy_ml_model = DummyMLModel()
    ml_input = pd.DataFrame(np.random.rand(10, 5))
    current_price = 100.0
    return {"ml_model": dummy_ml_model, "ml_input": ml_input, "current_price": current_price}

def test_aggregate_market_analysis(technical_data, onchain_data, sentiment_texts, ml_data):
    final_state = aggregate_market_analysis(technical_data, onchain_data, sentiment_texts, ml_data)
    # 통합 결과는 각 분석 영역에서 산출한 신호에 따라 "bullish", "bearish" 또는 "sideways" 중 하나여야 함
    assert final_state in ["bullish", "bearish", "sideways"]
