# tests/unit_tests/market_analysis/test_analyze_market.py
import pytest
import pandas as pd
import numpy as np
from market_analysis.analyze_market import (
    get_technical_signal,
    get_onchain_signal,
    get_sentiment_signal,
    get_ml_signal,
    determine_final_market_state,
    aggregate_market_analysis
)

# 더미 ML 모델 (예측값을 고정값으로 반환)
class DummyMLModel:
    def predict(self, ml_input):
        # ml_input의 샘플 수 만큼 예측값 105.0을 반환 (예: 현재 가격 100일 때 bullish 신호)
        return np.full((ml_input.shape[0], 1), 105.0)

@pytest.fixture
def price_series():
    dates = pd.date_range("2023-01-01", periods=100)
    # 점진적으로 상승하는 가격 시계열 생성
    return pd.Series(np.linspace(100, 110, 100), index=dates)

def test_get_technical_signal(price_series):
    signal = get_technical_signal(price_series, window=14)
    # 단순 검증: 결과가 지정된 상태 문자열 중 하나여야 함
    assert signal in ["bullish", "bearish", "sideways"]

def test_get_onchain_signal():
    # 예제 1: mvrv = 200e9/150e9 ≈1.33 (< threshold=2.0) → bullish, 
    # 그러나 exchange_inflow=5e8, exchange_outflow=3e8 -> analyze_exchange_flow returns "distribution" → bearish.
    # 서로 상반되므로 최종 신호는 "sideways"로 판단
    signal = get_onchain_signal(200e9, 150e9, 5e8, 3e8, mvrv_threshold=2.0)
    assert signal == "sideways"
    
    # 예제 2: exchange_inflow < exchange_outflow → "accumulation" → bullish 신호로 일치하는 경우
    signal2 = get_onchain_signal(200e9, 150e9, 3e8, 5e8, mvrv_threshold=2.0)
    assert signal2 == "bullish"

def test_get_sentiment_signal():
    bullish_texts = ["The market is bullish and strong gains.", "Investors are positive and optimistic."]
    bearish_texts = ["The market is bearish and weak.", "Investors are negative and cautious."]
    neutral_texts = ["The market is stable.", "No significant movement observed."]
    
    bullish_signal = get_sentiment_signal(bullish_texts)
    bearish_signal = get_sentiment_signal(bearish_texts)
    neutral_signal = get_sentiment_signal(neutral_texts)
    
    assert bullish_signal == "bullish"
    assert bearish_signal == "bearish"
    # 중립 텍스트는 보수적으로 "sideways"를 반환할 수 있음
    assert neutral_signal == "sideways"

def test_get_ml_signal():
    dummy_ml_model = DummyMLModel()
    ml_input = pd.DataFrame(np.random.rand(10, 5))
    current_price = 100.0
    # DummyMLModel은 105를 반환하므로, 105 > 100 * 1.01 → bullish 신호가 예상됨
    signal = get_ml_signal(dummy_ml_model, ml_input, current_price)
    assert signal == "bullish"
    
    # bearish 조건을 테스트하기 위해 DummyMLModel의 예측값을 95로 변경
    class DummyMLModelBearish:
        def predict(self, ml_input):
            return np.full((ml_input.shape[0], 1), 95.0)
    dummy_ml_model_bearish = DummyMLModelBearish()
    signal_bearish = get_ml_signal(dummy_ml_model_bearish, ml_input, current_price)
    assert signal_bearish == "bearish"

def test_determine_final_market_state():
    # 다수결 방식 검증
    signals = ("bullish", "bullish", "bearish", "sideways")
    state = determine_final_market_state(*signals)
    # bullish: 2회, bearish: 1회, sideways: 1회 → 최종은 "bullish"
    assert state == "bullish"
    
    # 신호 간 동점인 경우 (예: bullish와 bearish가 동일 회수) 보수적으로 "sideways" 반환
    signals_tie = ("bullish", "bearish", "bullish", "bearish")
    state_tie = determine_final_market_state(*signals_tie)
    assert state_tie == "sideways"

def test_aggregate_market_analysis(price_series):
    technical_data = {"price_series": price_series, "rsi_window": 14}
    onchain_data = {
        "market_cap": 200e9,
        "realized_cap": 150e9,
        "exchange_inflow": 3e8,
        "exchange_outflow": 5e8
    }
    sentiment_texts = ["The market is bullish and showing strength.", "Optimism is high among investors."]
    dummy_ml_model = DummyMLModel()
    ml_input = pd.DataFrame(np.random.rand(10, 5))
    ml_data = {"ml_model": dummy_ml_model, "ml_input": ml_input, "current_price": price_series.iloc[-1]}
    
    final_state = aggregate_market_analysis(technical_data, onchain_data, sentiment_texts, ml_data)
    assert final_state in ["bullish", "bearish", "sideways"]
