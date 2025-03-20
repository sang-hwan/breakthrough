# tests/unit_tests/market_analysis/test_analyze_market.py
import pytest
import numpy as np
import pandas as pd
from market_analysis.ml_market_analysis import MarketLSTMAnalyzer

def test_ml_model_not_trained():
    # 학습되지 않은 모델은 예측 시 기본값(여기서는 'sideways'를 나타내는 2)를 반환하도록 설계됨
    analyzer = MarketLSTMAnalyzer(input_shape=(10, 5))
    ml_input = pd.DataFrame(np.random.rand(10, 5))
    predictions = analyzer.predict(ml_input)
    np.testing.assert_array_equal(predictions, np.full((10,), 2))
