# tests/test_regime_model.py
import pytest
import pandas as pd
import numpy as np
from markets.regime_model import MarketRegimeHMM

@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "returns": np.random.normal(0, 0.01, 100),
        "volatility": np.random.normal(0.02, 0.005, 100),
        "sma": np.linspace(100, 110, 100),
        "rsi": np.random.uniform(30, 70, 100),
        "macd_macd": np.random.normal(0, 1, 100),
        "macd_signal": np.random.normal(0, 1, 100),
        "macd_diff": np.random.normal(0, 0.5, 100),
    }, index=dates)
    return df

def test_train_and_predict(sample_data):
    hmm = MarketRegimeHMM(n_components=3, retrain_interval_minutes=0)
    # 첫 학습 시도
    hmm.train(sample_data, feature_columns=["returns", "volatility", "sma", "rsi", "macd_macd", "macd_signal", "macd_diff"])
    assert hmm.trained
    states = hmm.predict(sample_data, feature_columns=["returns", "volatility", "sma", "rsi", "macd_macd", "macd_signal", "macd_diff"])
    assert len(states) == len(sample_data)
