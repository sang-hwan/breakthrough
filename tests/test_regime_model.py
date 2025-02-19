# tests/test_regime_model.py
import pandas as pd
import numpy as np
from markets.regime_model import MarketRegimeHMM

def test_hmm_training_with_sufficient_samples():
    # 최소 50개 이상의 샘플을 갖는 데이터프레임 생성 (여기서는 60개)
    dates = pd.date_range(start='2020-01-01', periods=60, freq='D')
    df = pd.DataFrame({
        'feature1': np.random.randn(60),
        'feature2': np.random.randn(60)
    }, index=dates)
    hmm_model = MarketRegimeHMM(n_components=3)
    hmm_model.train(df, feature_columns=['feature1', 'feature2'])
    # 충분한 샘플이 있어 학습이 완료되어야 함
    assert hmm_model.trained is True
    assert hmm_model.last_train_time is not None

def test_hmm_training_insufficient_samples():
    # 샘플 수가 50개 미만인 데이터프레임 생성 (여기서는 30개)
    dates = pd.date_range(start='2020-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'feature1': np.random.randn(30),
        'feature2': np.random.randn(30)
    }, index=dates)
    hmm_model = MarketRegimeHMM(n_components=3)
    hmm_model.train(df, feature_columns=['feature1', 'feature2'])
    # 샘플 수 부족으로 학습이 진행되지 않아야 함
    assert hmm_model.trained is False
