# tests/market/test_regime_model.py
# 이 파일은 MarketRegimeHMM 모델의 학습 기능을 테스트합니다.
# HMM (Hidden Markov Model)을 사용하여 시장 레짐을 학습하는 기능이 샘플 수에 따라 올바르게 동작하는지 검증합니다.

import pandas as pd
import numpy as np
from markets.regime_model import MarketRegimeHMM

def test_hmm_training_with_sufficient_samples():
    """
    충분한 샘플(50개 이상)을 가진 데이터로 HMM 학습 테스트

    목적:
      - 충분한 데이터(여기서는 60일치 데이터)를 사용했을 때, HMM 모델이 정상적으로 학습되고
        학습 완료 상태(trained)가 True가 되는지 확인.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문으로 학습 완료 여부 검증)
    """
    # 2020-01-01부터 시작하여 60일치 날짜 생성
    dates = pd.date_range(start='2020-01-01', periods=60, freq='D')
    # 2개의 랜덤 피처(feature1, feature2)를 포함하는 데이터프레임 생성
    df = pd.DataFrame({
        'feature1': np.random.randn(60),
        'feature2': np.random.randn(60)
    }, index=dates)
    # HMM 모델 인스턴스 생성 (3개의 상태 구성)
    hmm_model = MarketRegimeHMM(n_components=3)
    # feature1, feature2 컬럼을 사용하여 모델 학습
    hmm_model.train(df, feature_columns=['feature1', 'feature2'])
    # 충분한 샘플로 인해 학습이 성공적으로 완료되어야 함
    assert hmm_model.trained is True
    assert hmm_model.last_train_time is not None

def test_hmm_training_insufficient_samples():
    """
    샘플 수가 부족할 때 HMM 학습 테스트

    목적:
      - 50개 미만(여기서는 30일치)의 샘플을 사용하면, HMM 모델이 학습을 진행하지 않고
        학습 완료 상태(trained)가 False가 되는지 확인.
    
    Parameters:
      없음

    Returns:
      없음 (assert 구문으로 학습 미진행 여부 검증)
    """
    # 2020-01-01부터 시작하여 30일치 날짜 생성
    dates = pd.date_range(start='2020-01-01', periods=30, freq='D')
    df = pd.DataFrame({
        'feature1': np.random.randn(30),
        'feature2': np.random.randn(30)
    }, index=dates)
    # HMM 모델 인스턴스 생성 (3개의 상태 구성)
    hmm_model = MarketRegimeHMM(n_components=3)
    # 학습 시도: 데이터 샘플이 부족하여 학습이 진행되지 않아야 함
    hmm_model.train(df, feature_columns=['feature1', 'feature2'])
    assert hmm_model.trained is False
