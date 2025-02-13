# markets/regime_model.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from logs.logger_config import setup_logger
from datetime import timedelta
import warnings

class MarketRegimeHMM:
    """
    HMM 기반 시장 레짐 분석 모델.
    재학습은 마지막 학습 이후 지정한 시간 간격과 피처 변화 임계값을 만족할 때만 수행됩니다.
    """
    def __init__(self, n_components: int = 3, covariance_type: str = 'full',
                 n_iter: int = 1000, random_state: int = 42, retrain_interval_minutes: int = 60):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        self.logger = setup_logger(__name__)
        self.trained = False
        self.last_train_time = None
        self.retrain_interval_minutes = retrain_interval_minutes
        self.last_feature_stats = None
        self.retrain_feature_threshold = 0.01  # 평균 피처 변화 임계값

    def train(self, historical_data: pd.DataFrame, feature_columns: list = None,
              max_train_samples: int = None) -> None:
        """
        HMM 모델 학습.
          - historical_data: 학습에 사용할 데이터 (인덱스는 datetime)
          - feature_columns: 사용할 피처 목록 (None이면 전체 컬럼 사용)
          - max_train_samples: 최신 샘플 수만 사용할 경우 지정
          
        재학습 조건(시간 및 피처 변화)이 충족되지 않으면 학습을 건너뜁니다.
        """
        if historical_data.empty:
            self.logger.error("Historical data is empty. Training aborted.")
            raise ValueError("Historical data is empty.")
        feature_columns = feature_columns or historical_data.columns.tolist()

        training_data = (historical_data.iloc[-max_train_samples:]
                         if max_train_samples is not None and len(historical_data) > max_train_samples
                         else historical_data)
        current_last_time = training_data.index.max()

        if self.last_train_time is not None:
            elapsed = current_last_time - self.last_train_time
            if elapsed < timedelta(minutes=self.retrain_interval_minutes):
                self.logger.debug(f"HMM 재학습 건너뜀: 마지막 학습 후 {elapsed.total_seconds()/60:.2f}분 경과.")
                return
            if self.last_feature_stats is not None:
                current_means = training_data[feature_columns].mean()
                diff = np.abs(current_means - self.last_feature_stats).mean()
                if diff < self.retrain_feature_threshold:
                    self.logger.debug(f"HMM 재학습 건너뜀: 피처 평균 변화 {diff:.6f} < {self.retrain_feature_threshold}.")
                    return

        X = training_data[feature_columns].values
        self.logger.debug(f"HMM 학습 시작: 샘플 {X.shape[0]}, 피처 {X.shape[1]}")
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                self.model.fit(X)
                for warn in caught_warnings:
                    if "Model is not converging" in str(warn.message):
                        self.logger.warning(f"HMM 수렴 경고: {warn.message}")
        except Exception as e:
            self.logger.error(f"HMM 학습 에러: {e}", exc_info=True)
            raise

        self.trained = True
        self.last_train_time = current_last_time
        self.last_feature_stats = training_data[feature_columns].mean()
        self.logger.debug("HMM 학습 완료.")

    def predict(self, data: pd.DataFrame, feature_columns: list = None) -> np.ndarray:
        """
        주어진 데이터에 대해 HMM 모델 상태를 예측합니다.
        """
        if not self.trained:
            self.logger.error("모델 미학습 상태. 예측 중단.")
            raise ValueError("Model is not trained.")
        if data.empty:
            self.logger.error("입력 데이터 없음. 예측 중단.")
            raise ValueError("Input data is empty.")
        feature_columns = feature_columns or data.columns.tolist()
        X = data[feature_columns].values
        try:
            predicted_states = self.model.predict(X)
        except Exception as e:
            self.logger.error(f"HMM 예측 에러: {e}", exc_info=True)
            raise
        self.logger.debug(f"예측 완료: {X.shape[0]} 샘플")
        return predicted_states

    def predict_proba(self, data: pd.DataFrame, feature_columns: list = None) -> np.ndarray:
        """
        각 상태에 대한 예측 확률을 반환합니다.
        """
        if not self.trained:
            self.logger.error("모델 미학습 상태. predict_proba 중단.")
            raise ValueError("Model is not trained.")
        if data.empty:
            self.logger.error("입력 데이터 없음. predict_proba 중단.")
            raise ValueError("Input data is empty.")
        feature_columns = feature_columns or data.columns.tolist()
        X = data[feature_columns].values
        try:
            probabilities = self.model.predict_proba(X)
        except Exception as e:
            self.logger.error(f"predict_proba 에러: {e}", exc_info=True)
            raise
        self.logger.debug(f"predict_proba 완료: {X.shape[0]} 샘플")
        return probabilities

    def update(self, new_data: pd.DataFrame, feature_columns: list = None, max_train_samples: int = None) -> None:
        """
        새 데이터를 반영하여 모델을 재학습합니다.
        """
        self.logger.debug("HMM 모델 업데이트 시작.")
        self.train(new_data, feature_columns, max_train_samples)
        self.logger.debug("HMM 모델 업데이트 완료.")
