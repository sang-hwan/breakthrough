# markets_analysis/hmm_model.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from logs.logger_config import setup_logger
from datetime import timedelta

class MarketRegimeHMM:
    def __init__(self, n_components=3, covariance_type='full', n_iter=1000, random_state=42, retrain_interval_minutes=60):
        """
        HMM 모델 초기화.
        retrain_interval_minutes: 마지막 재학습 시각 이후 최소 재학습 간격(분)
        """
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
        self.last_train_time = None  # 마지막 학습 데이터의 마지막 타임스탬프
        self.retrain_interval_minutes = retrain_interval_minutes
        self.last_feature_stats = None  # 이전 학습 시 사용된 피처들의 평균값 저장
        self.retrain_feature_threshold = 0.01  # 피처 평균 변화의 임계값 (예: 1% 미만이면 재학습하지 않음)

    def train(self, historical_data: pd.DataFrame, feature_columns: list = None, max_train_samples: int = None):
        """
        HMM 모델 학습:
         - historical_data: 학습에 사용할 데이터프레임 (인덱스는 datetime)
         - feature_columns: 사용할 피처 목록 (None이면 전체 컬럼 사용)
         - max_train_samples: 최신 샘플 수만 사용할 경우 지정

         마지막 학습 시각 및 피처 평균 값의 변화가 작으면 재학습을 건너뜁니다.
        """
        if historical_data.empty:
            self.logger.error("Historical data is empty. Training aborted.")
            raise ValueError("Historical data is empty.")
        if feature_columns is None:
            feature_columns = historical_data.columns.tolist()

        # 최대 샘플 수 지정 시 최신 데이터만 사용 (INFO 레벨로 기록하여 집계 대상에 포함)
        if max_train_samples is not None and len(historical_data) > max_train_samples:
            training_data = historical_data.iloc[-max_train_samples:]
            self.logger.info(f"Using last {max_train_samples} samples for training.")
        else:
            training_data = historical_data

        current_last_time = training_data.index.max()

        # 시간 기반 재학습 조건 확인 (재학습 스킵 여부를 INFO 레벨로 기록)
        if self.last_train_time is not None:
            elapsed = current_last_time - self.last_train_time
            if elapsed < timedelta(minutes=self.retrain_interval_minutes):
                self.logger.info(f"Skipping HMM retraining: only {elapsed.total_seconds()/60:.2f} minutes elapsed since last training.")
                return
            # 피처 변화 기반 조건: 이전 학습 시의 피처 평균과 현재 피처 평균의 차이가 작으면 재학습 건너뜀
            if self.last_feature_stats is not None:
                current_means = training_data[feature_columns].mean()
                diff = np.abs(current_means - self.last_feature_stats).mean()
                if diff < self.retrain_feature_threshold:
                    self.logger.info(f"Skipping HMM retraining: average feature mean difference {diff:.6f} below threshold {self.retrain_feature_threshold}.")
                    return

        # 학습 데이터 준비 및 모델 학습 (주요 단계는 INFO 레벨로 기록)
        X = training_data[feature_columns].values
        self.logger.info(f"Training HMM model with {X.shape[0]} samples and {X.shape[1]} features.")
        try:
            self.model.fit(X)
        except Exception as e:
            self.logger.error(f"HMM 모델 학습 에러: {e}", exc_info=True)
            raise
        self.trained = True
        self.last_train_time = current_last_time
        self.last_feature_stats = training_data[feature_columns].mean()
        self.logger.info("HMM training completed.")

    def predict(self, data: pd.DataFrame, feature_columns: list = None):
        """
        주어진 데이터에 대해 HMM 모델로 상태 예측.
        """
        if not self.trained:
            self.logger.error("Model is not trained. Prediction aborted.")
            raise ValueError("Model is not trained.")
        if data.empty:
            self.logger.error("Input data is empty. Prediction aborted.")
            raise ValueError("Input data is empty.")
        if feature_columns is None:
            feature_columns = data.columns.tolist()
        X = data[feature_columns].values
        try:
            predicted_states = self.model.predict(X)
        except Exception as e:
            self.logger.error(f"HMM 예측 에러: {e}", exc_info=True)
            raise
        self.logger.info(f"Predicted states for {X.shape[0]} samples.")
        return predicted_states

    def predict_proba(self, data: pd.DataFrame, feature_columns: list = None):
        """
        주어진 데이터에 대해 각 상태의 예측 확률을 반환합니다.
        """
        if not self.trained:
            self.logger.error("Model is not trained. predict_proba aborted.")
            raise ValueError("Model is not trained.")
        if data.empty:
            self.logger.error("Input data is empty. predict_proba aborted.")
            raise ValueError("Input data is empty.")
        if feature_columns is None:
            feature_columns = data.columns.tolist()
        X = data[feature_columns].values
        try:
            probabilities = self.model.predict_proba(X)
        except Exception as e:
            self.logger.error(f"Error in predict_proba: {e}", exc_info=True)
            raise
        self.logger.info(f"Predicted probabilities for {X.shape[0]} samples.")
        return probabilities

    def update(self, new_data: pd.DataFrame, feature_columns: list = None, max_train_samples: int = None):
        """
        새 데이터가 들어올 때, 학습 조건에 따라 HMM 모델을 재학습합니다.
        """
        self.logger.info("Updating HMM model with new data.")
        self.train(new_data, feature_columns, max_train_samples)
        self.logger.info("HMM model update completed.")
