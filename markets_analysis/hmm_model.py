# markets_analysis/hmm_model.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from logs.logger_config import setup_logger

class MarketRegimeHMM:
    def __init__(self, n_components=3, covariance_type='full', n_iter=1000, random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = GaussianHMM(n_components=self.n_components,
                                 covariance_type=self.covariance_type,
                                 n_iter=self.n_iter,
                                 random_state=self.random_state)
        # setup_logger를 통해 로거를 설정 (모듈명과 함수명 등이 로그에 포함됨)
        self.logger = setup_logger(__name__)
        self.trained = False

    def train(self, historical_data: pd.DataFrame, feature_columns: list = None):
        if historical_data.empty:
            self.logger.error("Historical data is empty. Training aborted.")
            raise ValueError("Historical data is empty.")
        if feature_columns is None:
            feature_columns = historical_data.columns.tolist()
        
        X = historical_data[feature_columns].values
        self.logger.info(f"Training HMM model with {X.shape[0]} samples and {X.shape[1]} features.")
        self.model.fit(X)
        self.trained = True
        self.logger.info("HMM training completed.")

    def predict(self, data: pd.DataFrame, feature_columns: list = None):
        if not self.trained:
            self.logger.error("Model is not trained. Prediction aborted.")
            raise ValueError("Model is not trained.")
        if data.empty:
            self.logger.error("Input data is empty. Prediction aborted.")
            raise ValueError("Input data is empty.")
        if feature_columns is None:
            feature_columns = data.columns.tolist()
        
        X = data[feature_columns].values
        predicted_states = self.model.predict(X)
        self.logger.info(f"Predicted states for {X.shape[0]} samples.")
        return predicted_states

    def update(self, new_data: pd.DataFrame, feature_columns: list = None):
        self.logger.info("Updating HMM model with new data.")
        if feature_columns is None:
            feature_columns = new_data.columns.tolist()
        X_new = new_data[feature_columns].values
        self.model.fit(X_new)
        self.logger.info("HMM model update completed.")
