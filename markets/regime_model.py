# markets/regime_model.py
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from logs.logger_config import setup_logger
from datetime import timedelta
import warnings

class MarketRegimeHMM:
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
        self.retrain_feature_threshold = 0.01

    def train(self, historical_data: pd.DataFrame, feature_columns: list = None,
              max_train_samples: int = None, min_samples: int = 50) -> None:
        if historical_data.empty or len(historical_data) < min_samples:
            self.logger.warning(
                f"Insufficient data for HMM training: {len(historical_data)} samples available; minimum required is {min_samples}. "
                "Skipping training; default regime will be used."
            )
            self.trained = False
            return

        feature_columns = feature_columns or historical_data.columns.tolist()
        training_data = (historical_data.iloc[-max_train_samples:]
                         if max_train_samples is not None and len(historical_data) > max_train_samples
                         else historical_data)
        current_last_time = training_data.index.max()

        if self.last_train_time is not None:
            elapsed = current_last_time - self.last_train_time
            if elapsed < timedelta(minutes=self.retrain_interval_minutes):
                self.logger.info(f"HMM retraining skipped: only {elapsed.total_seconds()/60:.2f} minutes elapsed since last training.")
                return
            if self.last_feature_stats is not None:
                current_means = training_data[feature_columns].mean()
                diff = np.abs(current_means - self.last_feature_stats).mean()
                if diff < self.retrain_feature_threshold:
                    self.logger.info(f"HMM retraining skipped: feature mean change {diff:.6f} below threshold {self.retrain_feature_threshold}.")
                    return

        X = training_data[feature_columns].values
        self.logger.info(f"Starting HMM training: {X.shape[0]} samples, {X.shape[1]} features")
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                self.model.fit(X)
                for warn in caught_warnings:
                    if "Model is not converging" in str(warn.message):
                        self.logger.warning(f"HMM convergence warning: {warn.message}")
        except Exception as e:
            self.logger.error(f"HMM training error: {e}", exc_info=True)
            self.trained = False
            return

        self.trained = True
        self.last_train_time = current_last_time
        self.last_feature_stats = training_data[feature_columns].mean()
        self.logger.info("HMM training completed successfully.")

    def predict(self, data: pd.DataFrame, feature_columns: list = None) -> np.ndarray:
        if data.empty:
            self.logger.error("Input data is empty. Prediction aborted.", exc_info=True)
            raise ValueError("Input data is empty.")
        feature_columns = feature_columns or data.columns.tolist()
        if not self.trained:
            self.logger.warning("HMM model is not trained. Returning default regime 'sideways' for all samples.")
            return np.full(data.shape[0], 2)
        X = data[feature_columns].values
        try:
            predicted_states = self.model.predict(X)
        except Exception as e:
            self.logger.error(f"HMM prediction error: {e}", exc_info=True)
            raise
        self.logger.info(f"Prediction completed: {X.shape[0]} samples processed.")
        return predicted_states

    def predict_proba(self, data: pd.DataFrame, feature_columns: list = None) -> np.ndarray:
        if data.empty:
            self.logger.error("Input data is empty. predict_proba aborted.", exc_info=True)
            raise ValueError("Input data is empty.")
        feature_columns = feature_columns or data.columns.tolist()
        if not self.trained:
            self.logger.warning("HMM model is not trained. Returning default probability (all 'sideways') for all samples.")
            return np.tile([0, 0, 1], (data.shape[0], 1))
        X = data[feature_columns].values
        try:
            probabilities = self.model.predict_proba(X)
        except Exception as e:
            self.logger.error(f"predict_proba error: {e}", exc_info=True)
            raise
        self.logger.info(f"predict_proba completed: {X.shape[0]} samples processed.")
        return probabilities

    def update(self, new_data: pd.DataFrame, feature_columns: list = None, max_train_samples: int = None, min_samples: int = 50) -> None:
        self.logger.info("Starting HMM model update.")
        self.train(new_data, feature_columns, max_train_samples, min_samples)
        self.logger.info("HMM model update completed.")

    def map_state_to_regime(self, state: int) -> str:
        if self.n_components == 3:
            mapping = {0: "bullish", 1: "bearish", 2: "sideways"}
            return mapping.get(state, f"state_{state}")
        else:
            return f"state_{state}"

    def predict_regime_labels(self, data: pd.DataFrame, feature_columns: list = None) -> list:
        states = self.predict(data, feature_columns)
        regime_labels = [self.map_state_to_regime(s) for s in states]
        self.logger.info(f"Regime labels predicted: {regime_labels}")
        return regime_labels
