# market_analysis/ml_market_analysis.py
from logs.log_config import setup_logger
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

logger = setup_logger(__name__)

class MarketLSTMAnalyzer:
    """
    LSTM 모델을 활용한 머신러닝 기반 시장 분석 클래스입니다.
    
    Attributes:
        model (tf.keras.Model): LSTM 모델 객체.
        input_shape (tuple): (timesteps, features) 형태의 입력 데이터.
    """
    def __init__(self, input_shape: tuple):
        """
        초기화: LSTM 모델 구성 및 컴파일.
        
        Parameters:
            input_shape (tuple): (timesteps, features) 형태의 입력 데이터.
        """
        self.input_shape = input_shape
        self.model = self.build_model()
        logger.debug("MarketLSTMAnalyzer initialized.")
        
    def build_model(self) -> tf.keras.Model:
        """
        LSTM 모델을 구성하고 컴파일합니다.
        
        Returns:
            tf.keras.Model: 구성된 LSTM 모델.
        """
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=self.input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        logger.debug("LSTM model built and compiled.")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2) -> None:
        """
        LSTM 모델을 주어진 데이터로 학습합니다.
        
        Parameters:
            X_train (np.ndarray): 학습 입력 데이터.
            y_train (np.ndarray): 학습 대상 데이터.
            epochs (int): 학습 에포크 수.
            batch_size (int): 배치 크기.
            validation_split (float): 검증 데이터 비율.
        
        Returns:
            None
        """
        try:
            early_stop = EarlyStopping(monitor='val_loss', patience=5)
            self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stop],
                verbose=1
            )
            logger.debug("LSTM model training completed.")
        except Exception as e:
            logger.error(f"Error during LSTM training: {e}", exc_info=True)
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        주어진 입력 데이터에 대해 LSTM 모델의 예측을 수행합니다.
        
        Parameters:
            X (np.ndarray): 예측에 사용할 입력 데이터.
            
        Returns:
            np.ndarray: 예측 결과 배열.
        """
        try:
            predictions = self.model.predict(X)
            logger.debug(f"LSTM model prediction completed for {X.shape[0]} samples.")
            return predictions
        except Exception as e:
            logger.error(f"Error during LSTM prediction: {e}", exc_info=True)
            raise
