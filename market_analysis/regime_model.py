# markets/regime_model.py

import numpy as np      # 수치 계산 및 배열 처리
import pandas as pd     # 데이터프레임 사용 및 데이터 처리
from hmmlearn.hmm import GaussianHMM  # Gaussian Hidden Markov Model 구현을 위한 라이브러리
from logging.logger_config import setup_logger  # 로거 설정 모듈
from datetime import timedelta  # 시간 간격 계산에 사용
import warnings  # 경고 메시지 관리를 위한 모듈

class MarketRegimeHMM:
    """
    HMM(Hidden Markov Model)을 활용하여 시장의 숨겨진 레짐(상태)를 학습, 예측하는 클래스입니다.
    
    주요 기능:
      - 과거 데이터를 사용하여 HMM 모델을 학습합니다.
      - 새로운 데이터에 대해 상태(레짐)를 예측합니다.
      - 재학습 여부를 피처 변화량 및 시간 간격에 따라 결정합니다.
    
    Attributes:
      - n_components (int): 모델이 예측할 상태(레짐)의 수.
      - covariance_type (str): 공분산 행렬의 형태 ('full', 'diag' 등).
      - n_iter (int): HMM 학습 시 최대 반복 횟수.
      - random_state (int): 재현성을 위한 랜덤 시드.
      - model (GaussianHMM): 실제 HMM 모델 객체.
      - logger: 로그 기록을 위한 로거 객체.
      - trained (bool): 모델이 학습되었는지 여부.
      - last_train_time: 마지막 학습 시각.
      - retrain_interval_minutes (int): 재학습을 위한 최소 시간 간격 (분).
      - last_feature_stats: 마지막 학습 시 사용한 피처들의 평균 통계값.
      - retrain_feature_threshold (float): 피처 평균의 변화량 임계값; 변화가 작으면 재학습을 생략.
    """
    def __init__(self, n_components: int = 3, covariance_type: str = 'full',
                 n_iter: int = 1000, random_state: int = 42, retrain_interval_minutes: int = 60):
        # ───────────────────────────────────────────────────────
        # 초기 HMM 파라미터 설정:
        #  - n_components: 모델이 가정할 숨겨진 상태의 개수 (기본 3: bullish, bearish, sideways)
        #  - covariance_type, n_iter, random_state: 모델 학습에 필요한 설정 값들
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        # GaussianHMM 모델 객체 생성
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        # ───────────────────────────────────────────────────────
        
        # 로거 객체 설정: 모듈의 이벤트, 디버그, 에러 등을 기록
        self.logger = setup_logger(__name__)
        
        # 모델 학습 상태 플래그 (초기에는 아직 학습되지 않음)
        self.trained = False
        
        # 마지막 학습 시각 (초기에는 None)
        self.last_train_time = None
        
        # 재학습 최소 간격 (분 단위)
        self.retrain_interval_minutes = retrain_interval_minutes
        
        # 마지막 학습 시 사용한 피처들의 평균 통계값 (재학습 조건 판단에 사용)
        self.last_feature_stats = None
        
        # 피처 평균 변화량 임계값: 이 값보다 변화가 작으면 재학습을 건너뛸 수 있음
        self.retrain_feature_threshold = 0.01
        # ───────────────────────────────────────────────────────

    def train(self, historical_data: pd.DataFrame, feature_columns: list = None,
              max_train_samples: int = None, min_samples: int = 50) -> None:
        """
        HMM 모델을 주어진 과거 데이터(historical_data)로 학습합니다.
        
        데이터가 충분하지 않거나(최소 샘플 수 미달) 최근 업데이트 조건(시간 간격, 피처 변화량)이 
        만족되지 않을 경우 학습을 생략할 수 있습니다.
        
        Parameters:
          - historical_data (pd.DataFrame): 학습에 사용될 과거 데이터 (인덱스에 시간 정보가 포함될 수 있음).
          - feature_columns (list): 학습에 사용할 피처 컬럼 리스트. None인 경우, 데이터프레임의 모든 컬럼 사용.
          - max_train_samples (int): 학습에 사용할 최대 샘플 수. 지정하지 않으면 전체 데이터를 사용.
          - min_samples (int): 학습을 진행하기 위한 최소 샘플 수 (기본값: 50).
        
        Returns:
          - None
        """
        # 충분한 데이터가 없는 경우, 경고 로그를 남기고 학습을 생략합니다.
        if historical_data.empty or len(historical_data) < min_samples:
            self.logger.warning(
                f"Insufficient data for HMM training: {len(historical_data)} samples available; minimum required is {min_samples}. "
                "Skipping training; default regime will be used."
            )
            self.trained = False
            return

        # 사용할 피처 컬럼 결정: 별도 지정되지 않으면 전체 컬럼 사용
        feature_columns = feature_columns or historical_data.columns.tolist()
        # max_train_samples가 지정되어 있고, 데이터 양이 많을 경우 최신 데이터만 선택
        training_data = (historical_data.iloc[-max_train_samples:]
                         if max_train_samples is not None and len(historical_data) > max_train_samples
                         else historical_data)
        # 데이터프레임의 인덱스에서 마지막(최신) 시간 추출
        current_last_time = training_data.index.max()

        # 이전에 학습한 시점이 있다면, 재학습 조건을 확인합니다.
        if self.last_train_time is not None:
            # 마지막 학습 이후 경과 시간 계산
            elapsed = current_last_time - self.last_train_time
            retrain_interval = timedelta(minutes=self.retrain_interval_minutes)
            # 이전에 저장된 피처 평균값이 있다면 현재 피처 평균과의 차이를 계산하여 재학습 여부 결정
            if self.last_feature_stats is not None:
                current_means = training_data[feature_columns].mean()
                diff = np.abs(current_means - self.last_feature_stats).mean()
                # 경과 시간이 충분하지 않고 피처 변화량이 임계값 미만이면 학습을 생략
                if elapsed < retrain_interval and diff < self.retrain_feature_threshold:
                    self.logger.debug(
                        f"HMM retraining skipped: only {elapsed.total_seconds()/60:.2f} minutes elapsed and feature mean change {diff:.6f} is below threshold {self.retrain_feature_threshold}."
                    )
                    return
            else:
                # 피처 통계값이 없는 경우에도 지정된 시간 간격이 지나지 않았다면 학습 생략
                if elapsed < retrain_interval:
                    self.logger.debug(f"HMM retraining skipped: only {elapsed.total_seconds()/60:.2f} minutes elapsed.")
                    return

        # 선택한 피처들을 numpy 배열로 변환하여 HMM 모델 학습 데이터(X) 준비
        X = training_data[feature_columns].values
        self.logger.debug(f"Starting HMM training: {X.shape[0]} samples, {X.shape[1]} features")
        try:
            # 경고를 캡처하여 모델 학습 중 발생하는 경고들을 로깅
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                self.model.fit(X)
                # 발생한 경고 중 "Model is not converging" 메시지가 있으면 경고 로그 기록
                for warn in caught_warnings:
                    if "Model is not converging" in str(warn.message):
                        self.logger.warning(f"HMM convergence warning: {warn.message}")
        except Exception as e:
            # 예외 발생 시 에러 로그를 남기고 학습 실패 플래그 설정
            self.logger.error(f"HMM training error: {e}", exc_info=True)
            self.trained = False
            return

        # 학습 성공 시, 상태 플래그와 마지막 학습 시간, 피처 평균값을 업데이트합니다.
        self.trained = True
        self.last_train_time = current_last_time
        self.last_feature_stats = training_data[feature_columns].mean()
        self.logger.debug("HMM training completed successfully.")

    def predict(self, data: pd.DataFrame, feature_columns: list = None) -> np.ndarray:
        """
        입력 데이터에 대해 HMM 모델을 사용하여 각 샘플의 상태(레짐)를 예측합니다.
        
        Parameters:
          - data (pd.DataFrame): 예측에 사용할 데이터. 데이터가 비어있으면 예외 발생.
          - feature_columns (list): 예측에 사용할 피처 컬럼 리스트. None이면 전체 컬럼 사용.
        
        Returns:
          - np.ndarray: 각 샘플에 대해 예측된 상태 번호(정수 배열).
        
        Raises:
          - ValueError: 입력 데이터가 비어있을 경우.
        """
        if data.empty:
            self.logger.error("Input data is empty. Prediction aborted.", exc_info=True)
            raise ValueError("Input data is empty.")
        
        # 사용할 피처 컬럼 결정
        feature_columns = feature_columns or data.columns.tolist()
        # 모델이 아직 학습되지 않은 경우, 기본 상태 값(여기서는 2: 'sideways')를 반환
        if not self.trained:
            self.logger.warning("HMM model is not trained. Returning default regime 'sideways' for all samples.")
            return np.full(data.shape[0], 2)
        
        # 예측 데이터 준비: 데이터프레임에서 지정된 피처 값을 numpy 배열로 추출
        X = data[feature_columns].values
        try:
            # HMM 모델을 통해 상태 예측
            predicted_states = self.model.predict(X)
        except Exception as e:
            self.logger.error(f"HMM prediction error: {e}", exc_info=True)
            raise
        self.logger.debug(f"Prediction completed: {X.shape[0]} samples processed.")
        return predicted_states

    def predict_proba(self, data: pd.DataFrame, feature_columns: list = None) -> np.ndarray:
        """
        입력 데이터에 대해 HMM 모델을 사용하여 각 상태(레짐)의 확률 분포를 예측합니다.
        
        Parameters:
          - data (pd.DataFrame): 예측에 사용할 데이터.
          - feature_columns (list): 예측에 사용할 피처 컬럼 리스트. None이면 전체 컬럼 사용.
        
        Returns:
          - np.ndarray: 각 샘플에 대해 상태별 확률 배열.
        
        Raises:
          - ValueError: 입력 데이터가 비어있을 경우.
        """
        if data.empty:
            self.logger.error("Input data is empty. predict_proba aborted.", exc_info=True)
            raise ValueError("Input data is empty.")
        
        # 사용할 피처 컬럼 결정
        feature_columns = feature_columns or data.columns.tolist()
        # 모델이 아직 학습되지 않았다면, 기본 확률 배열 반환 (여기서는 'sideways' 상태에 해당하는 확률 1)
        if not self.trained:
            self.logger.warning("HMM model is not trained. Returning default probability (all 'sideways') for all samples.")
            return np.tile([0, 0, 1], (data.shape[0], 1))
        
        # 예측 데이터 준비
        X = data[feature_columns].values
        try:
            # 각 상태의 확률 예측
            probabilities = self.model.predict_proba(X)
        except Exception as e:
            self.logger.error(f"predict_proba error: {e}", exc_info=True)
            raise
        self.logger.debug(f"predict_proba completed: {X.shape[0]} samples processed.")
        return probabilities

    def update(self, new_data: pd.DataFrame, feature_columns: list = None, max_train_samples: int = None, min_samples: int = 50) -> None:
        """
        새로운 데이터를 받아 HMM 모델을 업데이트(재학습)합니다.
        
        내부적으로 train() 함수를 호출하여 모델을 최신 데이터에 맞게 재학습합니다.
        
        Parameters:
          - new_data (pd.DataFrame): 업데이트에 사용할 새로운 데이터.
          - feature_columns (list): 사용될 피처 컬럼 리스트. None이면 전체 컬럼 사용.
          - max_train_samples (int): 사용할 최대 샘플 수.
          - min_samples (int): 재학습에 필요한 최소 샘플 수.
        
        Returns:
          - None
        """
        self.logger.debug("Starting HMM model update.")
        self.train(new_data, feature_columns, max_train_samples, min_samples)
        self.logger.debug("HMM model update completed.")

    def map_state_to_regime(self, state: int) -> str:
        """
        학습된 HMM의 상태 번호를 시장 레짐 레이블(예: bullish, bearish, sideways)로 매핑합니다.
        
        HMM 모델의 첫 번째 피처(예: 수익률)의 평균값을 기준으로,
        상태들을 오름차순 정렬하여 가장 낮은 값은 bearish, 가장 높은 값은 bullish, 중간은 sideways로 매핑합니다.
        
        Parameters:
          - state (int): HMM 모델이 예측한 상태 번호.
        
        Returns:
          - str: 매핑된 시장 레짐 레이블. 만약 매핑 정보가 없으면 "state_{state}" 형식의 문자열을 반환.
        """
        # 모델이 학습되었으며, means_ 속성이 존재하고 상태의 개수가 맞는지 확인
        if self.trained and self.model.means_ is not None and len(self.model.means_) == self.n_components:
            # 첫 번째 피처(예: returns)의 평균값 리스트 생성
            returns_means = [mean[0] for mean in self.model.means_]
            # 평균값을 기준으로 오름차순 정렬한 인덱스를 얻음:
            # 낮은 값부터 높은 값까지 정렬되며, 이를 통해 각 상태를 bearish, sideways, bullish으로 매핑 가능
            sorted_indices = sorted(range(len(returns_means)), key=lambda i: returns_means[i])
            mapping = {}
            if self.n_components == 3:
                mapping[sorted_indices[0]] = "bearish"
                mapping[sorted_indices[1]] = "sideways"
                mapping[sorted_indices[2]] = "bullish"
                return mapping.get(state, f"state_{state}")
            else:
                # 상태 개수가 3가 아니면 기본 문자열 반환
                return f"state_{state}"
        else:
            # 모델이 학습되지 않았거나 means_ 속성이 없으면 기본 문자열 반환
            return f"state_{state}"

    def predict_regime_labels(self, data: pd.DataFrame, feature_columns: list = None) -> list:
        """
        입력 데이터에 대해 HMM 모델을 사용하여 각 샘플의 시장 레짐 레이블(예: bullish, bearish, sideways)을 예측합니다.
        
        내부적으로 predict()를 통해 상태 번호를 예측한 뒤,
        map_state_to_regime() 함수를 사용하여 상태 번호를 의미있는 레이블로 변환합니다.
        
        Parameters:
          - data (pd.DataFrame): 예측에 사용할 데이터.
          - feature_columns (list): 사용할 피처 컬럼 리스트. None이면 전체 컬럼 사용.
        
        Returns:
          - list: 각 샘플에 대해 예측된 시장 레짐 레이블의 리스트.
        """
        # 먼저 HMM 모델을 사용하여 상태 번호를 예측합니다.
        states = self.predict(data, feature_columns)
        # 각 상태 번호를 대응하는 레짐 레이블로 변환합니다.
        regime_labels = [self.map_state_to_regime(s) for s in states]
        self.logger.debug(f"Regime labels predicted: {regime_labels}")
        return regime_labels
