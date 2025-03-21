# optimization/signal_optimize.py
import optuna
import logging
from logs.log_config import setup_logger
from parameters.signal_parameters import ConfigManager  # 신호 계산 파라미터 관리
from signal_calc.calc_signal import SignalCalculator  # 신호 계산 수행 클래스

logger = setup_logger(__name__)

class SignalParameterOptimizer:
    """
    SignalParameterOptimizer는 신호 계산 모듈의 파라미터 최적화를 수행합니다.
    
    Attributes:
        n_trials (int): 최적화 시도 횟수.
        study (optuna.study.Study): 최적화 결과를 저장하는 study 객체.
        config_manager (ConfigManager): 기본 파라미터 관리 및 검증 객체.
    """

    def __init__(self, n_trials=10):
        """
        초기화 함수.
        
        Parameters:
            n_trials (int): 최적화 시도 횟수 (기본값: 10).
        """
        self.n_trials = n_trials
        self.study = None
        self.config_manager = ConfigManager()

    def objective(self, trial):
        """
        optuna의 objective 함수.
        
        제안받은 파라미터 조합에 대해 신호 계산 모듈의 성능을 평가하고,
        기본 파라미터와의 차이에 따른 정규화 패널티를 적용합니다.
        
        Parameters:
            trial (optuna.trial.Trial): 현재 최적화 시도의 trial 객체.
        
        Returns:
            float: 성능 점수 (낮을수록 우수).
        """
        try:
            base_params = self.config_manager.get_defaults()
            suggested_params = {
                "rsi_period": trial.suggest_int("rsi_period", 10, 30),
                "macd_fast": trial.suggest_int("macd_fast", 10, 20),
                "macd_slow": trial.suggest_int("macd_slow", 20, 40),
                # 추가 파라미터 정의 가능
            }
            params = {**base_params, **suggested_params}
            params = self.config_manager.validate_params(params)

            calculator = SignalCalculator(**params)
            signal_score = calculator.evaluate_signals()  # 신호 성능 평가

            # 정규화 패널티 적용
            reg_penalty = 0.1 * sum(
                (params.get(k, base_params.get(k)) - base_params.get(k)) ** 2
                for k in suggested_params
            )
            score = signal_score + reg_penalty
            return score
        except Exception as e:
            logger.error("Error in signal objective: " + str(e), exc_info=True)
            return 1e6

    def optimize(self):
        """
        최적화 실행 함수.
        
        n_trials 횟수 동안 objective 함수를 최적화하여 최적의 파라미터 조합을 도출합니다.
        
        Returns:
            optuna.trial.FrozenTrial: 최적의 파라미터 조합과 점수를 포함하는 trial 객체.
        """
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        best_trial = self.study.best_trial
        logger.info(f"Best trial: {best_trial.number}, Value: {best_trial.value:.2f}")
        logger.info(f"Best parameters: {best_trial.params}")
        return best_trial
