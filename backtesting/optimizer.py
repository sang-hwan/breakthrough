# backtesting/optimizer.py
import optuna
import pandas as pd
from logs.logger_config import setup_logger
from backtesting.backtester import Backtester
from dynamic_parameters.dynamic_param_manager import DynamicParamManager

logger = setup_logger(__name__)

class DynamicParameterOptimizer:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.study = None
        self.dynamic_param_manager = DynamicParamManager()
    
    def objective(self, trial):
        # 기본 파라미터 가져오기 및 최적화 대상 파라미터 제안 (불필요한 추세 관련 파라미터 제거)
        base_params = self.dynamic_param_manager.get_default_params()
        params = {
            "hmm_confidence_threshold": trial.suggest_float("hmm_confidence_threshold", 0.7, 0.95),
            "liquidity_info": trial.suggest_categorical("liquidity_info", ["high", "low"]),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 3.0),
            "profit_ratio": trial.suggest_float("profit_ratio", 0.05, 0.15),
            "risk_per_trade": trial.suggest_float("risk_per_trade", 0.005, 0.02),
            "scale_in_threshold": trial.suggest_float("scale_in_threshold", 0.01, 0.03),
            "partial_exit_ratio": trial.suggest_float("partial_exit_ratio", 0.4, 0.6),
            "partial_profit_ratio": trial.suggest_float("partial_profit_ratio", 0.02, 0.04),
            "final_profit_ratio": trial.suggest_float("final_profit_ratio", 0.05, 0.1)
        }
        # 최종 동적 파라미터
        dynamic_params = {**base_params, **params}
        
        # Walk-Forward 구간 정의
        splits = [
            {"train_start": "2018-06-01", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
            {"train_start": "2019-06-01", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
            {"train_start": "2020-06-01", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
            {"train_start": "2021-06-01", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
            {"train_start": "2022-06-01", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-02-01"},
        ]
        
        total_score = 0.0
        for split in splits:
            # 학습 구간 백테스트
            backtester = Backtester(symbol="BTC/USDT", account_size=10000)
            backtester.load_data(
                "ohlcv_{symbol}_{timeframe}",
                "ohlcv_{symbol}_{timeframe}",
                "4h", "1d",
                split["train_start"], split["train_end"]
            )
            try:
                trades_train, _ = backtester.run_backtest(dynamic_params)
            except Exception as e:
                logger.error(f"Training backtest failed: {e}")
                return 1e6
            total_pnl_train = sum(trade["pnl"] for trade in trades_train)
            roi_train = total_pnl_train / 10000 * 100

            # 테스트 구간 백테스트
            backtester.load_data(
                "ohlcv_{symbol}_{timeframe}",
                "ohlcv_{symbol}_{timeframe}",
                "4h", "1d",
                split["test_start"], split["test_end"]
            )
            try:
                trades_test, _ = backtester.run_backtest(dynamic_params)
            except Exception as e:
                logger.error(f"Test backtest failed: {e}")
                return 1e6
            total_pnl_test = sum(trade["pnl"] for trade in trades_test)
            roi_test = total_pnl_test / 10000 * 100

            # 과적합 패널티 및 점수 산출: 테스트 ROI와 학습 ROI의 차이를 고려
            overfit_penalty = abs(roi_train - roi_test)
            score = -roi_test + 0.5 * overfit_penalty
            total_score += score

        avg_score = total_score / len(splits)
        return avg_score

    def optimize(self):
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        trials_df = self.study.trials_dataframe()
        logger.info("Trial 결과:\n" + trials_df.to_string())
        
        return self.study.best_trial
