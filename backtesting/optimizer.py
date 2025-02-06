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
        # 기본 파라미터 및 최적화 대상 파라미터 제안
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
        dynamic_params = {**base_params, **params}
        
        assets = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        
        # 워크-포워드 검증: training 및 test split
        splits = [
            {
                "train_start": "2018-06-01",
                "train_end":   "2020-12-31",
                "test_start":  "2021-01-01",
                "test_end":    "2023-12-31"
            }
        ]
        # 홀드아웃 구간 (최종 검증 전용)
        holdout = {"holdout_start": "2024-01-01", "holdout_end": "2025-02-01"}
        
        total_score = 0.0
        num_evaluations = 0
        for split in splits:
            for asset in assets:
                # Training backtest
                backtester_train = Backtester(symbol=asset, account_size=10000)
                backtester_train.load_data(
                    "ohlcv_{symbol}_{timeframe}",
                    "ohlcv_{symbol}_{timeframe}",
                    "4h", "1d",
                    split["train_start"], split["train_end"]
                )
                try:
                    trades_train, _ = backtester_train.run_backtest(dynamic_params)
                except Exception as e:
                    logger.error(f"Training backtest failed for {asset} on split {split}: {e}")
                    return 1e6
                total_pnl_train = sum(trade["pnl"] for trade in trades_train)
                roi_train = total_pnl_train / 10000 * 100

                # Test backtest
                backtester_test = Backtester(symbol=asset, account_size=10000)
                backtester_test.load_data(
                    "ohlcv_{symbol}_{timeframe}",
                    "ohlcv_{symbol}_{timeframe}",
                    "4h", "1d",
                    split["test_start"], split["test_end"]
                )
                try:
                    trades_test, _ = backtester_test.run_backtest(dynamic_params)
                except Exception as e:
                    logger.error(f"Test backtest failed for {asset} on split {split}: {e}")
                    return 1e6
                total_pnl_test = sum(trade["pnl"] for trade in trades_test)
                roi_test = total_pnl_test / 10000 * 100

                # 홀드아웃 구간 backtest
                backtester_holdout = Backtester(symbol=asset, account_size=10000)
                backtester_holdout.load_data(
                    "ohlcv_{symbol}_{timeframe}",
                    "ohlcv_{symbol}_{timeframe}",
                    "4h", "1d",
                    holdout["holdout_start"], holdout["holdout_end"]
                )
                try:
                    trades_holdout, _ = backtester_holdout.run_backtest(dynamic_params)
                except Exception as e:
                    logger.error(f"Holdout backtest failed for {asset}: {e}")
                    return 1e6
                total_pnl_holdout = sum(trade["pnl"] for trade in trades_holdout)
                roi_holdout = total_pnl_holdout / 10000 * 100

                overfit_penalty = abs(roi_train - roi_test)
                # 홀드아웃 성과가 월간 ROI 2% 미만이면 페널티 부과
                holdout_penalty = 0 if roi_holdout >= 2.0 else (2.0 - roi_holdout) * 10

                score = -roi_test + overfit_penalty + holdout_penalty
                total_score += score
                num_evaluations += 1
        
        avg_score = total_score / num_evaluations if num_evaluations > 0 else total_score
        
        # 정규화 항 (Regularization penalty)
        reg_penalty = 0.0
        regularization_keys = ["atr_multiplier", "profit_ratio", "risk_per_trade", "scale_in_threshold"]
        for key in regularization_keys:
            default_value = base_params.get(key, 1.0)
            diff = dynamic_params.get(key, default_value) - default_value
            reg_penalty += (diff ** 2)
        reg_penalty *= 0.1
        
        final_score = avg_score + reg_penalty
        return final_score
  
    def optimize(self):
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        trials_df = self.study.trials_dataframe()
        logger.info("Trial 결과:\n" + trials_df.to_string())
        
        return self.study.best_trial
