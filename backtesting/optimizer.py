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
        try:
            base_params = self.dynamic_param_manager.get_default_params()
            suggested_params = {
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
            dynamic_params = {**base_params, **suggested_params}
            logger.info(f"[Optimizer] 병합된 파라미터: {dynamic_params}")

            assets = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
            splits = [
                {
                    "train_start": "2018-06-01",
                    "train_end": "2020-12-31",
                    "test_start": "2021-01-01",
                    "test_end": "2023-12-31"
                }
            ]
            holdout = {"holdout_start": "2024-01-01", "holdout_end": "2025-02-01"}

            total_score = 0.0
            num_evaluations = 0

            for split in splits:
                for asset in assets:
                    logger.info(f"[Optimizer] {asset} 평가, 스플릿: {split}")
                    
                    # Training 백테스트 수행
                    backtester_train = Backtester(symbol=asset, account_size=10000)
                    backtester_train.load_data(
                        short_table_format="ohlcv_{symbol}_{timeframe}",
                        long_table_format="ohlcv_{symbol}_{timeframe}",
                        short_tf="4h", long_tf="1d",
                        start_date=split["train_start"], end_date=split["train_end"]
                    )
                    try:
                        trades_train, _ = backtester_train.run_backtest(dynamic_params)
                    except Exception as e:
                        logger.error(f"[Optimizer] Training backtest 실패: {asset}, 스플릿 {split}: {e}", exc_info=True)
                        return 1e6
                    total_pnl_train = sum(trade["pnl"] for trade in trades_train)
                    roi_train = total_pnl_train / 10000 * 100
                    logger.info(f"[Optimizer] {asset} Training ROI: {roi_train:.2f}%")

                    # Test 백테스트 수행
                    backtester_test = Backtester(symbol=asset, account_size=10000)
                    backtester_test.load_data(
                        short_table_format="ohlcv_{symbol}_{timeframe}",
                        long_table_format="ohlcv_{symbol}_{timeframe}",
                        short_tf="4h", long_tf="1d",
                        start_date=split["test_start"], end_date=split["test_end"]
                    )
                    try:
                        trades_test, _ = backtester_test.run_backtest(dynamic_params)
                    except Exception as e:
                        logger.error(f"[Optimizer] Test backtest 실패: {asset}, 스플릿 {split}: {e}", exc_info=True)
                        return 1e6
                    total_pnl_test = sum(trade["pnl"] for trade in trades_test)
                    roi_test = total_pnl_test / 10000 * 100
                    logger.info(f"[Optimizer] {asset} Test ROI: {roi_test:.2f}%")

                    # Holdout 백테스트 수행
                    backtester_holdout = Backtester(symbol=asset, account_size=10000)
                    backtester_holdout.load_data(
                        short_table_format="ohlcv_{symbol}_{timeframe}",
                        long_table_format="ohlcv_{symbol}_{timeframe}",
                        short_tf="4h", long_tf="1d",
                        start_date=holdout["holdout_start"], end_date=holdout["holdout_end"]
                    )
                    try:
                        trades_holdout, _ = backtester_holdout.run_backtest(dynamic_params)
                    except Exception as e:
                        logger.error(f"[Optimizer] Holdout backtest 실패: {asset}: {e}", exc_info=True)
                        return 1e6
                    total_pnl_holdout = sum(trade["pnl"] for trade in trades_holdout)
                    roi_holdout = total_pnl_holdout / 10000 * 100
                    logger.info(f"[Optimizer] {asset} Holdout ROI: {roi_holdout:.2f}%")

                    # 평가 점수 계산 (Overfit, Holdout 페널티 포함)
                    overfit_penalty = abs(roi_train - roi_test)
                    holdout_penalty = 0 if roi_holdout >= 2.0 else (2.0 - roi_holdout) * 10
                    score = -roi_test + overfit_penalty + holdout_penalty
                    logger.info(f"[Optimizer] {asset} Score: {score:.2f} (Overfit: {overfit_penalty:.2f}, Holdout: {holdout_penalty:.2f})")
                    
                    total_score += score
                    num_evaluations += 1

            avg_score = total_score / num_evaluations if num_evaluations > 0 else total_score

            # 정규화 패널티 계산
            reg_penalty = 0.0
            regularization_keys = ["atr_multiplier", "profit_ratio", "risk_per_trade", "scale_in_threshold"]
            for key in regularization_keys:
                default_value = base_params.get(key, 1.0)
                diff = dynamic_params.get(key, default_value) - default_value
                reg_penalty += (diff ** 2)
            reg_penalty *= 0.1

            final_score = avg_score + reg_penalty
            logger.info(f"[Optimizer] 최종 트라이얼 점수: {final_score:.2f} (Avg: {avg_score:.2f}, Reg: {reg_penalty:.2f})")
            return final_score

        except Exception as e:
            logger.error(f"[Optimizer] Objective 함수 에러: {e}", exc_info=True)
            return 1e6

    def optimize(self):
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        logger.info(f"[Optimizer] {self.n_trials} 트라이얼로 최적화 시작.")
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        trials_df = self.study.trials_dataframe()
        # INFO 레벨 로그로 남겨 AggregatingHandler 가 집계하도록 함
        logger.info(f"[Optimizer] 트라이얼 결과:\n{trials_df.to_string()}")
        
        best_trial = self.study.best_trial
        logger.info(f"[Optimizer] Best trial: {best_trial.number} (Value: {best_trial.value:.2f})")
        logger.info(f"[Optimizer] Best parameters: {best_trial.params}")
        return best_trial
