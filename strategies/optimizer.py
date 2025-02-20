# strategies/optimizer.py
import optuna
from logs.logger_config import setup_logger
from backtesting.backtester import Backtester
from config.config_manager import ConfigManager, TradingConfig

logger = setup_logger(__name__)

class DynamicParameterOptimizer:
    def __init__(self, n_trials=10):
        self.n_trials = n_trials
        self.study = None
        self.config_manager = ConfigManager()

    def objective(self, trial):
        try:
            base_params = self.config_manager.get_defaults()
            suggested_params = {
                "hmm_confidence_threshold": trial.suggest_float("hmm_confidence_threshold", 0.7, 0.95),
                "liquidity_info": trial.suggest_categorical("liquidity_info", ["high", "low"]),
                "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 3.0),
                "profit_ratio": trial.suggest_float("profit_ratio", 0.05, 0.15),
                "risk_per_trade": trial.suggest_float("risk_per_trade", 0.005, 0.02),
                "scale_in_threshold": trial.suggest_float("scale_in_threshold", 0.01, 0.03),
                "partial_exit_ratio": trial.suggest_float("partial_exit_ratio", 0.4, 0.6),
                "partial_profit_ratio": trial.suggest_float("partial_profit_ratio", 0.02, 0.04),
                "final_profit_ratio": trial.suggest_float("final_profit_ratio", 0.05, 0.1),
                "weekly_breakout_threshold": trial.suggest_float("weekly_breakout_threshold", 0.005, 0.02),
                "weekly_momentum_threshold": trial.suggest_float("weekly_momentum_threshold", 0.3, 0.7)
            }
            dynamic_params = {**base_params, **suggested_params}
            # Validate dynamic parameters
            dynamic_params = self.config_manager.validate_params(dynamic_params)

            try:
                _ = TradingConfig(**dynamic_params)
            except Exception as e:
                logger.error("Validation error in dynamic_params: " + str(e), exc_info=True)
                return 1e6

            assets = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
            splits = [{
                "train_start": "2018-06-01",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2023-12-31"
            }]
            holdout = {"holdout_start": "2024-01-01", "holdout_end": "2025-02-01"}

            total_score, num_evaluations = 0.0, 0
            for split in splits:
                for asset in assets:
                    symbol_key = asset.replace("/", "").lower()

                    bt_train = Backtester(symbol=asset, account_size=10000)
                    bt_train.load_data(
                        short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        short_tf="4h", long_tf="1d",
                        start_date=split["train_start"], end_date=split["train_end"]
                    )
                    try:
                        trades_train, _ = bt_train.run_backtest(dynamic_params)
                    except Exception as e:
                        logger.error("Backtest train error for " + asset + ": " + str(e), exc_info=True)
                        return 1e6
                    roi_train = sum(trade["pnl"] for trade in trades_train) / 10000 * 100

                    bt_test = Backtester(symbol=asset, account_size=10000)
                    bt_test.load_data(
                        short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        short_tf="4h", long_tf="1d",
                        start_date=split["test_start"], end_date=split["test_end"]
                    )
                    try:
                        trades_test, _ = bt_test.run_backtest(dynamic_params)
                    except Exception as e:
                        logger.error("Backtest test error for " + asset + ": " + str(e), exc_info=True)
                        return 1e6
                    roi_test = sum(trade["pnl"] for trade in trades_test) / 10000 * 100

                    bt_holdout = Backtester(symbol=asset, account_size=10000)
                    bt_holdout.load_data(
                        short_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        long_table_format=f"ohlcv_{symbol_key}_{{timeframe}}",
                        short_tf="4h", long_tf="1d",
                        start_date=holdout["holdout_start"], end_date=holdout["holdout_end"]
                    )
                    try:
                        trades_holdout, _ = bt_holdout.run_backtest(dynamic_params)
                    except Exception as e:
                        logger.error("Backtest holdout error for " + asset + ": " + str(e), exc_info=True)
                        return 1e6
                    roi_holdout = sum(trade["pnl"] for trade in trades_holdout) / 10000 * 100

                    overfit_penalty = abs(roi_train - roi_test)
                    holdout_penalty = 0 if roi_holdout >= 2.0 else (2.0 - roi_holdout) * 10
                    score = -roi_test + overfit_penalty + holdout_penalty
                    total_score += score
                    num_evaluations += 1

            avg_score = total_score / num_evaluations if num_evaluations else total_score
            reg_penalty = 0.1 * sum(
                (dynamic_params.get(key, base_params.get(key, 1.0)) - base_params.get(key, 1.0)) ** 2
                for key in ["atr_multiplier", "profit_ratio", "risk_per_trade", "scale_in_threshold",
                            "weekly_breakout_threshold", "weekly_momentum_threshold"]
            )
            return avg_score + reg_penalty

        except Exception as e:
            logger.error("Unexpected error in objective: " + str(e), exc_info=True)
            return 1e6

    def optimize(self):
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        best_trial = self.study.best_trial
        logger.info(f"Best trial: {best_trial.number}, Value: {best_trial.value:.2f}")
        logger.info(f"Best parameters: {best_trial.params}")
        return best_trial
