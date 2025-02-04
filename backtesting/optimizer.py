# backtesting/optimizer.py
import optuna
import pandas as pd
from backtesting.backtester import Backtester
from dynamic_parameters.dynamic_param_manager import DynamicParamManager

class DynamicParameterOptimizer:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.study = None
        self.dynamic_param_manager = DynamicParamManager()
    
    def objective(self, trial):
        # 기본 파라미터에 trial 제안 값 적용
        params = self.dynamic_param_manager.get_default_params()
        params["lookback_window"] = trial.suggest_int("lookback_window", 10, 30)
        params["volume_factor"] = trial.suggest_float("volume_factor", 1.2, 2.0)
        params["confirmation_bars"] = trial.suggest_int("confirmation_bars", 1, 3)
        params["breakout_buffer"] = trial.suggest_float("breakout_buffer", 0.0, 0.005)
        params["atr_multiplier"] = trial.suggest_float("atr_multiplier", 1.5, 3.0)
        params["profit_ratio"] = trial.suggest_float("profit_ratio", 0.05, 0.15)
        params["risk_per_trade"] = trial.suggest_float("risk_per_trade", 0.005, 0.02)
        params["scale_in_threshold"] = trial.suggest_float("scale_in_threshold", 0.01, 0.03)
        params["use_trend_exit"] = trial.suggest_categorical("use_trend_exit", [True, False])
        params["use_partial_take_profit"] = trial.suggest_categorical("use_partial_take_profit", [True, False])
        params["entry_signal_mode"] = trial.suggest_categorical("entry_signal_mode", ["AND", "OR"])
        
        # 워크-포워드 테스트: 데이터의 70%는 학습용, 30%는 검증용으로 분리
        backtester = Backtester(symbol="BTC/USDT", account_size=10000)
        # 예시 기간 (2018-01-01 ~ 2020-01-01)
        backtester.load_data(
            "ohlcv_{symbol}_{timeframe}",
            "ohlcv_{symbol}_{timeframe}",
            "4h",
            "1d",
            "2018-01-01",
            "2020-01-01"
        )
        
        total_index = backtester.df_short.index
        split_point = int(len(total_index) * 0.7)
        train_index = total_index[:split_point]
        
        # 학습(Train) 구간 백테스트
        backtester.df_short = backtester.df_short.loc[train_index]
        backtester.df_long = backtester.df_long.reindex(train_index, method='ffill')
        market_data_train = {"volatility": 0.06, "trend_strength": 0.4}
        dynamic_params_train = self.dynamic_param_manager.update_dynamic_params(market_data_train)
        dynamic_params_train.update(params)
        
        try:
            trades_train, _ = backtester.run_backtest(dynamic_params_train)
        except Exception:
            return 1e6
        
        if not trades_train:
            return 1e6
        
        total_pnl_train = sum(trade["pnl"] for trade in trades_train)
        final_balance_train = 10000 + total_pnl_train
        roi_train = (final_balance_train - 10000) / 10000 * 100
        
        # 검증(Validation) 구간 백테스트
        val_start = total_index[split_point].strftime("%Y-%m-%d")
        backtester.load_data(
            "ohlcv_{symbol}_{timeframe}",
            "ohlcv_{symbol}_{timeframe}",
            "4h",
            "1d",
            val_start,
            "2020-01-01"
        )
        market_data_val = {"volatility": 0.07, "trend_strength": 0.5}
        dynamic_params_val = self.dynamic_param_manager.update_dynamic_params(market_data_val)
        dynamic_params_val.update(params)
        
        try:
            trades_val, _ = backtester.run_backtest(dynamic_params_val)
        except Exception:
            return 1e6

        if not trades_val:
            return 1e6
        
        total_pnl_val = sum(trade["pnl"] for trade in trades_val)
        final_balance_val = 10000 + total_pnl_val
        roi_val = (final_balance_val - 10000) / 10000 * 100
        
        # 학습과 검증 간의 차이에 패널티를 줘서 오버피팅 방지
        overfit_penalty = abs(roi_train - roi_val)
        
        # 목표: 최적화 방향이 "minimize" 이므로, 음의 방향 스코어(ROI 낮을수록, 차이 클수록 패널티)
        score = -roi_val + 0.5 * overfit_penalty
        return score

    def optimize(self):
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Trial 결과를 DataFrame으로 저장
        trials_df = self.study.trials_dataframe()
        trials_df.to_csv("optimizer_trials.csv", index=False)
        print("Trial 결과가 optimizer_trials.csv에 저장되었습니다.")
        
        return self.study.best_trial