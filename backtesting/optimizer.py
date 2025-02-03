# backtesting/optimizer.py
import optuna
from backtesting.backtester import Backtester
from dynamic_parameters.dynamic_param_manager import DynamicParamManager

class DynamicParameterOptimizer:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.study = None
        self.dynamic_param_manager = DynamicParamManager()
    
    def objective(self, trial):
        # trial를 통해 최적화할 파라미터 범위 지정 (예시)
        params = self.dynamic_param_manager.get_default_params()
        params["lookback_window"] = trial.suggest_int("lookback_window", 10, 30)
        params["volume_factor"] = trial.suggest_float("volume_factor", 1.2, 2.0)
        params["confirmation_bars"] = trial.suggest_int("confirmation_bars", 1, 3)
        params["breakout_buffer"] = trial.suggest_float("breakout_buffer", 0.0, 0.005)
        # 추가 파라미터도 trial.suggest_... 로 지정 가능
        
        # 백테스터 생성 및 데이터 로드 (예시: 2018~2020)
        backtester = Backtester(symbol="BTC/USDT", account_size=10000)
        backtester.load_data("ohlcv_{symbol}_{timeframe}", "ohlcv_{symbol}_{timeframe}", "4h", "1d", "2018-01-01", "2020-01-01")
        
        # 예시 시장 데이터 (동적 파라미터 조정을 위한 값)
        market_data = {"volatility": 0.06, "trend_strength": 0.4}
        dynamic_params = self.dynamic_param_manager.update_dynamic_params(market_data)
        dynamic_params.update(params)  # trial에서 제시한 값 적용
        
        try:
            trades, _ = backtester.run_backtest(dynamic_params)
        except Exception:
            return 1e6  # 실패 시 큰 페널티
        
        if not trades:
            return 1e6
        total_pnl = sum(trade["pnl"] for trade in trades)
        final_balance = 10000 + total_pnl
        roi = (final_balance - 10000) / 10000 * 100
        return -roi  # 최소화를 위해 음수 ROI 반환

    def optimize(self):
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return self.study.best_trial
