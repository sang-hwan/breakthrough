# trading/ensemble_manager.py
from logs.logger_config import setup_logger
from trading.strategies import TradingStrategies

class EnsembleManager:
    def __init__(self):
        self.logger = setup_logger(__name__)
        # 초기 가중치는 동일하게 설정 (추후 실제 성과 기반 업데이트 가능)
        self.strategy_weights = {
            "base": 1.0,
            "trend_following": 1.0,
            "breakout": 1.0,
            "counter_trend": 1.0,
            "high_frequency": 1.0
        }
        self.strategy_manager = TradingStrategies()
    
    def get_final_signal(self, market_regime, liquidity_info, data, current_time):
        """
        여러 전략의 신호를 개별 호출한 후, 가중치 기반 투표로 최종 거래 신호를 결정하여 반환합니다.
        """
        signals = {}
        signals["base"] = self.strategy_manager.select_strategy(market_regime, liquidity_info, data, current_time)
        signals["trend_following"] = self.strategy_manager.trend_following_strategy(data, current_time)
        signals["breakout"] = self.strategy_manager.breakout_strategy(data, current_time)
        signals["counter_trend"] = self.strategy_manager.counter_trend_strategy(data, current_time)
        signals["high_frequency"] = self.strategy_manager.high_frequency_strategy(data, current_time)
        
        vote_enter = 0.0
        vote_exit = 0.0
        for key, signal in signals.items():
            weight = self.strategy_weights.get(key, 1.0)
            if signal == "enter_long":
                vote_enter += weight
            elif signal == "exit_all":
                vote_exit += weight
        
        if vote_exit > vote_enter:
            final_signal = "exit_all"
        elif vote_enter > vote_exit:
            final_signal = "enter_long"
        else:
            final_signal = "hold"
        
        self.logger.info(f"EnsembleManager final signal at {current_time}: {final_signal}, details: {signals}")
        return final_signal

    def update_strategy_weights(self, performance_metrics):
        """
        실시간 성과 지표(performance_metrics)에 따라 각 전략의 가중치를 조정합니다.
        (구현 예시 – 상세 로직은 실제 피드백 루프에 맞게 확장)
        """
        for strat, perf in performance_metrics.items():
            if perf < 0:
                self.strategy_weights[strat] *= 0.95
            else:
                self.strategy_weights[strat] *= 1.05
        self.logger.info(f"Updated strategy weights: {self.strategy_weights}")
