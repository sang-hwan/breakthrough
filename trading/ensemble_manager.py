# trading/ensemble_manager.py
from logs.logger_config import setup_logger
from trading.strategies import TradingStrategies

class EnsembleManager:
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        # 각 전략별 가중치 (필요 시 동적 조정)
        self.strategy_weights = {
            "base": 1.0,
            "trend_following": 1.0,
            "breakout": 1.0,
            "counter_trend": 1.0,
            "high_frequency": 1.0
        }
        self.strategy_manager = TradingStrategies()
        # 최종 신호의 마지막 값을 저장 (신호 변경 감지용)
        self.last_final_signal = None

    def get_final_signal(self, market_regime, liquidity_info, data, current_time):
        # 각 전략의 원시 신호 산출 (세부 정보는 DEBUG 레벨로 기록)
        signals = {
            "base": self.strategy_manager.select_strategy(market_regime, liquidity_info, data, current_time),
            "trend_following": self.strategy_manager.trend_following_strategy(data, current_time),
            "breakout": self.strategy_manager.breakout_strategy(data, current_time),
            "counter_trend": self.strategy_manager.counter_trend_strategy(data, current_time),
            "high_frequency": self.strategy_manager.high_frequency_strategy(data, current_time)
        }
        self.logger.debug(f"각 전략 원시 신호: {signals}")

        # 가중치 기반 투표: 'enter_long'와 'exit_all' 신호의 가중치 합산
        vote_enter = sum(self.strategy_weights.get(k, 1.0) for k, sig in signals.items() if sig == "enter_long")
        vote_exit  = sum(self.strategy_weights.get(k, 1.0) for k, sig in signals.items() if sig == "exit_all")
        
        if vote_exit > vote_enter:
            final_signal = "exit_all"
        elif vote_enter > vote_exit:
            final_signal = "enter_long"
        else:
            final_signal = "hold"

        # 신호 변경 시 INFO 레벨로 기록 (전역 AggregatingHandler가 집계됨)
        if self.last_final_signal != final_signal:
            self.logger.info(f"신호 변경: 이전 신호={self.last_final_signal}, 새로운 신호={final_signal} at {current_time}")
            self.last_final_signal = final_signal
        else:
            self.logger.debug(f"신호 유지: '{final_signal}' at {current_time}")

        return final_signal

    def update_strategy_weights(self, performance_metrics):
        """
        실시간 성과 지표에 따라 각 전략의 가중치를 조정합니다.
        """
        for strat, perf in performance_metrics.items():
            if perf < 0:
                self.strategy_weights[strat] *= 0.95
            else:
                self.strategy_weights[strat] *= 1.05
        self.logger.info(f"전략 가중치 업데이트: {self.strategy_weights}")
