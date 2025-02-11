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
            "high_frequency": 1.0,
            "weekly_breakout": 1.0,
            "weekly_momentum": 1.0
        }
        self.strategy_manager = TradingStrategies()
        # 최종 신호의 마지막 값을 저장 (신호 변경 감지용)
        self.last_final_signal = None

    def get_final_signal(self, market_regime, liquidity_info, data, current_time, data_weekly=None):
        """
        단기 전략 신호와 주간 전략 신호를 가중치 기반으로 종합하여 최종 거래 신호를 산출합니다.
        
        - data: 단기 데이터 (예: 4h 캔들)
        - data_weekly: 주간 데이터 (예: 주간 캔들; 없으면 단기 전략만 반영)
        """
        # 단기 전략 신호 산출
        signals = {
            "base": self.strategy_manager.select_strategy(market_regime, liquidity_info, data, current_time),
            "trend_following": self.strategy_manager.trend_following_strategy(data, current_time),
            "breakout": self.strategy_manager.breakout_strategy(data, current_time),
            "counter_trend": self.strategy_manager.counter_trend_strategy(data, current_time),
            "high_frequency": self.strategy_manager.high_frequency_strategy(data, current_time)
        }
        # 주간 데이터가 제공되면 주간 전략 신호 추가
        if data_weekly is not None:
            signals["weekly_breakout"] = self.strategy_manager.weekly_breakout_strategy(data_weekly, current_time)
            signals["weekly_momentum"] = self.strategy_manager.weekly_momentum_strategy(data_weekly, current_time)
        
        self.logger.debug(f"각 전략 원시 신호: {signals}")
        
        # 단기와 주간 신호의 가중치 (예: 단기 0.7, 주간 0.3)
        short_term_weight = 0.7
        weekly_weight = 0.3 if data_weekly is not None else 0.0
        
        vote_enter = 0.0
        vote_exit  = 0.0
        
        # 단기 전략 투표
        for key in ["base", "trend_following", "breakout", "counter_trend", "high_frequency"]:
            sig = signals.get(key)
            if sig == "enter_long":
                vote_enter += short_term_weight * self.strategy_weights.get(key, 1.0)
            elif sig == "exit_all":
                vote_exit += short_term_weight * self.strategy_weights.get(key, 1.0)
        
        # 주간 전략 투표
        for key in ["weekly_breakout", "weekly_momentum"]:
            sig = signals.get(key)
            if sig == "enter_long":
                vote_enter += weekly_weight * self.strategy_weights.get(key, 1.0)
            elif sig == "exit_all":
                vote_exit += weekly_weight * self.strategy_weights.get(key, 1.0)
        
        if vote_exit > vote_enter:
            final_signal = "exit_all"
        elif vote_enter > vote_exit:
            final_signal = "enter_long"
        else:
            final_signal = "hold"
        
        if self.last_final_signal != final_signal:
            self.logger.info(f"신호 변경: 이전 신호={self.last_final_signal}, 새로운 신호={final_signal} at {current_time}")
            self.last_final_signal = final_signal
        else:
            # 신호 유지 로그의 레벨을 INFO로 승격하여 기록
            self.logger.info(f"신호 유지: '{final_signal}' at {current_time}")
        
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
