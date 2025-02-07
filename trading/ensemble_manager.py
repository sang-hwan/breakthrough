# trading/ensemble_manager.py
from logs.logger_config import setup_logger
from trading.strategies import TradingStrategies
from datetime import timedelta

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
        # 최소 신호 전환 간격
        self.min_signal_interval = timedelta(minutes=60)
        self.last_signal_time = None
        self.last_final_signal = "hold"
        self.last_signals = None

    def get_final_signal(self, market_regime, liquidity_info, data, current_time):
        """
        여러 전략의 신호를 개별로 산출한 후, 가중치 기반 투표 및 추가 필터링을 통해 최종 거래 신호를 결정합니다.
        최소 신호 간격을 적용하여 잦은 신호 변경을 억제합니다.
        """
        signals = {
            "base": self.strategy_manager.select_strategy(market_regime, liquidity_info, data, current_time),
            "trend_following": self.strategy_manager.trend_following_strategy(data, current_time),
            "breakout": self.strategy_manager.breakout_strategy(data, current_time),
            "counter_trend": self.strategy_manager.counter_trend_strategy(data, current_time),
            "high_frequency": self.strategy_manager.high_frequency_strategy(data, current_time)
        }

        self.logger.debug(f"EnsembleManager 원시 신호 ({current_time}): {signals}")

        vote_enter = sum(self.strategy_weights.get(k, 1.0) for k, sig in signals.items() if sig == "enter_long")
        vote_exit  = sum(self.strategy_weights.get(k, 1.0) for k, sig in signals.items() if sig == "exit_all")
        
        if vote_exit > vote_enter:
            final_signal = "exit_all"
        elif vote_enter > vote_exit:
            final_signal = "enter_long"
        else:
            final_signal = "hold"
        
        # 최소 신호 간격 적용
        if self.last_signal_time is not None:
            elapsed = current_time - self.last_signal_time
            if elapsed < self.min_signal_interval:
                self.logger.debug(f"신호 전환 억제: 마지막 신호 후 {elapsed.total_seconds()/60:.2f}분 경과.")
                final_signal = self.last_final_signal
            else:
                self.last_signal_time = current_time
                self.last_final_signal = final_signal
        else:
            if final_signal != "hold":
                self.last_signal_time = current_time
                self.last_final_signal = final_signal

        if self.last_signals == signals:
            self.logger.debug(f"이전 신호와 동일: 최종 신호 {final_signal} 유지.")
        else:
            self.last_signals = signals.copy()
            self.logger.debug(f"신호 업데이트: {signals}")

        self.logger.info(f"EnsembleManager 최종 신호 ({current_time}): {final_signal}")
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
