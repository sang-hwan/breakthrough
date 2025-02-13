# trading/ensemble.py
from logs.logger_config import setup_logger
from strategies.trading_strategies import (
    SelectStrategy, TrendFollowingStrategy, BreakoutStrategy,
    CounterTrendStrategy, HighFrequencyStrategy, WeeklyBreakoutStrategy, WeeklyMomentumStrategy
)

class Ensemble:
    """
    다양한 전략의 신호를 종합하는 클래스.
    신호별 가중치는 단기 0.7, 주간 0.3으로 적용하며, 투표 결과에 따라 최종 신호를 산출합니다.
    """
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.select_strategy = SelectStrategy()
        self.trend_following_strategy = TrendFollowingStrategy()
        self.breakout_strategy = BreakoutStrategy()
        self.counter_trend_strategy = CounterTrendStrategy()
        self.high_frequency_strategy = HighFrequencyStrategy()
        self.weekly_breakout_strategy = WeeklyBreakoutStrategy()
        self.weekly_momentum_strategy = WeeklyMomentumStrategy()
        self.last_final_signal = None

    def get_final_signal(self, market_regime, liquidity_info, data, current_time, data_weekly=None):
        # 단기 전략 신호 수집
        signals = {
            "select": self.select_strategy.get_signal(data, current_time),
            "trend_following": self.trend_following_strategy.get_signal(data, current_time),
            "breakout": self.breakout_strategy.get_signal(data, current_time),
            "counter_trend": self.counter_trend_strategy.get_signal(data, current_time),
            "high_frequency": self.high_frequency_strategy.get_signal(data, current_time)
        }
        # 주간 전략 신호 수집 (데이터 제공 시)
        if data_weekly is not None:
            signals["weekly_breakout"] = self.weekly_breakout_strategy.get_signal(data_weekly, current_time)
            signals["weekly_momentum"] = self.weekly_momentum_strategy.get_signal(data_weekly, current_time)
        self.logger.debug(f"개별 전략 원시 신호: {signals}")

        # 가중치 적용: 단기 0.7, 주간 0.3
        short_weight = 0.7
        weekly_weight = 0.3 if data_weekly is not None else 0.0

        vote_enter = 0.0
        vote_exit = 0.0
        for key in ["select", "trend_following", "breakout", "counter_trend", "high_frequency"]:
            if signals.get(key) == "enter_long":
                vote_enter += short_weight
            elif signals.get(key) == "exit_all":
                vote_exit += short_weight
        for key in ["weekly_breakout", "weekly_momentum"]:
            if signals.get(key) == "enter_long":
                vote_enter += weekly_weight
            elif signals.get(key) == "exit_all":
                vote_exit += weekly_weight

        if vote_exit > vote_enter:
            final_signal = "exit_all"
        elif vote_enter > vote_exit:
            final_signal = "enter_long"
        else:
            final_signal = "hold"

        self.logger.debug(f"최종 종합 신호: {final_signal} (vote_enter={vote_enter}, vote_exit={vote_exit})")
        if self.last_final_signal != final_signal:
            self.logger.debug(f"신호 변경: 이전 신호={self.last_final_signal}, 새로운 신호={final_signal}")
            self.last_final_signal = final_signal
        return final_signal
