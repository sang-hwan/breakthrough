# trading/ensemble.py
from logs.logger_config import setup_logger
from strategies.trading_strategies import (
    SelectStrategy, TrendFollowingStrategy, BreakoutStrategy,
    CounterTrendStrategy, HighFrequencyStrategy, WeeklyBreakoutStrategy, WeeklyMomentumStrategy
)

class Ensemble:
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
        signals = {
            "select": self.select_strategy.get_signal(data, current_time),
            "trend": self.trend_following_strategy.get_signal(data, current_time),
            "breakout": self.breakout_strategy.get_signal(data, current_time),
            "counter": self.counter_trend_strategy.get_signal(data, current_time),
            "hf": self.high_frequency_strategy.get_signal(data, current_time)
        }
        if data_weekly is not None:
            signals["weekly_breakout"] = self.weekly_breakout_strategy.get_signal(data_weekly, current_time)
            signals["weekly_momentum"] = self.weekly_momentum_strategy.get_signal(data_weekly, current_time)

        short_weight = 0.7
        weekly_weight = 0.3 if data_weekly is not None else 0.0

        vote_enter = sum(short_weight for key in ["select", "trend", "breakout", "counter", "hf"] if signals.get(key) == "enter_long")
        vote_exit = sum(short_weight for key in ["select", "trend", "breakout", "counter", "hf"] if signals.get(key) == "exit_all")
        if data_weekly is not None:
            vote_enter += sum(weekly_weight for key in ["weekly_breakout", "weekly_momentum"] if signals.get(key) == "enter_long")
            vote_exit += sum(weekly_weight for key in ["weekly_breakout", "weekly_momentum"] if signals.get(key) == "exit_all")

        final_signal = "exit_all" if vote_exit > vote_enter else ("enter_long" if vote_enter > vote_exit else "hold")
        if self.last_final_signal != final_signal:
            self.logger.debug(f"Ensemble final signal changed to {final_signal} at {current_time}")
            self.last_final_signal = final_signal
        return final_signal
