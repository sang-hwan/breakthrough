# trading/ensemble.py
from logs.logger_config import setup_logger
from strategies.trading_strategies import (
    SelectStrategy, TrendFollowingStrategy, BreakoutStrategy,
    CounterTrendStrategy, HighFrequencyStrategy, WeeklyBreakoutStrategy, WeeklyMomentumStrategy
)
from config.config_manager import ConfigManager

def compute_dynamic_weights(market_volatility: float, liquidity_info: str, volume: float = None):
    if market_volatility is None:
        market_volatility = 0.02

    config = ConfigManager().get_dynamic_params()
    if liquidity_info.lower() == "high":
        short_weight = config.get("liquidity_weight_high", 0.8)
        weekly_weight = 1 - short_weight
    else:
        short_weight = config.get("liquidity_weight_low", 0.6)
        weekly_weight = 1 - short_weight

    vol_threshold = config.get("weight_vol_threshold", 0.05)
    if market_volatility > vol_threshold:
        factor = config.get("vol_weight_factor", 0.9)
        short_weight *= factor
        weekly_weight = 1 - short_weight

    if volume is not None and volume < 1000:
        short_weight *= 0.8
        weekly_weight = 1 - short_weight

    return short_weight, weekly_weight

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

    def get_final_signal(self, market_regime, liquidity_info, data, current_time, data_weekly=None, 
                         market_volatility: float = None, volume: float = None):
        short_weight, weekly_weight = compute_dynamic_weights(market_volatility, liquidity_info, volume)

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

        vote_enter = sum(short_weight for key in ["select", "trend", "breakout", "counter", "hf"] if signals.get(key) == "enter_long")
        vote_exit = sum(short_weight for key in ["select", "trend", "breakout", "counter", "hf"] if signals.get(key) == "exit_all")
        if data_weekly is not None:
            vote_enter += sum(weekly_weight for key in ["weekly_breakout", "weekly_momentum"] if signals.get(key) == "enter_long")
            vote_exit += sum(weekly_weight for key in ["weekly_breakout", "weekly_momentum"] if signals.get(key) == "exit_all")

        final_signal = "exit_all" if vote_exit > vote_enter else ("enter_long" if vote_enter > vote_exit else "hold")
        if self.last_final_signal != final_signal:
            self.logger.debug(
                f"Ensemble final signal changed to {final_signal} at {current_time} "
                f"with dynamic weights: short={short_weight}, weekly={weekly_weight}, "
                f"signals: {signals}"
            )
            self.last_final_signal = final_signal
        return final_signal
