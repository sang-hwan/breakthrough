# strategies/trading_strategies.py
from logs.logger_config import setup_logger
from strategies.base_strategy import BaseStrategy

class SelectStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def _get_candle_pattern_signal(self, row):
        open_price = row.get('open')
        close_price = row.get('close')
        if open_price is None or close_price is None:
            return None
        return "bullish" if close_price > open_price * 1.005 else ("bearish" if close_price < open_price * 0.99 else None)

    def _get_sma_rsi_signal(self, row, previous_sma):
        sma = row.get('sma')
        rsi = row.get('rsi')
        return "enter_long" if sma is not None and previous_sma is not None and sma > previous_sma and rsi is not None and rsi < 35 else "hold"

    def _get_bb_signal(self, row):
        bb_lband = row.get('bb_lband')
        close_price = row.get('close', 0)
        return "enter_long" if bb_lband is not None and close_price <= bb_lband * 1.002 else "hold"

    def get_signal(self, data, current_time, **kwargs):
        try:
            current_row = data.loc[current_time]
        except Exception:
            return "hold"
        # 최소한 신호가 변경될 때만 로깅
        signals = [
            "enter_long" if self._get_candle_pattern_signal(current_row) == "bullish" else "hold",
            self._get_sma_rsi_signal(current_row, data.loc[:current_time].iloc[-2].get('sma') if len(data.loc[:current_time]) > 1 else current_row.get('sma')),
            self._get_bb_signal(current_row)
        ]
        final_signal = "enter_long" if "enter_long" in signals else "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"SelectStrategy signal changed to {final_signal} at {current_time}")
            self.previous_signal = final_signal
        return final_signal

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        try:
            row = data.loc[current_time]
        except Exception:
            return "hold"
        final_signal = "enter_long" if row.get('sma') is not None and row.get('close') > row.get('sma') else "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"TrendFollowingStrategy signal changed to {final_signal} at {current_time}")
            self.previous_signal = final_signal
        return final_signal

class BreakoutStrategy(BaseStrategy):
    def __init__(self, window=20):
        super().__init__()
        self.window = window
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        try:
            data_sub = data.loc[:current_time]
            if len(data_sub) < self.window:
                final_signal = "hold"
            else:
                recent_high = data_sub['high'].iloc[-self.window:].max()
                final_signal = "enter_long" if data.loc[current_time, 'close'] > recent_high else "hold"
        except Exception:
            final_signal = "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"BreakoutStrategy signal changed to {final_signal} at {current_time}")
            self.previous_signal = final_signal
        return final_signal

class CounterTrendStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        try:
            row = data.loc[current_time]
        except Exception:
            return "hold"
        rsi = row.get('rsi')
        if rsi is not None:
            final_signal = "enter_long" if rsi < 30 else ("exit_all" if rsi > 70 else "hold")
        else:
            final_signal = "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"CounterTrendStrategy signal changed to {final_signal} at {current_time} (RSI: {rsi})")
            self.previous_signal = final_signal
        return final_signal

class HighFrequencyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        try:
            idx = data.index.get_loc(current_time)
            if idx == 0:
                final_signal = "hold"
            else:
                current_row = data.iloc[idx]
                prev_row = data.iloc[idx - 1]
                cp, pp = current_row.get('close'), prev_row.get('close')
                if cp is None or pp is None:
                    final_signal = "hold"
                else:
                    threshold = 0.002
                    price_change = (cp - pp) / pp
                    final_signal = "enter_long" if price_change > threshold else ("exit_all" if price_change < -threshold else "hold")
        except Exception:
            final_signal = "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"HighFrequencyStrategy signal changed to {final_signal} at {current_time}")
            self.previous_signal = final_signal
        return final_signal

class WeeklyBreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def get_signal(self, data_weekly, current_time, breakout_threshold=0.01, **kwargs):
        try:
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if len(weekly_data) < 2:
                return "hold"
            prev_week = weekly_data.iloc[-2]
            current_week = weekly_data.iloc[-1]
            if current_week.get('close') is None or prev_week.get('high') is None or prev_week.get('low') is None:
                return "hold"
            if current_week.get('close') >= prev_week.get('high') * (1 + breakout_threshold):
                signal = "enter_long"
            elif current_week.get('close') <= prev_week.get('low') * (1 - breakout_threshold):
                signal = "exit_all"
            else:
                signal = "hold"
            return signal
        except Exception:
            return "hold"

class WeeklyMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def get_signal(self, data_weekly, current_time, momentum_threshold=0.5, **kwargs):
        try:
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if weekly_data.empty:
                return "hold"
            momentum = weekly_data.iloc[-1].get('weekly_momentum')
            if momentum is None:
                return "hold"
            return "enter_long" if momentum >= momentum_threshold else ("exit_all" if momentum <= -momentum_threshold else "hold")
        except Exception:
            return "hold"

class TradingStrategies:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.select = SelectStrategy()
        self.trend = TrendFollowingStrategy()
        self.breakout = BreakoutStrategy()
        self.counter = CounterTrendStrategy()
        self.hf = HighFrequencyStrategy()
        self.weekly_breakout = WeeklyBreakoutStrategy()
        self.weekly_momentum = WeeklyMomentumStrategy()
        self.last_final_signal = None

    def get_final_signal(self, market_regime, liquidity_info, data, current_time, data_weekly=None):
        signals = {
            "select": self.select.get_signal(data, current_time),
            "trend": self.trend.get_signal(data, current_time),
            "breakout": self.breakout.get_signal(data, current_time),
            "counter": self.counter.get_signal(data, current_time),
            "hf": self.hf.get_signal(data, current_time)
        }
        if data_weekly is not None:
            signals["weekly_breakout"] = self.weekly_breakout.get_signal(data_weekly, current_time)
            signals["weekly_momentum"] = self.weekly_momentum.get_signal(data_weekly, current_time)

        # 단기 전략 가중치 0.7, 주간 0.3
        vote_enter = sum(0.7 for key in ["select", "trend", "breakout", "counter", "hf"] if signals.get(key) == "enter_long")
        vote_exit = sum(0.7 for key in ["select", "trend", "breakout", "counter", "hf"] if signals.get(key) == "exit_all")
        if data_weekly is not None:
            vote_enter += sum(0.3 for key in ["weekly_breakout", "weekly_momentum"] if signals.get(key) == "enter_long")
            vote_exit += sum(0.3 for key in ["weekly_breakout", "weekly_momentum"] if signals.get(key) == "exit_all")
        final_signal = "exit_all" if vote_exit > vote_enter else ("enter_long" if vote_enter > vote_exit else "hold")
        if self.last_final_signal != final_signal:
            self.logger.debug(f"Final ensemble signal changed to {final_signal} at {current_time}")
            self.last_final_signal = final_signal
        return final_signal
