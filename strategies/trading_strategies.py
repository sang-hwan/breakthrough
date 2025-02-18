# strategies/trading_strategies.py
from logs.logger_config import setup_logger
from strategies.base_strategy import BaseStrategy
from markets.regime_filter import determine_weekly_extreme_signal

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
        signals = [
            "enter_long" if self._get_candle_pattern_signal(current_row) == "bullish" else "hold",
            self._get_sma_rsi_signal(
                current_row,
                data.loc[:current_time].iloc[-2].get('sma') if len(data.loc[:current_time]) > 1 else current_row.get('sma')
            ),
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
        self.previous_signal = None

    def get_signal(self, data_weekly, current_time, breakout_threshold=0.01, **kwargs):
        try:
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if len(weekly_data) < 2:
                return "hold"
            prev_week = weekly_data.iloc[-2]
            current_week = weekly_data.iloc[-1]
            # 주간 극값 신호 판단: aggregator에서 재명명된 'weekly_low', 'weekly_high' 사용
            price_data = {"current_price": current_week.get('close')}
            weekly_extremes = {"weekly_low": prev_week.get('weekly_low'), "weekly_high": prev_week.get('weekly_high')}
            extreme_signal = determine_weekly_extreme_signal(price_data, weekly_extremes, threshold=breakout_threshold)
            if extreme_signal:
                signal = extreme_signal
            else:
                if current_week.get('close') >= prev_week.get('weekly_high') * (1 + breakout_threshold):
                    signal = "enter_long"
                elif current_week.get('close') <= prev_week.get('weekly_low') * (1 - breakout_threshold):
                    signal = "exit_all"
                else:
                    signal = "hold"
            if self.previous_signal != signal:
                self.logger.debug(f"WeeklyBreakoutStrategy signal changed to {signal} at {current_time}")
                self.previous_signal = signal
            return signal
        except Exception:
            return "hold"

class WeeklyMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data_weekly, current_time, momentum_threshold=0.5, **kwargs):
        try:
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if weekly_data.empty:
                return "hold"
            momentum = weekly_data.iloc[-1].get('weekly_momentum')
            if momentum is None:
                return "hold"
            signal = "enter_long" if momentum >= momentum_threshold else ("exit_all" if momentum <= -momentum_threshold else "hold")
            if self.previous_signal != signal:
                self.logger.debug(f"WeeklyMomentumStrategy signal changed to {signal} at {current_time}")
                self.previous_signal = signal
            return signal
        except Exception:
            return "hold"

class TradingStrategies:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        from trading.ensemble import Ensemble
        self.ensemble = Ensemble()
        self.weekly_breakout = self.ensemble.weekly_breakout_strategy
        self.weekly_momentum = self.ensemble.weekly_momentum_strategy

    def get_final_signal(self, market_regime, liquidity_info, data, current_time, data_weekly=None, **kwargs):
        # 우선 Ensemble의 최종 신호를 획득
        ensemble_signal = self.ensemble.get_final_signal(market_regime, liquidity_info, data, current_time, data_weekly, **kwargs)
        # HMM 기반 시장 레짐에 따른 신호 override:
        # 예를 들어, bearish 레짐이면 위험 관리 차원에서 강제 청산(exit_all)을 적용합니다.
        if market_regime == "bearish":
            self.logger.debug("Market regime bearish: overriding final signal to exit_all")
            return "exit_all"
        elif market_regime == "bullish":
            self.logger.debug("Market regime bullish: ensuring signal is at least enter_long")
            return "enter_long" if ensemble_signal == "hold" else ensemble_signal
        return ensemble_signal
