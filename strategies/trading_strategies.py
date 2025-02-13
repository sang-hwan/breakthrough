# strategies/trading_strategies.py
from logs.logger_config import setup_logger
from strategies.base_strategy import BaseStrategy

class SelectStrategy(BaseStrategy):
    """
    캔들 패턴, SMA+RSI, Bollinger Bands 신호를 결합하여 최종 진입 신호를 산출합니다.
    """
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def _get_candle_pattern_signal(self, row):
        try:
            open_price = row.get('open')
            close_price = row.get('close')
            if open_price is None or close_price is None:
                return None
            if close_price > open_price * 1.005:
                return "bullish"
            elif close_price < open_price * 0.99:
                return "bearish"
        except Exception as e:
            self.logger.error(f"_get_candle_pattern_signal error: {e}", exc_info=True)
        return None

    def _get_sma_rsi_signal(self, row, previous_sma):
        try:
            sma = row.get('sma')
            rsi = row.get('rsi')
            if sma is not None and previous_sma is not None and sma > previous_sma and rsi is not None and rsi < 35:
                return "enter_long"
        except Exception as e:
            self.logger.error(f"_get_sma_rsi_signal error: {e}", exc_info=True)
        return "hold"

    def _get_bb_signal(self, row):
        try:
            bb_lband = row.get('bb_lband')
            close_price = row.get('close', 0)
            if bb_lband is not None and close_price <= bb_lband * 1.002:
                return "enter_long"
        except Exception as e:
            self.logger.error(f"_get_bb_signal error: {e}", exc_info=True)
        return "hold"

    def get_signal(self, data, current_time, **kwargs):
        try:
            current_row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"SelectStrategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            return "hold"

        signals = []
        # 캔들 패턴 신호
        candle_signal = self._get_candle_pattern_signal(current_row)
        signals.append("enter_long" if candle_signal == "bullish" else "hold")
        
        # SMA+RSI 신호 (이전 행의 SMA 사용)
        try:
            previous_rows = data.loc[:current_time]
            if len(previous_rows) > 1:
                previous_sma = previous_rows.iloc[-2].get('sma', current_row.get('sma'))
            else:
                previous_sma = current_row.get('sma')
        except Exception as e:
            self.logger.error(f"SelectStrategy: 이전 데이터 조회 실패: {e}", exc_info=True)
            previous_sma = current_row.get('sma')
        sma_rsi_signal = self._get_sma_rsi_signal(current_row, previous_sma)
        signals.append(sma_rsi_signal)

        # Bollinger Bands 신호
        bb_signal = self._get_bb_signal(current_row)
        signals.append(bb_signal)

        final_signal = "enter_long" if "enter_long" in signals else "hold"

        if self.previous_signal != final_signal:
            self.logger.debug(f"SelectStrategy: 신호 변경, 최종 신호: {final_signal} at {current_time}")
            self.previous_signal = final_signal
        else:
            self.logger.debug(f"SelectStrategy: 신호 유지: '{final_signal}' at {current_time}")

        return final_signal

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        try:
            row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"TrendFollowingStrategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            return "hold"
        sma = row.get('sma')
        price = row.get('close')
        final_signal = "enter_long" if sma is not None and price is not None and price > sma else "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"TrendFollowingStrategy: 신호 변경, 최종 신호: {final_signal} at {current_time}")
            self.previous_signal = final_signal
        else:
            self.logger.debug(f"TrendFollowingStrategy: 신호 유지: '{final_signal}' at {current_time}")
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
                price = data.loc[current_time, 'close']
                final_signal = "enter_long" if price > recent_high else "hold"
        except Exception as e:
            self.logger.error(f"BreakoutStrategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            final_signal = "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"BreakoutStrategy: 신호 변경, 최종 신호: {final_signal} at {current_time}")
            self.previous_signal = final_signal
        else:
            self.logger.debug(f"BreakoutStrategy: 신호 유지: '{final_signal}' at {current_time}")
        return final_signal

class CounterTrendStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        try:
            row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"CounterTrendStrategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            return "hold"
        rsi = row.get('rsi')
        if rsi is not None:
            if rsi < 30:
                final_signal = "enter_long"
            elif rsi > 70:
                final_signal = "exit_all"
            else:
                final_signal = "hold"
        else:
            final_signal = "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"CounterTrendStrategy: 신호 변경, 최종 신호: {final_signal} at {current_time} (rsi: {rsi})")
            self.previous_signal = final_signal
        else:
            self.logger.debug(f"CounterTrendStrategy: 신호 유지: '{final_signal}' at {current_time} (rsi: {rsi})")
        return final_signal

class HighFrequencyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        try:
            current_index = data.index.get_loc(current_time)
            if current_index == 0:
                final_signal = "hold"
            else:
                prev_time = data.index[current_index - 1]
                current_row = data.loc[current_time]
                prev_row = data.loc[prev_time]
                current_price = current_row.get('close')
                prev_price = prev_row.get('close')
                if current_price is None or prev_price is None:
                    final_signal = "hold"
                else:
                    threshold = 0.002  # 0.2%
                    price_change = (current_price - prev_price) / prev_price
                    if price_change > threshold:
                        final_signal = "enter_long"
                    elif price_change < -threshold:
                        final_signal = "exit_all"
                    else:
                        final_signal = "hold"
        except Exception as e:
            self.logger.error(f"HighFrequencyStrategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            final_signal = "hold"
        if self.previous_signal != final_signal:
            self.logger.debug(f"HighFrequencyStrategy: 신호 변경, 최종 신호: {final_signal} at {current_time}")
            self.previous_signal = final_signal
        else:
            self.logger.debug(f"HighFrequencyStrategy: 신호 유지: '{final_signal}' at {current_time}")
        return final_signal

class WeeklyBreakoutStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def get_signal(self, data_weekly, current_time, breakout_threshold=0.01, **kwargs):
        try:
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if len(weekly_data) < 2:
                self.logger.debug("주간 데이터 부족: 최소 2주 이상의 데이터 필요")
                return "hold"
            prev_week = weekly_data.iloc[-2]
            current_week = weekly_data.iloc[-1]
            prev_high = prev_week.get('high')
            prev_low = prev_week.get('low')
            current_close = current_week.get('close')
            if current_close is None or prev_high is None or prev_low is None:
                self.logger.error("주간 데이터에 필요한 컬럼 누락")
                return "hold"
            if current_close >= prev_high * (1 + breakout_threshold):
                signal = "enter_long"
            elif current_close <= prev_low * (1 - breakout_threshold):
                signal = "exit_all"
            else:
                signal = "hold"
            self.logger.debug(f"WeeklyBreakoutStrategy: prev_high={prev_high}, prev_low={prev_low}, current_close={current_close}, breakout_threshold={breakout_threshold}, signal={signal}")
            return signal
        except Exception as e:
            self.logger.error(f"WeeklyBreakoutStrategy 에러: {e}", exc_info=True)
            return "hold"

class WeeklyMomentumStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def get_signal(self, data_weekly, current_time, momentum_threshold=0.5, **kwargs):
        try:
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if weekly_data.empty:
                self.logger.debug("주간 데이터가 없습니다.")
                return "hold"
            current_week = weekly_data.iloc[-1]
            momentum = current_week.get('weekly_momentum')
            if momentum is None:
                self.logger.debug("weekly_momentum 컬럼이 누락되어 기본 'hold' 적용")
                return "hold"
            if momentum >= momentum_threshold:
                signal = "enter_long"
            elif momentum <= -momentum_threshold:
                signal = "exit_all"
            else:
                signal = "hold"
            self.logger.debug(f"WeeklyMomentumStrategy: momentum={momentum}, momentum_threshold={momentum_threshold}, signal={signal}")
            return signal
        except Exception as e:
            self.logger.error(f"WeeklyMomentumStrategy 에러: {e}", exc_info=True)
            return "hold"

class TradingStrategies:
    """
    종합 전략 관리 클래스.
    개별 전략들을 인스턴스화하고, 단기 및 주간 신호를 가중치 기반으로 종합하여 최종 신호를 산출합니다.
    """
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.select_strategy = SelectStrategy()
        self.trend_following_strategy = TrendFollowingStrategy()
        self.breakout_strategy = BreakoutStrategy()
        self.counter_trend_strategy = CounterTrendStrategy()
        self.high_frequency_strategy = HighFrequencyStrategy()
        self.weekly_breakout_strategy = WeeklyBreakoutStrategy()
        self.weekly_momentum_strategy = WeeklyMomentumStrategy()

    def get_final_signal(self, market_regime, liquidity_info, data, current_time, data_weekly=None):
        # 단기 전략 신호
        signals = {
            "select": self.select_strategy.get_signal(data, current_time),
            "trend_following": self.trend_following_strategy.get_signal(data, current_time),
            "breakout": self.breakout_strategy.get_signal(data, current_time),
            "counter_trend": self.counter_trend_strategy.get_signal(data, current_time),
            "high_frequency": self.high_frequency_strategy.get_signal(data, current_time)
        }
        # 주간 전략 신호 (데이터 제공 시)
        if data_weekly is not None:
            signals["weekly_breakout"] = self.weekly_breakout_strategy.get_signal(data_weekly, current_time)
            signals["weekly_momentum"] = self.weekly_momentum_strategy.get_signal(data_weekly, current_time)
        
        self.logger.debug(f"각 전략 원시 신호: {signals}")

        # 가중치 적용 (단기: 0.7, 주간: 0.3)
        short_term_weight = 0.7
        weekly_weight = 0.3 if data_weekly is not None else 0.0

        vote_enter = 0.0
        vote_exit = 0.0

        for key in ["select", "trend_following", "breakout", "counter_trend", "high_frequency"]:
            sig = signals.get(key)
            if sig == "enter_long":
                vote_enter += short_term_weight
            elif sig == "exit_all":
                vote_exit += short_term_weight

        for key in ["weekly_breakout", "weekly_momentum"]:
            sig = signals.get(key)
            if sig == "enter_long":
                vote_enter += weekly_weight
            elif sig == "exit_all":
                vote_exit += weekly_weight

        if vote_exit > vote_enter:
            final_signal = "exit_all"
        elif vote_enter > vote_exit:
            final_signal = "enter_long"
        else:
            final_signal = "hold"

        self.logger.debug(f"최종 종합 신호: {final_signal} (vote_enter={vote_enter}, vote_exit={vote_exit})")
        return final_signal
