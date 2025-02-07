# trading/strategies.py
from logs.logger_config import setup_logger

class TradingStrategies:
    def __init__(self):
        self.logger = setup_logger(__name__)
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
            self.logger.error(f"_get_candle_pattern_signal error: {e}")
        return None

    def _get_sma_rsi_signal(self, row, previous_sma):
        try:
            sma = row.get('sma')
            rsi = row.get('rsi')
            if sma is not None and previous_sma is not None and sma > previous_sma and rsi is not None and rsi < 35:
                return "enter_long"
        except Exception as e:
            self.logger.error(f"_get_sma_rsi_signal error: {e}")
        return "hold"

    def _get_bb_signal(self, row):
        try:
            bb_lband = row.get('bb_lband')
            close_price = row.get('close', 0)
            if bb_lband is not None and close_price <= bb_lband * 1.002:
                return "enter_long"
        except Exception as e:
            self.logger.error(f"_get_bb_signal error: {e}")
        return "hold"

    def select_strategy(self, market_regime: str, liquidity_info: str, data, current_time, market_type: str = "crypto") -> str:
        regime = market_regime.lower()
        signals = []

        try:
            current_row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"select_strategy: 데이터 조회 실패 for time {current_time}: {e}")
            return "hold"

        if regime == "bullish":
            candle_signal = self._get_candle_pattern_signal(current_row)
            self.logger.debug(f"_get_candle_pattern_signal: {candle_signal} at {current_time}")
            signals.append("enter_long" if candle_signal == "bullish" else "hold")
            
            try:
                previous_rows = data.loc[:current_time]
                if len(previous_rows) > 1:
                    previous_sma = previous_rows.iloc[-2].get('sma', current_row.get('sma'))
                else:
                    previous_sma = current_row.get('sma')
            except Exception as e:
                self.logger.error(f"select_strategy: 이전 데이터 조회 실패: {e}")
                previous_sma = current_row.get('sma')
                
            sma_rsi_signal = self._get_sma_rsi_signal(current_row, previous_sma)
            self.logger.debug(f"_get_sma_rsi_signal: {sma_rsi_signal} at {current_time}")
            signals.append(sma_rsi_signal)
            
            bb_signal = self._get_bb_signal(current_row)
            self.logger.debug(f"_get_bb_signal: {bb_signal} at {current_time}")
            signals.append(bb_signal)
            
            final_signal = "enter_long" if "enter_long" in signals else "hold"
        elif regime == "bearish":
            final_signal = "exit_all"
        elif regime == "sideways":
            final_signal = "range_trade" if liquidity_info.lower() == "high" else "mean_reversion"
        else:
            final_signal = "hold"

        self.logger.debug(f"select_strategy: regime={regime}, liquidity_info={liquidity_info}, signals={signals}")
        self.logger.info(f"select_strategy: 최종 신호={final_signal} at {current_time}")
        self.previous_signal = final_signal
        return final_signal

    def trend_following_strategy(self, data, current_time):
        try:
            row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"trend_following_strategy: 데이터 조회 실패 for time {current_time}: {e}")
            return "hold"
        sma = row.get('sma')
        price = row.get('close')
        if sma is not None and price is not None and price > sma:
            self.logger.debug(f"trend_following_strategy: bullish signal at {current_time}")
            return "enter_long"
        return "hold"

    def breakout_strategy(self, data, current_time, window=20):
        try:
            data_sub = data.loc[:current_time]
            if len(data_sub) < window:
                return "hold"
            recent_high = data_sub['high'].iloc[-window:].max()
            price = data.loc[current_time, 'close']
            if price > recent_high:
                self.logger.debug(f"breakout_strategy: breakout detected at {current_time}")
                return "enter_long"
        except Exception as e:
            self.logger.error(f"breakout_strategy: 데이터 조회 실패 for time {current_time}: {e}")
        return "hold"

    def counter_trend_strategy(self, data, current_time):
        try:
            row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"counter_trend_strategy: 데이터 조회 실패 for time {current_time}: {e}")
            return "hold"
        rsi = row.get('rsi')
        if rsi is not None:
            if rsi < 30:
                self.logger.debug(f"counter_trend_strategy: bullish signal at {current_time} (rsi: {rsi})")
                return "enter_long"
            elif rsi > 70:
                self.logger.debug(f"counter_trend_strategy: bearish signal at {current_time} (rsi: {rsi})")
                return "exit_all"
        return "hold"

    def high_frequency_strategy(self, data, current_time):
        try:
            current_index = data.index.get_loc(current_time)
            if current_index == 0:
                return "hold"
            prev_time = data.index[current_index - 1]
            current_row = data.loc[current_time]
            prev_row = data.loc[prev_time]
        except Exception as e:
            self.logger.error(f"high_frequency_strategy: 데이터 조회 실패 for time {current_time}: {e}")
            return "hold"
        current_price = current_row.get('close')
        prev_price = prev_row.get('close')
        if current_price is None or prev_price is None:
            return "hold"
        threshold = 0.002  # 0.2% 임계치
        price_change = (current_price - prev_price) / prev_price
        if price_change > threshold:
            self.logger.debug(f"high_frequency_strategy: bullish price change ({price_change:.4f}) at {current_time}")
            return "enter_long"
        elif price_change < -threshold:
            self.logger.debug(f"high_frequency_strategy: bearish price change ({price_change:.4f}) at {current_time}")
            return "exit_all"
        return "hold"
