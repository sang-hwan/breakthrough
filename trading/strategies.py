# trading/strategies.py
from logs.logger_config import setup_logger

class TradingStrategies:
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        # 각 전략별 마지막 최종 신호를 저장 (신호 변경 감지용)
        self.previous_signals = {
            "select_strategy": None,
            "trend_following_strategy": None,
            "breakout_strategy": None,
            "counter_trend_strategy": None,
            "high_frequency_strategy": None
        }

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

    def select_strategy(self, market_regime: str, liquidity_info: str, data, current_time, market_type: str = "crypto") -> str:
        """
        여러 보조 신호를 집계하여 최종 거래 신호를 결정합니다.
        """
        regime = market_regime.lower()
        signals = []

        try:
            current_row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"select_strategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            return "hold"

        if regime == "bullish":
            candle_signal = self._get_candle_pattern_signal(current_row)
            signals.append("enter_long" if candle_signal == "bullish" else "hold")
            
            try:
                previous_rows = data.loc[:current_time]
                if len(previous_rows) > 1:
                    previous_sma = previous_rows.iloc[-2].get('sma', current_row.get('sma'))
                else:
                    previous_sma = current_row.get('sma')
            except Exception as e:
                self.logger.error(f"select_strategy: 이전 데이터 조회 실패: {e}", exc_info=True)
                previous_sma = current_row.get('sma')
            sma_rsi_signal = self._get_sma_rsi_signal(current_row, previous_sma)
            signals.append(sma_rsi_signal)
            
            bb_signal = self._get_bb_signal(current_row)
            signals.append(bb_signal)
            
            final_signal = "enter_long" if "enter_long" in signals else "hold"
        elif regime == "bearish":
            final_signal = "exit_all"
        elif regime == "sideways":
            final_signal = "range_trade" if liquidity_info.lower() == "high" else "mean_reversion"
        else:
            final_signal = "hold"

        # 신호 변경 감지: 신호가 변경되었으면 INFO 레벨로 기록합니다.
        # (신호 유지의 경우 DEBUG로 남기지만, 프로젝트에서는 INFO 이상만 출력됨)
        key = "select_strategy"
        if self.previous_signals.get(key) != final_signal:
            self.logger.info(f"select_strategy: 신호 변경, 최종 신호: {final_signal} at {current_time}")
            self.previous_signals[key] = final_signal
        else:
            self.logger.debug(f"select_strategy: 신호 유지: '{final_signal}' at {current_time}")

        return final_signal

    def trend_following_strategy(self, data, current_time):
        key = "trend_following_strategy"
        try:
            row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"trend_following_strategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            return "hold"
        sma = row.get('sma')
        price = row.get('close')
        final_signal = "enter_long" if sma is not None and price is not None and price > sma else "hold"
        
        if self.previous_signals.get(key) != final_signal:
            self.logger.info(f"trend_following_strategy: 신호 변경, 최종 신호: {final_signal} at {current_time}")
            self.previous_signals[key] = final_signal
        else:
            self.logger.debug(f"trend_following_strategy: 신호 유지: '{final_signal}' at {current_time}")
            
        return final_signal

    def breakout_strategy(self, data, current_time, window=20):
        key = "breakout_strategy"
        try:
            data_sub = data.loc[:current_time]
            if len(data_sub) < window:
                final_signal = "hold"
            else:
                recent_high = data_sub['high'].iloc[-window:].max()
                price = data.loc[current_time, 'close']
                final_signal = "enter_long" if price > recent_high else "hold"
        except Exception as e:
            self.logger.error(f"breakout_strategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            final_signal = "hold"
        
        if self.previous_signals.get(key) != final_signal:
            self.logger.info(f"breakout_strategy: 신호 변경, 최종 신호: {final_signal} at {current_time}")
            self.previous_signals[key] = final_signal
        else:
            self.logger.debug(f"breakout_strategy: 신호 유지: '{final_signal}' at {current_time}")
        
        return final_signal

    def counter_trend_strategy(self, data, current_time):
        key = "counter_trend_strategy"
        try:
            row = data.loc[current_time]
        except Exception as e:
            self.logger.error(f"counter_trend_strategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
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
        
        if self.previous_signals.get(key) != final_signal:
            self.logger.info(f"counter_trend_strategy: 신호 변경, 최종 신호: {final_signal} at {current_time} (rsi: {rsi})")
            self.previous_signals[key] = final_signal
        else:
            self.logger.debug(f"counter_trend_strategy: 신호 유지: '{final_signal}' at {current_time} (rsi: {rsi})")
        
        return final_signal

    def high_frequency_strategy(self, data, current_time):
        key = "high_frequency_strategy"
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
                    threshold = 0.002  # 0.2% 임계치
                    price_change = (current_price - prev_price) / prev_price
                    if price_change > threshold:
                        final_signal = "enter_long"
                    elif price_change < -threshold:
                        final_signal = "exit_all"
                    else:
                        final_signal = "hold"
        except Exception as e:
            self.logger.error(f"high_frequency_strategy: 데이터 조회 실패 for time {current_time}: {e}", exc_info=True)
            final_signal = "hold"
        
        if self.previous_signals.get(key) != final_signal:
            self.logger.info(f"high_frequency_strategy: 신호 변경, 최종 신호: {final_signal} at {current_time}")
            self.previous_signals[key] = final_signal
        else:
            self.logger.debug(f"high_frequency_strategy: 신호 유지: '{final_signal}' at {current_time}")
        
        return final_signal
