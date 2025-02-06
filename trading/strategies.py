# trading/strategies.py
from logs.logger_config import setup_logger

class TradingStrategies:
    def __init__(self):
        self.logger = setup_logger(__name__)
    
    def select_strategy(self, market_regime: str, liquidity_info: str, data, current_time, market_type: str = "crypto") -> str:
        """
        기본 전략 선택:
        - 기술적 지표(SMA, RSI, Bollinger Bands 등)와 캔들 패턴을 기반으로 신호를 생성합니다.
        """
        regime = market_regime.lower()
        try:
            current_row = data.loc[current_time]
        except Exception:
            current_row = {}
        
        sma = current_row.get('sma', None)
        rsi = current_row.get('rsi', None)
        bb_lband = current_row.get('bb_lband', None)
        
        candle_pattern_signal = None
        if current_row.get('open') and current_row.get('close'):
            if current_row['close'] > current_row['open'] * 1.01:
                candle_pattern_signal = "bullish"
            elif current_row['close'] < current_row['open'] * 0.99:
                candle_pattern_signal = "bearish"
        
        previous_sma = sma
        previous_rows = data.loc[:current_time]
        if len(previous_rows) > 1:
            previous_sma = previous_rows.iloc[-2].get('sma', sma)
        
        if regime == "bullish":
            if (sma is not None and previous_sma is not None and sma > previous_sma and rsi is not None and rsi < 30) \
                or (candle_pattern_signal == "bullish") \
                or (bb_lband is not None and current_row.get('close', 0) <= bb_lband):
                return "enter_long"
            else:
                return "hold"
        elif regime == "bearish":
            return "exit_all"
        elif regime == "sideways":
            if liquidity_info.lower() == "high":
                return "range_trade"
            else:
                return "mean_reversion"
        else:
            return "hold"
    
    def trend_following_strategy(self, data, current_time):
        """
        추세 추종 전략:
        - 현재 가격이 단순 이동평균(SMA) 위에 있으면 매수 신호를 반환합니다.
        """
        try:
            row = data.loc[current_time]
        except Exception:
            return None
        sma = row.get('sma')
        price = row.get('close')
        if sma is not None and price is not None and price > sma:
            return "enter_long"
        return "hold"
    
    def breakout_strategy(self, data, current_time, window=20):
        """
        돌파 전략:
        - 최근 window 기간 내 최고가를 돌파하면 매수 신호를 반환합니다.
        """
        data_sub = data.loc[:current_time]
        if len(data_sub) < window:
            return None
        recent_high = data_sub['high'].iloc[-window:].max()
        price = data.loc[current_time, 'close']
        if price > recent_high:
            return "enter_long"
        return "hold"
    
    def counter_trend_strategy(self, data, current_time):
        """
        역추세 전략:
        - RSI가 과매도(예: 30 이하)인 경우 매수, 과매수(70 이상)일 경우 청산 신호를 반환합니다.
        """
        try:
            row = data.loc[current_time]
        except Exception:
            return None
        rsi = row.get('rsi')
        if rsi is not None:
            if rsi < 30:
                return "enter_long"
            elif rsi > 70:
                return "exit_all"
        return "hold"
    
    def high_frequency_strategy(self, data, current_time):
        """
        고빈도 전략:
        - 단기 차트(예: 15분 혹은 1시간봉) 데이터를 사용하여 미세한 가격 변동(예: 0.2% 이상 상승 또는 하락)을 포착합니다.
        - 상승 변동 시 매수("enter_long"), 하락 변동 시 청산("exit_all") 신호를 반환하며, 그 외엔 "hold"를 반환합니다.
        """
        try:
            current_index = data.index.get_loc(current_time)
            if current_index == 0:
                return "hold"
            prev_time = data.index[current_index - 1]
            current_row = data.loc[current_time]
            prev_row = data.loc[prev_time]
        except Exception as e:
            self.logger.error(f"High frequency strategy error: {e}")
            return "hold"
        
        current_price = current_row.get('close')
        prev_price = prev_row.get('close')
        if current_price is None or prev_price is None:
            return "hold"
        
        threshold = 0.002  # 0.2% 임계치
        price_change = (current_price - prev_price) / prev_price
        if price_change > threshold:
            return "enter_long"
        elif price_change < -threshold:
            return "exit_all"
        return "hold"
