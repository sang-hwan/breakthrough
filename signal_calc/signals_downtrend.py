# signal_calc/signals_downtrend.py
from logs.log_config import setup_logger
from signal_calc.calc_signal import BaseStrategy


class CounterTrendStrategy(BaseStrategy):
    """
    CounterTrendStrategy: RSI 지표를 활용하여 과매수 시 "exit_all", 과매도 시 "enter_long" 신호를 생성합니다.
    """
    def __init__(self, rsi_overbought=70, rsi_oversold=30):
        super().__init__()
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        """
        RSI 값을 평가하여 신호를 생성합니다.
        """
        try:
            row = data.loc[current_time]
            rsi = row.get('rsi')
            if rsi is None:
                signal = "hold"
            elif rsi > self.rsi_overbought:
                signal = "exit_all"
            elif rsi < self.rsi_oversold:
                signal = "enter_long"
            else:
                signal = "hold"
        except Exception:
            signal = "hold"

        if self.previous_signal != signal:
            self.logger.info(f"CounterTrendStrategy signal changed to {signal} at {current_time} (RSI: {rsi})")
            self.previous_signal = signal
        return signal

class HighFrequencyStrategy(BaseStrategy):
    """
    HighFrequencyStrategy: 인접 데이터 간의 가격 변화율을 평가하여 빠른 신호를 생성합니다.
    """
    def __init__(self, threshold=0.002):
        super().__init__()
        self.threshold = threshold
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        """
        인접 데이터 간의 가격 변화율을 계산하여 신호("enter_long"/"exit_all")를 생성합니다.
        """
        try:
            idx = data.index.get_loc(current_time)
            if idx == 0:
                signal = "hold"
            else:
                current_row = data.iloc[idx]
                prev_row = data.iloc[idx - 1]
                cp = current_row.get('close')
                pp = prev_row.get('close')
                if cp is None or pp is None:
                    signal = "hold"
                else:
                    price_change = (cp - pp) / pp
                    if price_change > self.threshold:
                        signal = "enter_long"
                    elif price_change < -self.threshold:
                        signal = "exit_all"
                    else:
                        signal = "hold"
        except Exception:
            signal = "hold"

        if self.previous_signal != signal:
            self.logger.info(f"HighFrequencyStrategy signal changed to {signal} at {current_time}")
            self.previous_signal = signal
        return signal
