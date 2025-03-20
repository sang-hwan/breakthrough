# signal_calc/signals_sideways.py
from logs.log_config import setup_logger
from signal_calc.calc_signal import BaseStrategy

class RangeTradingStrategy(BaseStrategy):
    """
    RangeTradingStrategy: Bollinger Bands의 상한/하한을 활용한 레인지 트레이딩 전략.
    가격이 하한 근접 시 "enter_long", 상한 근접 시 "exit_all" 신호를 생성합니다.
    """
    def __init__(self, tolerance=0.002):
        super().__init__()
        self.tolerance = tolerance
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        """
        Bollinger Bands 하한/상한과 종가를 비교하여 거래 신호를 생성합니다.
        """
        try:
            row = data.loc[current_time]
            bb_lband = row.get('bb_lband')
            bb_hband = row.get('bb_hband')
            close_price = row.get('close')
            if bb_lband is None or bb_hband is None or close_price is None:
                signal = "hold"
            elif close_price <= bb_lband * (1 + self.tolerance):
                signal = "enter_long"
            elif close_price >= bb_hband * (1 - self.tolerance):
                signal = "exit_all"
            else:
                signal = "hold"
        except Exception:
            signal = "hold"

        if self.previous_signal != signal:
            self.logger.info(f"RangeTradingStrategy signal changed to {signal} at {current_time}")
            self.previous_signal = signal
        return signal
