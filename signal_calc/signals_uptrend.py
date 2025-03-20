# signal_calc/signals_uptrend.py
from logs.log_config import setup_logger
from signal_calc.calc_signal import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    """
    TrendFollowingStrategy: 단순 이동평균 기반 추세 추종 전략.
    상승 추세에서 매수 신호("enter_long")를 도출합니다.
    
    가정:
      - 데이터프레임에 'sma' 컬럼이 존재.
    """
    def __init__(self):
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        """
        종가가 단순 이동평균(SMA)보다 높으면 "enter_long", 그렇지 않으면 "hold"를 반환합니다.
        """
        try:
            row = data.loc[current_time]
            if row.get('sma') is not None and row.get('close') > row.get('sma'):
                signal = "enter_long"
            else:
                signal = "hold"
        except Exception:
            signal = "hold"

        if self.previous_signal != signal:
            self.logger.info(f"TrendFollowingStrategy signal changed to {signal} at {current_time}")
            self.previous_signal = signal
        return signal

class BreakoutStrategy(BaseStrategy):
    """
    BreakoutStrategy: 최근 고점을 돌파하면 매수("enter_long") 신호를 생성하는 전략.
    """
    def __init__(self, window=20):
        super().__init__()
        self.window = window
        self.previous_signal = None


    def get_signal(self, data, current_time, **kwargs):
        """
        최근 'window' 기간 동안의 최고가를 돌파하면 "enter_long" 신호를 반환합니다.
        """
        try:
            data_sub = data.loc[:current_time]
            if len(data_sub) < self.window:
                signal = "hold"
            else:
                recent_high = data_sub['high'].iloc[-self.window:].max()
                if data.loc[current_time, 'close'] > recent_high:
                    signal = "enter_long"
                else:
                    signal = "hold"
        except Exception:
            signal = "hold"

        if self.previous_signal != signal:
            self.logger.info(f"BreakoutStrategy signal changed to {signal} at {current_time}")
            self.previous_signal = signal
        return signal

class WeeklyMomentumStrategy(BaseStrategy):
    """
    WeeklyMomentumStrategy: 주간 모멘텀 기반 전략.
    주간 데이터의 모멘텀 지표가 임계값 이상이면 "enter_long" 신호를 생성합니다.
    
    가정:
      - 데이터프레임에 'weekly_momentum' 컬럼이 존재.
    """
    def __init__(self, momentum_threshold=0.5):
        super().__init__()
        self.momentum_threshold = momentum_threshold
        self.previous_signal = None

    def get_signal(self, data_weekly, current_time, **kwargs):
        """
        주간 시장 데이터에서 모멘텀 지표를 평가하여 신호를 생성합니다.
        """
        try:
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if weekly_data.empty:
                signal = "hold"
            else:
                momentum = weekly_data.iloc[-1].get('weekly_momentum')
                if momentum is not None and momentum >= self.momentum_threshold:
                    signal = "enter_long"
                else:
                    signal = "hold"
        except Exception:
            signal = "hold"

        if self.previous_signal != signal:
            self.logger.info(f"WeeklyMomentumStrategy signal changed to {signal} at {current_time}")
            self.previous_signal = signal
        return signal
