# strategies/trading_strategies.py
from logs.logger_config import setup_logger  # 로거 설정 함수
from strategies.base_strategy import BaseStrategy  # 기본 전략 클래스
from markets.regime_filter import determine_weekly_extreme_signal  # 주간 극단값 신호를 판단하는 함수

# ------------------------------------------------------------------
# SelectStrategy: 여러 개의 개별 신호(캔들 패턴, SMA/RSI, 볼린저 밴드)를 결합해 최종 거래 신호를 생성
# ------------------------------------------------------------------
class SelectStrategy(BaseStrategy):
    def __init__(self):
        """
        SelectStrategy 생성자.
        
        목적:
          - BaseStrategy를 상속받아 로거를 초기화하고, 이전 신호를 저장할 변수를 설정합니다.
        """
        super().__init__()
        self.previous_signal = None  # 이전에 생성된 거래 신호를 저장 (변경 시 로깅용)

    def _get_candle_pattern_signal(self, row):
        """
        캔들 패턴 데이터를 기반으로 상승 또는 하락 신호를 판단합니다.
        
        Parameters:
            row (dict): 현재 캔들 데이터 (open, close 등 포함).
        
        Returns:
            str or None: "bullish" (상승) 또는 "bearish" (하락) 신호, 해당하지 않으면 None.
        """
        open_price = row.get('open')
        close_price = row.get('close')
        if open_price is None or close_price is None:
            return None
        # 상승: 종가가 시가 대비 0.5% 이상 상승, 하락: 종가가 시가 대비 1% 이상 하락
        return "bullish" if close_price > open_price * 1.005 else ("bearish" if close_price < open_price * 0.99 else None)

    def _get_sma_rsi_signal(self, row, previous_sma):
        """
        단순 이동평균(SMA)과 RSI 지표를 이용해 거래 신호를 생성합니다.
        
        Parameters:
            row (dict): 현재 시점의 데이터 (SMA, RSI 포함).
            previous_sma (float): 바로 이전 시점의 SMA 값.
        
        Returns:
            str: 조건에 따라 "enter_long" (매수 진입) 또는 "hold" (대기) 신호.
        """
        sma = row.get('sma')
        rsi = row.get('rsi')
        # 조건: 현재 SMA가 이전 SMA보다 상승하고 RSI가 35 미만이면 매수 신호
        return "enter_long" if sma is not None and previous_sma is not None and sma > previous_sma and rsi is not None and rsi < 35 else "hold"

    def _get_bb_signal(self, row):
        """
        볼린저 밴드 하단을 기준으로 거래 신호를 생성합니다.
        
        Parameters:
            row (dict): 현재 데이터 행 (종가, 볼린저 밴드 하단 포함).
        
        Returns:
            str: "enter_long" (매수 진입) 또는 "hold" (대기).
        """
        bb_lband = row.get('bb_lband')
        close_price = row.get('close', 0)
        # 조건: 종가가 볼린저 밴드 하단의 0.2% 범위 내이면 매수 신호
        return "enter_long" if bb_lband is not None and close_price <= bb_lband * 1.002 else "hold"

    def get_signal(self, data, current_time, **kwargs):
        """
        여러 개의 신호 생성 함수를 결합하여 최종 거래 신호를 도출합니다.
        
        Parameters:
            data (pandas.DataFrame): 시계열 시장 데이터 (각종 지표 포함).
            current_time (datetime): 기준 시점.
            **kwargs: 추가 인자들.
        
        Returns:
            str: "enter_long" 또는 "hold"와 같이 거래 진입 여부를 나타내는 신호.
        """
        try:
            # 현재 시점에 해당하는 데이터 행 선택
            current_row = data.loc[current_time]
        except Exception:
            return "hold"
        # 세 가지 개별 신호를 생성 후, 하나라도 "enter_long"이면 최종 신호는 "enter_long"로 결정
        signals = [
            "enter_long" if self._get_candle_pattern_signal(current_row) == "bullish" else "hold",
            self._get_sma_rsi_signal(
                current_row,
                # 현재 시점까지의 데이터 중 바로 이전 행의 SMA 값 사용 (데이터가 1개인 경우 현재값 사용)
                data.loc[:current_time].iloc[-2].get('sma') if len(data.loc[:current_time]) > 1 else current_row.get('sma')
            ),
            self._get_bb_signal(current_row)
        ]
        final_signal = "enter_long" if "enter_long" in signals else "hold"
        # 신호가 변경된 경우에만 로깅 처리
        if self.previous_signal != final_signal:
            self.logger.info(f"SelectStrategy 신호: {final_signal}")
            self.previous_signal = final_signal
        return final_signal

# ------------------------------------------------------------------
# TrendFollowingStrategy: 단순 이동평균을 기반으로 추세 추종 거래 전략을 구현
# ------------------------------------------------------------------
class TrendFollowingStrategy(BaseStrategy):
    def __init__(self):
        """
        TrendFollowingStrategy 생성자.
        
        목적:
          - 이동평균(SMA) 기반 전략을 사용하여 단순 추세에 따라 거래 신호를 결정합니다.
        """
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        """
        이동평균과 종가를 비교하여 거래 신호를 생성합니다.
        
        Parameters:
            data (pandas.DataFrame): 시장 데이터 (SMA, 종가 포함).
            current_time (datetime): 기준 시점.
            **kwargs: 추가 인자들.
        
        Returns:
            str: "enter_long" (매수 진입) 또는 "hold" (대기) 신호.
        """
        try:
            row = data.loc[current_time]
        except Exception:
            return "hold"
        # 조건: 종가가 SMA보다 높으면 매수 신호로 판단
        final_signal = "enter_long" if row.get('sma') is not None and row.get('close') > row.get('sma') else "hold"
        if self.previous_signal != final_signal:
            self.logger.info(f"TrendFollowingStrategy signal changed to {final_signal} at {current_time}")
            self.previous_signal = final_signal
        return final_signal

# ------------------------------------------------------------------
# BreakoutStrategy: 최근 고점을 돌파하는 경우 매수 신호를 발생시키는 전략
# ------------------------------------------------------------------
class BreakoutStrategy(BaseStrategy):
    def __init__(self, window=20):
        """
        BreakoutStrategy 생성자.
        
        Parameters:
            window (int): 돌파 판단을 위한 최근 데이터 포인트 수 (기본값: 20).
        """
        super().__init__()
        self.window = window  # 고점 계산에 사용할 기간
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        """
        최근 'window' 기간 동안의 최고가를 기준으로 돌파 여부를 판단하여 거래 신호를 생성합니다.
        
        Parameters:
            data (pandas.DataFrame): 시장 데이터 (고가, 종가 등 포함).
            current_time (datetime): 기준 시점.
            **kwargs: 추가 인자들.
        
        Returns:
            str: 돌파 시 "enter_long", 아니면 "hold".
        """
        try:
            # 현재 시점까지의 데이터를 선택
            data_sub = data.loc[:current_time]
            # 충분한 데이터가 없는 경우 대기
            if len(data_sub) < self.window:
                final_signal = "hold"
            else:
                # 최근 'window' 기간 동안의 최고가 계산
                recent_high = data_sub['high'].iloc[-self.window:].max()
                final_signal = "enter_long" if data.loc[current_time, 'close'] > recent_high else "hold"
        except Exception:
            final_signal = "hold"
        if self.previous_signal != final_signal:
            self.logger.info(f"BreakoutStrategy signal changed to {final_signal} at {current_time}")
            self.previous_signal = final_signal
        return final_signal

# ------------------------------------------------------------------
# CounterTrendStrategy: RSI 지표를 이용한 역추세 거래 전략
# ------------------------------------------------------------------
class CounterTrendStrategy(BaseStrategy):
    def __init__(self):
        """
        CounterTrendStrategy 생성자.
        
        목적:
          - RSI 지표를 활용해 과매도 혹은 과매수 상황에서 거래 신호를 생성합니다.
        """
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        """
        RSI 값에 따라 거래 신호를 생성합니다.
        
        Parameters:
            data (pandas.DataFrame): 시장 데이터 (RSI 포함).
            current_time (datetime): 기준 시점.
            **kwargs: 추가 인자들.
        
        Returns:
            str: RSI가 30 미만이면 "enter_long", 70 초과면 "exit_all", 그 외에는 "hold".
        """
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
            self.logger.info(f"CounterTrendStrategy signal changed to {final_signal} at {current_time} (RSI: {rsi})")
            self.previous_signal = final_signal
        return final_signal

# ------------------------------------------------------------------
# HighFrequencyStrategy: 고빈도 거래를 위한 빠른 신호 결정 전략
# ------------------------------------------------------------------
class HighFrequencyStrategy(BaseStrategy):
    def __init__(self):
        """
        HighFrequencyStrategy 생성자.
        
        목적:
          - 인접한 데이터 포인트 간의 미세한 가격 변화를 분석하여 빠른 거래 신호를 생성합니다.
        """
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data, current_time, **kwargs):
        """
        인접 데이터 간의 가격 변화율을 계산하여 거래 신호를 생성합니다.
        
        Parameters:
            data (pandas.DataFrame): 고빈도 시계열 데이터 (종가 포함).
            current_time (datetime): 기준 시점.
            **kwargs: 추가 인자들.
        
        Returns:
            str: 가격 상승 시 "enter_long", 하락 시 "exit_all", 아니면 "hold".
        """
        try:
            # 현재 시점의 인덱스 위치 확인
            idx = data.index.get_loc(current_time)
            if idx == 0:
                final_signal = "hold"  # 첫 데이터인 경우 이전 데이터가 없으므로 대기
            else:
                current_row = data.iloc[idx]
                prev_row = data.iloc[idx - 1]
                cp, pp = current_row.get('close'), prev_row.get('close')
                if cp is None or pp is None:
                    final_signal = "hold"
                else:
                    threshold = 0.002  # 가격 변화 임계값 (0.2%)
                    price_change = (cp - pp) / pp  # 상대 가격 변화율 계산
                    final_signal = "enter_long" if price_change > threshold else ("exit_all" if price_change < -threshold else "hold")
        except Exception:
            final_signal = "hold"
        if self.previous_signal != final_signal:
            self.logger.info(f"HighFrequencyStrategy signal changed to {final_signal} at {current_time}")
            self.previous_signal = final_signal
        return final_signal

# ------------------------------------------------------------------
# WeeklyBreakoutStrategy: 주간 데이터 기반 돌파 전략
# ------------------------------------------------------------------
class WeeklyBreakoutStrategy(BaseStrategy):
    def __init__(self):
        """
        WeeklyBreakoutStrategy 생성자.
        
        목적:
          - 주간 데이터를 분석하여 돌파 상황(고점 돌파 또는 저점 하향 돌파)을 판단하고 거래 신호를 생성합니다.
        """
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data_weekly, current_time, breakout_threshold=0.01, **kwargs):
        """
        주간 데이터에 기반하여 거래 신호를 결정합니다.
        
        Parameters:
            data_weekly (pandas.DataFrame): 주간 시장 데이터 (고가, 저가, 종가 등 포함).
            current_time (datetime): 기준 시점.
            breakout_threshold (float): 돌파 임계값 (예: 1%).
            **kwargs: 추가 인자들.
        
        Returns:
            str: 돌파 발생 시 "enter_long" 또는 "exit_all", 아니면 "hold".
        """
        try:
            # 현재 시점까지의 주간 데이터 선택
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if len(weekly_data) < 2:
                return "hold"
            # 바로 이전 주와 현재 주 데이터 추출
            prev_week = weekly_data.iloc[-2]
            current_week = weekly_data.iloc[-1]
            price_data = {"current_price": current_week.get('close')}
            weekly_extremes = {"weekly_low": prev_week.get('weekly_low'), "weekly_high": prev_week.get('weekly_high')}
            # 주간 극단값(최고/최저) 기준 신호 결정
            extreme_signal = determine_weekly_extreme_signal(price_data, weekly_extremes, threshold=breakout_threshold)
            if extreme_signal:
                signal = extreme_signal
            else:
                # 추가 조건: 현재 주 종가가 이전 주의 고점 또는 저점을 돌파하는 경우 신호 발생
                if current_week.get('close') >= prev_week.get('weekly_high') * (1 + breakout_threshold):
                    signal = "enter_long"
                elif current_week.get('close') <= prev_week.get('weekly_low') * (1 - breakout_threshold):
                    signal = "exit_all"
                else:
                    signal = "hold"
            if self.previous_signal != signal:
                self.logger.info(f"WeeklyBreakoutStrategy signal changed to {signal} at {current_time}")
                self.previous_signal = signal
            return signal
        except Exception:
            return "hold"

# ------------------------------------------------------------------
# WeeklyMomentumStrategy: 주간 모멘텀 지표를 활용한 거래 전략
# ------------------------------------------------------------------
class WeeklyMomentumStrategy(BaseStrategy):
    def __init__(self):
        """
        WeeklyMomentumStrategy 생성자.
        
        목적:
          - 주간 모멘텀을 분석하여 거래 진입 또는 청산 신호를 결정합니다.
        """
        super().__init__()
        self.previous_signal = None

    def get_signal(self, data_weekly, current_time, momentum_threshold=0.5, **kwargs):
        """
        주간 모멘텀 지표를 기반으로 거래 신호를 생성합니다.
        
        Parameters:
            data_weekly (pandas.DataFrame): 주간 시장 데이터 (모멘텀 포함).
            current_time (datetime): 기준 시점.
            momentum_threshold (float): 모멘텀 임계값.
            **kwargs: 추가 인자들.
        
        Returns:
            str: 모멘텀이 임계값 이상이면 "enter_long", 이하이면 "exit_all", 그 외는 "hold".
        """
        try:
            weekly_data = data_weekly.loc[data_weekly.index <= current_time]
            if weekly_data.empty:
                return "hold"
            momentum = weekly_data.iloc[-1].get('weekly_momentum')
            if momentum is None:
                return "hold"
            signal = "enter_long" if momentum >= momentum_threshold else ("exit_all" if momentum <= -momentum_threshold else "hold")
            if self.previous_signal != signal:
                self.logger.info(f"WeeklyMomentumStrategy signal changed to {signal} at {current_time}")
                self.previous_signal = signal
            return signal
        except Exception:
            return "hold"

# ------------------------------------------------------------------
# TradingStrategies: 여러 전략(앙상블)을 결합하여 최종 거래 신호를 도출하는 클래스
# ------------------------------------------------------------------
class TradingStrategies:
    def __init__(self):
        """
        TradingStrategies 생성자.
        
        목적:
          - 개별 전략들을 결합한 앙상블 전략을 초기화하고, 최종 거래 신호 도출을 위한 환경을 구성합니다.
        """
        self.logger = setup_logger(self.__class__.__name__)
        from trading.ensemble import Ensemble
        self.ensemble = Ensemble()
        # 앙상블 내 주간 돌파 및 모멘텀 전략 할당
        self.weekly_breakout = self.ensemble.weekly_breakout_strategy
        self.weekly_momentum = self.ensemble.weekly_momentum_strategy

    def get_final_signal(self, market_regime, liquidity_info, data, current_time, data_weekly=None, **kwargs):
        """
        앙상블 전략 및 시장 상황에 따라 최종 거래 신호를 생성합니다.
        
        Parameters:
            market_regime (str): 현재 시장의 추세 ("bullish", "bearish" 등).
            liquidity_info (str): 자산의 유동성 정보.
            data (pandas.DataFrame): 시계열 시장 데이터.
            current_time (datetime): 기준 시점.
            data_weekly (pandas.DataFrame, optional): 주간 시장 데이터.
            **kwargs: 추가 인자들.
        
        Returns:
            str: 최종 거래 신호 ("enter_long", "exit_all", "hold" 등).
        """
        # 앙상블 전략으로부터 기본 신호 도출
        ensemble_signal = self.ensemble.get_final_signal(market_regime, liquidity_info, data, current_time, data_weekly, **kwargs)
        # 시장 상황에 따른 신호 보정:
        # - 하락장(bearish)에서는 모든 신호를 청산("exit_all")으로 강제
        if market_regime == "bearish":
            self.logger.info("Market regime bearish: overriding final signal to exit_all")
            return "exit_all"
        # - 상승장(bullish)에서는 신호가 없으면 최소한 매수("enter_long")로 보정
        elif market_regime == "bullish":
            self.logger.info("Market regime bullish: ensuring signal is at least enter_long")
            return "enter_long" if ensemble_signal == "hold" else ensemble_signal
        return ensemble_signal
