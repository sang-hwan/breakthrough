# signal_calc/calc_signal.py
from logs.log_config import setup_logger
from parameters.signal_parameters import ConfigManager
from datetime import datetime

# 하위 전략 모듈에서 상승, 횡보, 하락 전략 임포트
from signal_calc.signals_uptrend import TrendFollowingStrategy, BreakoutStrategy, WeeklyMomentumStrategy
from signal_calc.signals_sideways import RangeTradingStrategy
from signal_calc.signals_downtrend import CounterTrendStrategy, HighFrequencyStrategy

logger = setup_logger(__name__)

class BaseStrategy:
    """
    기본 전략 클래스.
    
    모든 개별 전략은 이 클래스를 상속받아 get_signal() 메서드를 구현해야 합니다.
    """
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def get_signal(self, data, current_time, **kwargs):
        """
        거래 신호 생성 추상 메서드.
        
        Parameters:
            data (pandas.DataFrame): 시장 데이터 (예: OHLCV, 기술 지표 등).
            current_time (datetime): 신호 산출 기준 시점.
            **kwargs: 추가 인자.
            
        Returns:
            str: "enter_long", "exit_all", "hold" 중 하나.
        """
        raise NotImplementedError("Subclasses must implement get_signal()")

def compute_dynamic_weights(market_volatility: float, liquidity_info: str, volume: float = None):
    """
    동적 가중치 계산 함수.
    
    시장 변동성, 유동성 및 거래량을 고려하여 전략별 가중치를 산출합니다.
    
    Parameters:
        market_volatility (float): 시장 변동성 수치.
        liquidity_info (str): 유동성 정보 ("high" 또는 "low").
        volume (float, optional): 거래량.
        
    Returns:
        tuple: (short_weight, weekly_weight)
    """
    # 기본값 설정
    if market_volatility is None:
        market_volatility = 0.02

    config = ConfigManager().get_dynamic_params()
    if liquidity_info.lower() == "high":
        short_weight = config.get("liquidity_weight_high", 0.8)
    else:
        short_weight = config.get("liquidity_weight_low", 0.6)
    weekly_weight = 1 - short_weight

    vol_threshold = config.get("weight_vol_threshold", 0.05)
    if market_volatility > vol_threshold:
        factor = config.get("vol_weight_factor", 0.9)
        short_weight *= factor
        weekly_weight = 1 - short_weight

    if volume is not None and volume < 1000:
        short_weight *= 0.8
        weekly_weight = 1 - short_weight

    return short_weight, weekly_weight

class Ensemble:
    """
    앙상블 전략 클래스.
    
    상승장, 횡보장, 하락장 전략들을 집계하여 최종 거래 신호를 도출합니다.
    """
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        # 상승장 전략 인스턴스
        self.trend_following = TrendFollowingStrategy()
        self.breakout = BreakoutStrategy()
        self.weekly_momentum = WeeklyMomentumStrategy()
        # 횡보장 전략 인스턴스
        self.range_trading = RangeTradingStrategy()
        # 하락장 전략 인스턴스
        self.counter_trend = CounterTrendStrategy()
        self.high_frequency = HighFrequencyStrategy()
        # 마지막 최종 신호 저장 (변경 시 로깅에 활용)
        self.last_final_signal = None

    def get_final_signal(self, market_regime: str, liquidity_info: str, data, current_time: datetime,
                         data_weekly=None, market_volatility: float = None, volume: float = None):
        """
        최종 거래 신호 도출 함수.
        
        여러 하위 전략의 신호를 동적 가중치 기반으로 집계합니다.
        시장 상황에 따라 기본 신호를 보정하며, 과적합 방지를 위해 외부 파라미터를 활용합니다.
        
        Parameters:
            market_regime (str): 시장 상황 ("bullish", "bearish", "sideways").
            liquidity_info (str): 유동성 정보.
            data: 단기 시장 데이터.
            current_time (datetime): 기준 시점.
            data_weekly: 주간 시장 데이터 (옵션).
            market_volatility (float, optional): 시장 변동성.
            volume (float, optional): 거래량.
            
        Returns:
            str: 최종 거래 신호 ("enter_long", "exit_all", 또는 "hold").
        """
        short_weight, weekly_weight = compute_dynamic_weights(market_volatility, liquidity_info, volume)
        
        # 각 하위 전략으로부터 신호 수집
        signals = {}
        # 상승장 전략 신호
        signals["trend"] = self.trend_following.get_signal(data, current_time)
        signals["breakout"] = self.breakout.get_signal(data, current_time)
        if data_weekly is not None:
            signals["weekly_momentum"] = self.weekly_momentum.get_signal(data_weekly, current_time)
        # 횡보장 전략 신호
        signals["range"] = self.range_trading.get_signal(data, current_time)
        # 하락장 전략 신호
        signals["counter"] = self.counter_trend.get_signal(data, current_time)
        signals["high_freq"] = self.high_frequency.get_signal(data, current_time)
        
        # 가중치 기반 투표 (상승/횡보 전략은 "enter_long", 하락 전략은 "exit_all" 신호로 평가)
        vote_enter = 0
        vote_exit = 0
        
        # 상승장 신호 평가
        for key in ["trend", "breakout"]:
            if signals.get(key) == "enter_long":
                vote_enter += short_weight
            elif signals.get(key) == "exit_all":
                vote_exit += short_weight
        
        if "weekly_momentum" in signals:
            if signals.get("weekly_momentum") == "enter_long":
                vote_enter += weekly_weight
            elif signals.get("weekly_momentum") == "exit_all":
                vote_exit += weekly_weight
        
        # 횡보장 신호: 중립적 특성으로 0.5 * short_weight 적용
        if signals.get("range") == "enter_long":
            vote_enter += 0.5 * short_weight
        elif signals.get("range") == "exit_all":
            vote_exit += 0.5 * short_weight
        
        # 하락장 신호 평가
        for key in ["counter", "high_freq"]:
            if signals.get(key) == "exit_all":
                vote_exit += short_weight
            elif signals.get(key) == "enter_long":
                vote_enter += short_weight
        
        # 가중치 투표 결과에 따라 최종 신호 결정
        final_signal = "exit_all" if vote_exit > vote_enter else ("enter_long" if vote_enter > vote_exit else "hold")
        
        # 시장 상황에 따른 보정: bearish일 경우 강제 청산, bullish에서 신호가 없으면 기본 매수
        if market_regime == "bearish":
            final_signal = "exit_all"
        elif market_regime == "bullish" and final_signal == "hold":
            final_signal = "enter_long"
        
        if self.last_final_signal != final_signal:
            self.logger.debug(
                f"Final signal changed to {final_signal} at {current_time}, "
                f"signals: {signals}, votes -> enter: {vote_enter}, exit: {vote_exit}"
            )
            self.last_final_signal = final_signal
        
        return final_signal
