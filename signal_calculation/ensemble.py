# trading/ensemble.py

# 로깅 설정
from logging.logger_config import setup_logger
# 다양한 트레이딩 전략 임포트:
# SelectStrategy, TrendFollowingStrategy, BreakoutStrategy, CounterTrendStrategy,
# HighFrequencyStrategy, WeeklyBreakoutStrategy, WeeklyMomentumStrategy
from strategies.trading_strategies import (
    SelectStrategy, TrendFollowingStrategy, BreakoutStrategy,
    CounterTrendStrategy, HighFrequencyStrategy, WeeklyBreakoutStrategy, WeeklyMomentumStrategy
)
# 설정 관리: 동적 파라미터를 가져오기 위해 사용
from parameter_management.config_manager import ConfigManager

def compute_dynamic_weights(market_volatility: float, liquidity_info: str, volume: float = None):
    """
    시장 변동성, 유동성 정보 및 거래량을 고려하여 단기 전략과 주간 전략에 적용할 동적 가중치를 계산합니다.
    
    Parameters:
        market_volatility (float): 현재 시장 변동성 (없을 경우 기본값 사용).
        liquidity_info (str): 유동성 정보, "high"일 경우 높은 유동성으로 간주.
        volume (float, optional): 거래량; 낮은 경우 추가 조정.
    
    Returns:
        tuple: (short_weight, weekly_weight)
            - short_weight: 단기 전략에 적용할 가중치.
            - weekly_weight: 주간 전략에 적용할 가중치.
    """
    # market_volatility가 None이면 기본값 설정
    if market_volatility is None:
        market_volatility = 0.02

    # 설정 관리자를 통해 동적 파라미터를 불러옴
    config = ConfigManager().get_dynamic_params()
    # 유동성이 높은 경우와 그렇지 않은 경우에 따라 기본 가중치 할당
    if liquidity_info.lower() == "high":
        short_weight = config.get("liquidity_weight_high", 0.8)
        weekly_weight = 1 - short_weight
    else:
        short_weight = config.get("liquidity_weight_low", 0.6)
        weekly_weight = 1 - short_weight

    # 시장 변동성이 특정 임계값을 넘으면 가중치 조정
    vol_threshold = config.get("weight_vol_threshold", 0.05)
    if market_volatility > vol_threshold:
        factor = config.get("vol_weight_factor", 0.9)
        short_weight *= factor
        weekly_weight = 1 - short_weight

    # 거래량(volume)이 주어지고 낮은 경우 추가 가중치 조정
    if volume is not None and volume < 1000:
        short_weight *= 0.8
        weekly_weight = 1 - short_weight

    return short_weight, weekly_weight

class Ensemble:
    def __init__(self):
        """
        다양한 트레이딩 전략들을 집계(ensemble)하여 최종 거래 신호를 도출하기 위한 Ensemble 객체를 초기화합니다.
        
        각 하위 전략(Select, TrendFollowing, Breakout, CounterTrend, HighFrequency, WeeklyBreakout, WeeklyMomentum)
        인스턴스를 생성하고, 로깅을 위한 로거 객체를 설정합니다.
        """
        self.logger = setup_logger(__name__)
        self.select_strategy = SelectStrategy()
        self.trend_following_strategy = TrendFollowingStrategy()
        self.breakout_strategy = BreakoutStrategy()
        self.counter_trend_strategy = CounterTrendStrategy()
        self.high_frequency_strategy = HighFrequencyStrategy()
        self.weekly_breakout_strategy = WeeklyBreakoutStrategy()
        self.weekly_momentum_strategy = WeeklyMomentumStrategy()
        # 마지막 최종 신호를 저장하여 변경 시 로깅에 활용
        self.last_final_signal = None

    def get_final_signal(self, market_regime, liquidity_info, data, current_time, data_weekly=None, 
                         market_volatility: float = None, volume: float = None):
        """
        여러 하위 전략들로부터 개별 거래 신호를 수집한 후, 동적 가중치 기반 투표 방식을 통해 최종 거래 신호를 결정합니다.
        
        1. 동적 가중치(short_weight, weekly_weight)를 계산.
        2. 단기 전략(Select, Trend, Breakout, Counter, HF) 및 (옵션) 주간 전략에서 신호를 수집.
        3. 각 신호에 대해 가중치 합산을 통해 'enter_long'와 'exit_all' 중 우세한 신호를 결정.
        4. 투표 결과에 따라 최종 신호("enter_long", "exit_all", 또는 "hold")를 반환.
        
        Parameters:
            market_regime (str): 현재 시장 상황(직접 사용되지는 않음).
            liquidity_info (str): 시장 유동성 정보 (예: "high").
            data: 단기 전략에 필요한 시장 데이터.
            current_time: 신호 산출 시점의 타임스탬프.
            data_weekly: 주간 전략에 사용할 주간 시장 데이터 (옵션).
            market_volatility (float, optional): 시장 변동성 수치.
            volume (float, optional): 거래량 수치.
        
        Returns:
            str: 최종 거래 신호 ("enter_long", "exit_all", 또는 "hold").
        """
        # 동적 가중치 계산: 단기와 주간 전략의 영향력 비율 결정
        short_weight, weekly_weight = compute_dynamic_weights(market_volatility, liquidity_info, volume)

        # 각 하위 전략에서 신호 수집 (단기 데이터 기반)
        signals = {
            "select": self.select_strategy.get_signal(data, current_time),
            "trend": self.trend_following_strategy.get_signal(data, current_time),
            "breakout": self.breakout_strategy.get_signal(data, current_time),
            "counter": self.counter_trend_strategy.get_signal(data, current_time),
            "hf": self.high_frequency_strategy.get_signal(data, current_time)
        }
        # 주간 데이터가 제공되면 주간 전략 신호 추가
        if data_weekly is not None:
            signals["weekly_breakout"] = self.weekly_breakout_strategy.get_signal(data_weekly, current_time)
            signals["weekly_momentum"] = self.weekly_momentum_strategy.get_signal(data_weekly, current_time)

        # 단기 전략 신호에 대해 'enter_long'와 'exit_all' 가중치 합산
        vote_enter = sum(short_weight for key in ["select", "trend", "breakout", "counter", "hf"] if signals.get(key) == "enter_long")
        vote_exit = sum(short_weight for key in ["select", "trend", "breakout", "counter", "hf"] if signals.get(key) == "exit_all")
        # 주간 전략 신호가 있으면 해당 가중치도 포함
        if data_weekly is not None:
            vote_enter += sum(weekly_weight for key in ["weekly_breakout", "weekly_momentum"] if signals.get(key) == "enter_long")
            vote_exit += sum(weekly_weight for key in ["weekly_breakout", "weekly_momentum"] if signals.get(key) == "exit_all")

        # 가중치 투표 결과에 따라 최종 신호 결정:
        # - exit_all이 더 많은 경우: "exit_all"
        # - enter_long이 더 많은 경우: "enter_long"
        # - 동점이면 "hold"
        final_signal = "exit_all" if vote_exit > vote_enter else ("enter_long" if vote_enter > vote_exit else "hold")
        # 최종 신호가 변경되었으면 디버그 로그 기록
        if self.last_final_signal != final_signal:
            self.logger.debug(
                f"Ensemble final signal changed to {final_signal} at {current_time} "
                f"with dynamic weights: short={short_weight}, weekly={weekly_weight}, "
                f"signals: {signals}"
            )
            self.last_final_signal = final_signal
        return final_signal
