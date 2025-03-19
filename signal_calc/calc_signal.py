# signal_calc/calc_signal.py

# 이 파일은 모든 트레이딩 전략 클래스들이 상속할 기본 전략(BaseStrategy)을 정의합니다.
# 각 전략 클래스는 get_signal() 메서드를 구현하여 거래 신호를 생성해야 합니다.

from logs.log_config import setup_logger  # 로깅 설정을 위한 함수 임포트

class BaseStrategy:
    def __init__(self):
        """
        기본 전략 클래스의 생성자.
        
        목적:
          - 자식 클래스에서 사용할 로거(logger) 객체를 초기화합니다.
        
        동작:
          - 클래스 이름을 이용해 로거를 설정함으로써, 로그 메시지에 전략 이름을 포함시킵니다.
        """
        self.logger = setup_logger(self.__class__.__name__)
    
    def get_signal(self, data, current_time, **kwargs):
        """
        거래 신호를 생성하기 위한 추상 메서드.
        
        Parameters:
            data (pandas.DataFrame): 거래 데이터 (예: OHLCV 데이터 등).
            current_time (datetime): 거래 신호를 생성할 기준 시점.
            **kwargs: 추가 인자들.
        
        Returns:
            str: 거래 신호 (예: "enter_long", "exit_all", "hold").
        
        주의:
            - 이 메서드는 구현되어 있지 않으므로 반드시 자식 클래스에서 오버라이딩 해야 합니다.
        """
        raise NotImplementedError("Subclasses must implement get_signal()")

# 로깅 설정
from logs.log_config import setup_logger
# 다양한 트레이딩 전략 임포트:
# SelectStrategy, TrendFollowingStrategy, BreakoutStrategy, CounterTrendStrategy,
# HighFrequencyStrategy, WeeklyBreakoutStrategy, WeeklyMomentumStrategy
from strategies.trading_strategies import (
    SelectStrategy, TrendFollowingStrategy, BreakoutStrategy,
    CounterTrendStrategy, HighFrequencyStrategy, WeeklyBreakoutStrategy, WeeklyMomentumStrategy
)
# 설정 관리: 동적 파라미터를 가져오기 위해 사용
from parameters_sensitivity.config_manager import ConfigManager

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

import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from logs.log_config import setup_logger

# 전역 변수 및 객체 정의
# logger: 이 모듈에서 발생하는 디버그 및 에러 메시지를 기록하는 로깅 객체입니다.
logger = setup_logger(__name__)


def compute_sma(data: pd.DataFrame, price_column: str = 'close', period: int = 20, fillna: bool = False, output_col: str = 'sma') -> pd.DataFrame:
    """
    주어진 데이터프레임의 지정된 가격 열을 기준으로 단순 이동평균(SMA)을 계산합니다.
    
    Parameters:
      - data (pd.DataFrame): 가격 정보가 포함된 입력 데이터프레임.
      - price_column (str): SMA 계산에 사용할 가격 열 이름 (예: 'close').
      - period (int): SMA 계산에 사용될 기간 (예: 20일).
      - fillna (bool): 결측치가 있을 경우 보정 여부.
      - output_col (str): 계산된 SMA 결과를 저장할 컬럼 이름.
    
    Returns:
      - pd.DataFrame: 입력 데이터프레임에 SMA 컬럼이 추가된 결과 데이터프레임.
    
    동작 방식:
      - ta 라이브러리의 SMAIndicator 클래스를 사용하여 주어진 기간의 단순 이동평균을 계산합니다.
      - 계산된 결과를 데이터프레임의 새로운 컬럼(output_col)에 추가합니다.
    """
    try:
        # SMAIndicator 객체를 생성하여 지정된 가격 열과 기간에 따른 SMA를 계산하도록 설정합니다.
        sma = SMAIndicator(close=data[price_column], window=period, fillna=fillna)
        # 계산된 SMA 값을 output_col 이름의 컬럼에 저장합니다.
        data[output_col] = sma.sma_indicator()
        logger.debug(f"SMA computed with period {period}")
    except Exception as e:
        # 예외 발생 시 에러 로그에 기록합니다.
        logger.error(f"compute_sma error: {e}", exc_info=True)
    return data


def compute_macd(data: pd.DataFrame, price_column: str = 'close', slow_period: int = 26, fast_period: int = 12, signal_period: int = 9, fillna: bool = False, prefix: str = 'macd_') -> pd.DataFrame:
    """
    주어진 데이터프레임의 가격 정보를 바탕으로 MACD (이동평균 수렴·발산 지표)를 계산합니다.
    
    Parameters:
      - data (pd.DataFrame): 가격 정보가 포함된 입력 데이터프레임.
      - price_column (str): MACD 계산에 사용할 가격 열 이름.
      - slow_period (int): 느린 이동평균에 사용되는 기간.
      - fast_period (int): 빠른 이동평균에 사용되는 기간.
      - signal_period (int): 시그널 라인 계산에 사용되는 기간.
      - fillna (bool): 결측치 보정 여부.
      - prefix (str): 결과 컬럼 이름에 붙일 접두사.
    
    Returns:
      - pd.DataFrame: MACD, 시그널 라인, 그리고 두 값의 차이(diff)를 포함하는 컬럼들이 추가된 데이터프레임.
    
    동작 방식:
      - ta 라이브러리의 MACD 클래스를 활용해 MACD 관련 값을 계산하고,
        각 결과를 접두사(prefix)를 붙여 데이터프레임에 저장합니다.
    """
    try:
        # MACD 객체 생성: 지정된 가격 열과 기간 값들을 이용하여 MACD 계산을 설정합니다.
        macd = MACD(close=data[price_column],
                    window_slow=slow_period,
                    window_fast=fast_period,
                    window_sign=signal_period,
                    fillna=fillna)
        # MACD 관련 값을 각각 새로운 컬럼에 저장합니다.
        data[f'{prefix}macd'] = macd.macd()
        data[f'{prefix}signal'] = macd.macd_signal()
        data[f'{prefix}diff'] = macd.macd_diff()
        logger.debug(f"MACD computed (slow={slow_period}, fast={fast_period}, signal={signal_period})")
    except Exception as e:
        logger.error(f"compute_macd error: {e}", exc_info=True)
    return data


def compute_rsi(data: pd.DataFrame, price_column: str = 'close', period: int = 14, fillna: bool = False, output_col: str = 'rsi') -> pd.DataFrame:
    """
    주어진 데이터프레임의 가격 정보를 바탕으로 상대 강도 지수(RSI)를 계산합니다.
    
    Parameters:
      - data (pd.DataFrame): 가격 정보가 포함된 입력 데이터프레임.
      - price_column (str): RSI 계산에 사용할 가격 열 이름.
      - period (int): RSI 계산에 사용될 기간.
      - fillna (bool): 결측치 보정 여부.
      - output_col (str): 계산된 RSI 결과를 저장할 컬럼 이름.
    
    Returns:
      - pd.DataFrame: RSI 컬럼이 추가된 데이터프레임.
    
    동작 방식:
      - ta 라이브러리의 RSIIndicator 클래스를 사용하여 주어진 기간 동안의 RSI를 계산하고,
        결과를 데이터프레임에 추가합니다.
    """
    try:
        # RSIIndicator 객체 생성: 지정된 가격 열과 기간에 따른 RSI를 계산하도록 설정합니다.
        rsi = RSIIndicator(close=data[price_column], window=period, fillna=fillna)
        # 계산된 RSI 값을 output_col 이름의 컬럼에 저장합니다.
        data[output_col] = rsi.rsi()
        logger.debug(f"RSI computed with period {period}")
    except Exception as e:
        logger.error(f"compute_rsi error: {e}", exc_info=True)
    return data


def compute_bollinger_bands(data: pd.DataFrame, price_column: str = 'close', period: int = 20, std_multiplier: float = 2.0, fillna: bool = False, prefix: str = 'bb_') -> pd.DataFrame:
    """
    주어진 가격 데이터를 바탕으로 Bollinger Bands(볼린저 밴드) 및 관련 지표들을 계산합니다.
    
    Parameters:
      - data (pd.DataFrame): 가격 정보가 포함된 입력 데이터프레임.
      - price_column (str): Bollinger Bands 계산에 사용할 가격 열 이름.
      - period (int): 중간 이동평균 계산에 사용되는 기간.
      - std_multiplier (float): 표준편차 배수 (상한/하한 밴드 계산에 사용).
      - fillna (bool): 결측치 보정 여부.
      - prefix (str): 결과 컬럼 이름에 붙일 접두사.
    
    Returns:
      - pd.DataFrame: 중간 이동평균, 상한/하한 밴드, 퍼센트 밴드, 폭 밴드 및 밴드 지표들을 포함하는 컬럼들이 추가된 데이터프레임.
    
    동작 방식:
      - ta 라이브러리의 BollingerBands 클래스를 사용하여 각 밴드와 지표들을 계산한 후,
        계산 결과를 각각의 컬럼에 저장합니다.
    """
    try:
        # BollingerBands 객체 생성: 지정된 가격 열, 기간, 표준편차 배수를 이용하여 볼린저 밴드를 계산하도록 설정합니다.
        bb = BollingerBands(close=data[price_column], window=period, window_dev=std_multiplier, fillna=fillna)
        # 계산된 각 지표들을 접두사(prefix)를 붙여 데이터프레임에 저장합니다.
        data[f'{prefix}mavg'] = bb.bollinger_mavg()
        data[f'{prefix}hband'] = bb.bollinger_hband()
        data[f'{prefix}lband'] = bb.bollinger_lband()
        data[f'{prefix}pband'] = bb.bollinger_pband()
        data[f'{prefix}wband'] = bb.bollinger_wband()
        data[f'{prefix}hband_ind'] = bb.bollinger_hband_indicator()
        data[f'{prefix}lband_ind'] = bb.bollinger_lband_indicator()
        logger.debug(f"Bollinger Bands computed (period={period}, std_multiplier={std_multiplier})")
    except Exception as e:
        logger.error(f"compute_bollinger_bands error: {e}", exc_info=True)
    return data

from logs.log_config import setup_logger  # 로거 설정 함수
from strategies.base_strategy import BaseStrategy  # 기본 전략 클래스
from market_analysis.regime_filter import determine_weekly_extreme_signal  # 주간 극단값 신호를 판단하는 함수

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
        from signal_calculation.ensemble import Ensemble
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
