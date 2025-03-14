# trading/calculators.py

# pandas: 데이터 처리 및 통계 관련 기능 제공
# ta: 기술적 분석(Technical Analysis) 라이브러리, ATR(평균 실제 범위) 계산 등 사용
import pandas as pd
import ta
from logs.logger_config import setup_logger

# 모듈 전반에 걸쳐 디버그 및 에러 로깅을 위한 로거 객체 생성
logger = setup_logger(__name__)

class InvalidEntryPriceError(ValueError):
    """
    거래 진입 가격이 유효하지 않을 경우 발생시키는 사용자 정의 예외.
    예) 음수 혹은 0인 경우.
    """
    pass

def calculate_atr(data: pd.DataFrame, period: int = 14, min_atr: float = None) -> pd.DataFrame:
    """
    주어진 시장 데이터로부터 ATR(Average True Range)을 계산합니다.
    
    데이터 전처리 및 이상치 제거를 포함하여, TA 라이브러리의 AverageTrueRange를 활용해 ATR을 산출합니다.
    또한, ATR 값이 특정 최소치(min_atr) 이하로 떨어지지 않도록 보정합니다.
    
    Parameters:
        data (pd.DataFrame): 'high', 'low', 'close' 컬럼을 포함하는 시장 데이터.
        period (int): ATR 계산을 위한 이동 윈도우 기간 (기본값: 14).
        min_atr (float): 최소 ATR 값; 제공되지 않으면 평균 종가의 1%로 설정.
    
    Returns:
        pd.DataFrame: 원본 데이터프레임에 'atr' 컬럼이 추가된 결과.
    """
    # 필수 컬럼들이 데이터에 포함되어 있는지 확인
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # 'high'가 'low'보다 작은 행은 제외하여 데이터의 신뢰성을 확보
    data = data[data['high'] >= data['low']].copy()
    
    # 'close' 값이 'low'보다 낮으면 'low'로, 'high'보다 높으면 'high'로 조정
    data.loc[data['close'] < data['low'], 'close'] = data['low']
    data.loc[data['close'] > data['high'], 'close'] = data['high']
    
    # 각 행의 가격 범위 계산 및 중앙값(typical range) 산출
    range_series = data['high'] - data['low']
    typical_range = range_series.median()
    if typical_range > 0:
        # 범위가 중앙값의 3배를 넘는 이상치 제거
        data = data[range_series <= (3 * typical_range)]
    else:
        logger.debug("Typical range is zero; skipping outlier filtering.")
    
    # 데이터 길이에 따라 효과적인 period 설정
    effective_period = period if len(data) >= period else len(data)
    
    try:
        if effective_period < 1:
            data['atr'] = 0
        elif len(data) < effective_period:
            # 데이터 포인트가 충분하지 않은 경우 단순히 (high - low) 값을 ATR로 사용
            data['atr'] = data['high'] - data['low']
        else:
            # ta 라이브러리를 사용해 Average True Range 계산
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=effective_period,
                fillna=True
            )
            data['atr'] = atr_indicator.average_true_range()
    except Exception as e:
        # 계산 중 오류 발생 시 에러 로그 기록 후, fallback으로 (high - low) 사용
        logger.error("calculate_atr error: " + str(e), exc_info=True)
        data['atr'] = data['high'] - data['low']
    
    # 평균 종가를 기반으로 최소 ATR 값 결정 (기본: 평균 종가의 1%)
    avg_close = data['close'].mean()
    if min_atr is None:
        min_atr = avg_close * 0.01

    # 각 ATR 값이 min_atr 이상이 되도록 보정
    data['atr'] = data['atr'].apply(lambda x: max(x, min_atr))
    
    return data

def calculate_dynamic_stop_and_take(entry_price: float, atr: float, risk_params: dict):
    """
    동적 스톱로스와 테이크프로핏 가격을 계산합니다.
    
    주어진 entry_price와 ATR 값을 기반으로, risk_params에 명시된 매개변수를 활용하여
    스톱로스와 테이크프로핏 수준을 동적으로 산출합니다.
    
    Parameters:
        entry_price (float): 거래 진입 가격.
        atr (float): 시장 변동성을 나타내는 ATR 값.
        risk_params (dict): 리스크 파라미터 사전, 포함 항목:
            - "atr_multiplier": ATR에 적용되는 곱셈 인자.
            - "volatility_multiplier": 변동성 조정을 위한 추가 인자.
            - "risk_reward_ratio": 스톱로스 대비 테이크프로핏의 비율.
            - "fallback_atr": ATR 값이 유효하지 않을 때 사용할 대체 ATR.
    
    Returns:
        tuple: (stop_loss_price, take_profit_price)
    """
    # 유효한 진입 가격 확인: 0 이하이면 예외 발생
    if entry_price <= 0:
        logger.error(f"Invalid entry_price: {entry_price}. Must be positive.", exc_info=True)
        raise InvalidEntryPriceError(f"Invalid entry_price: {entry_price}. Must be positive.")
    # ATR 값이 0 이하일 경우 risk_params에 있는 fallback ATR 사용 또는 기본값 할당
    if atr <= 0:
        logger.error(f"ATR value is non-positive ({atr}). Using fallback ATR value from risk_params if available.", exc_info=True)
        fallback_atr = risk_params.get("fallback_atr", entry_price * 0.01)
        if fallback_atr <= 0:
            fallback_atr = entry_price * 0.01
        atr = fallback_atr
    try:
        # risk_params에서 각 인자들을 가져오되, 기본값 사용
        atr_multiplier = risk_params.get("atr_multiplier", 2.0)
        volatility_multiplier = risk_params.get("volatility_multiplier", 1.0)
        risk_reward_ratio = risk_params.get("risk_reward_ratio", 2.0)
        
        # 인자들의 값을 1.0~5.0 범위로 제한
        atr_multiplier = max(1.0, min(atr_multiplier, 5.0))
        risk_reward_ratio = max(1.0, min(risk_reward_ratio, 5.0))
        
        # 스톱로스 가격 계산: entry_price에서 ATR, atr_multiplier, volatility_multiplier의 곱만큼 차감
        stop_loss_price = entry_price - (atr * atr_multiplier * volatility_multiplier)
        if stop_loss_price <= 0:
            logger.error("Computed stop_loss_price is non-positive; adjusting to at least 50% of entry_price.", exc_info=True)
            stop_loss_price = entry_price * 0.5
        
        # 테이크프로핏 가격 계산: 스톱로스와의 차이를 risk_reward_ratio로 확장
        take_profit_price = entry_price + (entry_price - stop_loss_price) * risk_reward_ratio
        
        logger.debug(f"Calculated stop_loss={stop_loss_price:.2f}, take_profit={take_profit_price:.2f} "
                     f"(entry_price={entry_price}, atr={atr}, atr_multiplier={atr_multiplier}, "
                     f"volatility_multiplier={volatility_multiplier}, risk_reward_ratio={risk_reward_ratio})")
        return stop_loss_price, take_profit_price
    except Exception as e:
        logger.error("calculate_dynamic_stop_and_take error: " + str(e), exc_info=True)
        raise

def calculate_partial_exit_targets(entry_price: float, partial_exit_ratio: float = 0.5,
                                     partial_profit_ratio: float = 0.03, final_profit_ratio: float = 0.06,
                                     final_exit_ratio: float = 1.0, use_weekly_target: bool = False,
                                     weekly_momentum: float = None, weekly_adjustment_factor: float = 0.5):
    """
    부분 청산 타겟 가격들을 계산합니다.
    
    거래 진입 가격을 기반으로 부분 청산과 최종 청산 목표 가격을 산출하며,
    옵션으로 주간 모멘텀(weekly momentum)을 활용해 목표 수익률을 조정할 수 있습니다.
    
    Parameters:
        entry_price (float): 거래 진입 가격.
        partial_exit_ratio (float): 부분 청산 시 청산할 포지션 비율.
        partial_profit_ratio (float): 부분 청산 목표 수익률.
        final_profit_ratio (float): 최종 청산 목표 수익률.
        final_exit_ratio (float): 최종 청산 시 청산할 포지션 비율.
        use_weekly_target (bool): 주간 모멘텀을 적용할지 여부.
        weekly_momentum (float, optional): 주간 모멘텀 값.
        weekly_adjustment_factor (float): 주간 모멘텀에 따른 조정 인자.
    
    Returns:
        list: [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]
              각 튜플은 목표 가격과 해당 가격에서 청산할 비율을 의미.
    """
    # 유효한 entry_price 확인
    if entry_price <= 0:
        logger.error(f"Invalid entry_price: {entry_price}. Must be positive.", exc_info=True)
        raise InvalidEntryPriceError(f"Invalid entry_price: {entry_price}. Must be positive.")
    try:
        # 주간 모멘텀이 적용되는 경우, 목표 수익률 조정
        if use_weekly_target and weekly_momentum is not None:
            adjusted_partial = partial_profit_ratio + weekly_adjustment_factor * weekly_momentum
            adjusted_final = final_profit_ratio + weekly_adjustment_factor * weekly_momentum
        else:
            adjusted_partial = partial_profit_ratio
            adjusted_final = final_profit_ratio
        # 목표 가격 계산: entry_price에 (1 + 목표 수익률)을 곱함
        partial_target = entry_price * (1 + adjusted_partial)
        final_target = entry_price * (1 + adjusted_final)
        logger.debug(f"Partial targets: partial={partial_target:.2f}, final={final_target:.2f} "
                     f"(entry_price={entry_price}, adjusted_partial={adjusted_partial}, adjusted_final={adjusted_final})")
        return [(partial_target, partial_exit_ratio), (final_target, final_exit_ratio)]
    except Exception as e:
        logger.error("calculate_partial_exit_targets error: " + str(e), exc_info=True)
        raise

def adjust_trailing_stop(current_stop: float, current_price: float, highest_price: float, trailing_percentage: float,
                           volatility: float = 0.0, weekly_high: float = None, weekly_volatility: float = None) -> float:
    """
    현재 가격 및 변동성을 고려하여 트레일링 스톱을 조정합니다.
    
    인트라데이와 주간 데이터를 모두 활용해, 현재까지의 최고 가격을 기준으로
    새로운 후보 스톱 가격을 산출하고, 이 후보가 기존 스톱보다 개선된 경우에만 업데이트합니다.
    
    Parameters:
        current_stop (float): 현재 설정된 트레일링 스톱 가격.
        current_price (float): 현재 시장 가격.
        highest_price (float): 거래 진입 이후 도달한 최고 가격.
        trailing_percentage (float): 트레일링 스톱 계산에 사용되는 기본 비율.
        volatility (float): 인트라데이 변동성 (기본값: 0.0).
        weekly_high (float, optional): 주간 최고 가격.
        weekly_volatility (float, optional): 주간 변동성.
    
    Returns:
        float: 조정된 트레일링 스톱 가격.
    """
    # current_price와 highest_price가 양수인지 확인
    if current_price <= 0 or highest_price <= 0:
        logger.error(f"Invalid current_price ({current_price}) or highest_price ({highest_price}).", exc_info=True)
        raise ValueError("current_price and highest_price must be positive.")
    # trailing_percentage가 음수이면 에러 발생
    if trailing_percentage < 0:
        logger.error(f"Invalid trailing_percentage ({trailing_percentage}). Must be non-negative.", exc_info=True)
        raise ValueError("trailing_percentage must be non-negative.")
    try:
        # current_stop이 설정되지 않았거나 음수일 경우, 최고가를 기준으로 초기화
        if current_stop is None or current_stop <= 0:
            current_stop = highest_price * (1 - trailing_percentage * (1 + volatility))
        # 인트라데이 기반 후보 스톱 계산
        new_stop_intraday = highest_price * (1 - trailing_percentage * (1 + volatility))
        if weekly_high is not None:
            # 주간 데이터가 제공되면, 주간 후보 스톱 계산 및 두 후보 중 높은 값을 선택
            w_vol = weekly_volatility if weekly_volatility is not None else 0.0
            new_stop_weekly = weekly_high * (1 - trailing_percentage * (1 + w_vol))
            candidate_stop = max(new_stop_intraday, new_stop_weekly)
        else:
            candidate_stop = new_stop_intraday
        # 후보 스톱이 개선되고 현재 가격 이하일 때만 업데이트
        adjusted_stop = candidate_stop if candidate_stop > current_stop and candidate_stop < current_price else current_stop
        logger.debug(f"Adjusted trailing stop: {adjusted_stop:.2f} (current_stop={current_stop}, candidate_stop={candidate_stop}, current_price={current_price})")
        return adjusted_stop
    except Exception as e:
        logger.error("adjust_trailing_stop error: " + str(e), exc_info=True)
        raise
