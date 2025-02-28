# markets/regime_filter.py
from logs.logger_config import setup_logger

logger = setup_logger(__name__)

def determine_market_regime(price_data: dict) -> str:
    """
    주어진 가격 데이터를 기반으로 시장 레짐을 결정합니다.
    
    파라미터:
      - price_data (dict): 'current_price'와 'previous_price' 키를 포함해야 함.
    
    반환값:
      - str: 'bullish', 'bearish', 'sideways' 또는 에러 시 'unknown'
    """
    try:
        current_price = price_data.get("current_price")
        previous_price = price_data.get("previous_price")
        if current_price is None or previous_price is None:
            logger.error("필수 가격 데이터 누락: 'current_price' 또는 'previous_price'가 제공되지 않음.", exc_info=True)
            return "unknown"
        
        change_percent = (current_price - previous_price) / previous_price
        
        if change_percent > 0.02:
            regime = "bullish"
        elif change_percent < -0.02:
            regime = "bearish"
        else:
            regime = "sideways"
        
        logger.debug(f"시장 레짐 결정: {regime} (변화율: {change_percent:.2%})")
        return regime
    except Exception as e:
        logger.error(f"시장 레짐 결정 중 에러 발생: {e}", exc_info=True)
        return "unknown"

def filter_regime(price_data: dict, target_regime: str = "bullish") -> bool:
    """
    결정된 시장 레짐이 목표 레짐과 일치하는지 확인합니다.
    
    파라미터:
      - price_data (dict): 가격 데이터
      - target_regime (str): 목표 레짐 (기본 'bullish')
    
    반환값:
      - bool: 목표와 일치하면 True, 아니면 False
    """
    regime = determine_market_regime(price_data)
    match = (regime == target_regime)
    logger.debug(f"레짐 필터링: 목표={target_regime}, 결정={regime}, 일치 여부={match}")
    return match

def determine_weekly_extreme_signal(price_data: dict, weekly_data: dict, threshold: float = 0.002) -> str:
    """
    주어진 가격 데이터와 주간 데이터(weekly_low, weekly_high)를 기반으로,
    주간 저점 매수 혹은 고점 매도 신호를 반환합니다.
    
    파라미터:
      - price_data (dict): 'current_price' 키를 포함해야 함.
      - weekly_data (dict): 'weekly_low'와 'weekly_high' 키를 포함해야 함.
      - threshold (float): 가격이 극값에 근접한 것으로 판단하는 비율 (기본 0.002, 즉 0.2%).
    
    반환값:
      - str: 주간 신호. 'enter_long'이면 주간 저점 매수, 'exit_all'이면 주간 고점 매도, 그 외는 빈 문자열.
    """
    try:
        current_price = price_data.get("current_price")
        weekly_low = weekly_data.get("weekly_low")
        weekly_high = weekly_data.get("weekly_high")
        if current_price is None or weekly_low is None or weekly_high is None:
            logger.error("필수 데이터 누락: 'current_price', 'weekly_low', 'weekly_high' 모두 제공되어야 합니다.", exc_info=True)
            return ""
        
        if abs(current_price - weekly_low) / weekly_low <= threshold:
            logger.debug(f"주간 저점 신호 감지: current_price={current_price}, weekly_low={weekly_low}")
            return "enter_long"
        elif abs(weekly_high - current_price) / weekly_high <= threshold:
            logger.debug(f"주간 고점 신호 감지: current_price={current_price}, weekly_high={weekly_high}")
            return "exit_all"
        return ""
    except Exception as e:
        logger.error(f"주간 극값 신호 결정 중 에러 발생: {e}", exc_info=True)
        return ""
